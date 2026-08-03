#!/usr/bin/env python
"""
Plot turbulence statistics from computed statistics file.

Usage:
    python plot_statistics.py <stats_file> [options]

Arguments:
    stats_file: Path to .npz statistics file (required)

Options:
    --Re REYNOLDS_NUMBER     Reynolds number (optional, overrides value in stats file)
    --nu VISCOSITY           Kinematic viscosity (optional, overrides value in stats file)
    --Re_tau RE_TAU          Friction Reynolds number (optional, used to compute u_tau)
    --output OUTPUT_PREFIX   Output file prefix (default: derived from stats_file, saved in same directory)
    --format FORMAT          Output format: pdf, png, or both (default: pdf)
    --dpi DPI                DPI for PNG output (default: 300)
    --checkpoint             Input file is a checkpoint/state file with running sums
    --config CONFIG_FILE     Config file (required when using --checkpoint)

Examples:
    python plot_statistics.py test_stats.npz
    python plot_statistics.py results/my_stats.npz --format both
    python plot_statistics.py test_stats.npz --Re 2000 --Re_tau 180 --output my_plots
    python plot_statistics.py turbulence_stats_state.npz --checkpoint --config config.yaml

Note:
    By default, plots are saved in the same directory as the stats file with the name
    <stats_filename>_plots_<type>.pdf (e.g., test_stats_plots_profiles.pdf)
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import yaml

# LaTeX formatting
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'text.latex.preamble': r'\usepackage{amsmath}'
})


def compute_wall_coordinates(z_c, u_tau, nu):
    """Compute z+ = z * u_tau / nu for wall coordinates."""
    # z_c should already be interior points only (no ghost cells)
    # For channel with Lz = 2, delta = 1, z+ spans from 0 to 2*Re_tau
    z_plus = z_c * u_tau / nu
    return z_plus


def compute_dUdz(U_mean, z_c, Lz, open_channel=False):
    """
    Compute dU/dz from mean velocity profile using correct grid spacing.

    Augments the grid with a point at z=0 where U=0 (no-slip) to improve
    accuracy at the wall, and -- for a CLOSED channel only -- a second one at
    z=Lz.

    Args:
        U_mean: Mean velocity profile at cell centers (nz,)
        z_c: Cell center positions (nz,)
        Lz: Domain height
        open_channel: True for a free-slip/symmetry top (delta = Lz). Then U(Lz)
            is the free-surface velocity, NOT zero, so the top is extrapolated
            from the profile instead of being pinned to zero. Pinning it to zero
            manufactures a huge spurious "wall" gradient at the free surface.

    Returns:
        dUdz: Velocity gradient at cell centers (nz,)
        dUdz_wall_bot: Velocity gradient at bottom wall (z=0)
        dUdz_wall_top: Velocity gradient at z=Lz (meaningless if open_channel)
    """
    nz = len(U_mean)

    # Augment arrays with wall points
    # z_aug = [0, z_c..., Lz]
    # U_aug = [0, U_mean..., U(Lz)]
    if open_channel:
        # Linear extrapolation of U to the free surface from the last two cells.
        if len(z_c) >= 2 and z_c[-1] != z_c[-2]:
            slope = (U_mean[-1] - U_mean[-2]) / (z_c[-1] - z_c[-2])
            U_top = U_mean[-1] + slope * (Lz - z_c[-1])
        else:
            U_top = U_mean[-1]
    else:
        U_top = 0.0

    z_aug = np.concatenate(([0.0], z_c, [Lz]))
    U_aug = np.concatenate(([0.0], U_mean, [U_top]))
    
    n_aug = len(U_aug)
    dUdz_aug = np.zeros(n_aug)

    # 2nd order accurate derivative on non-uniform grid (augmented)
    
    # Interior points of augmented array (indices 1 to n_aug-2)
    # These correspond to the original cell centers z_c
    
    h_minus = z_aug[1:-1] - z_aug[0:-2]
    h_plus = z_aug[2:] - z_aug[1:-1]
    
    c_minus = -h_plus / (h_minus * (h_minus + h_plus))
    c_center = (h_plus - h_minus) / (h_minus * h_plus)
    c_plus = h_minus / (h_plus * (h_minus + h_plus))
    
    dUdz_aug[1:-1] = (c_minus * U_aug[0:-2] + 
                      c_center * U_aug[1:-1] + 
                      c_plus * U_aug[2:])

    # Bottom wall (index 0)
    h0 = z_aug[1] - z_aug[0]
    h1 = z_aug[2] - z_aug[1]
    c0 = -(2*h0 + h1) / (h0 * (h0 + h1))
    c1 = (h0 + h1) / (h0 * h1)
    c2 = -h0 / (h1 * (h0 + h1))
    dUdz_aug[0] = c0 * U_aug[0] + c1 * U_aug[1] + c2 * U_aug[2]

    # Top wall (index -1)
    h_last = z_aug[-1] - z_aug[-2]
    h_prev = z_aug[-2] - z_aug[-3]
    c_last = (2*h_last + h_prev) / (h_last * (h_last + h_prev))
    c_prev = -(h_last + h_prev) / (h_last * h_prev)
    c_prev2 = h_last / (h_prev * (h_last + h_prev))
    dUdz_aug[-1] = c_last * U_aug[-1] + c_prev * U_aug[-2] + c_prev2 * U_aug[-3]

    # Extract results
    dUdz = dUdz_aug[1:-1]  # At cell centers
    dUdz_wall_bot = dUdz_aug[0]
    dUdz_wall_top = dUdz_aug[-1]

    return dUdz, dUdz_wall_bot, dUdz_wall_top


def reconstruct_grid_from_config(config_file, nx, ny, nz):
    """
    Reconstruct grid data (z_c, kx, ky) from config file and grid dimensions.

    Args:
        config_file: Path to configuration YAML file
        nx, ny, nz: Grid dimensions from checkpoint file

    Returns:
        z_c: Cell center positions (nz,) - interior points only
        kx: Wavenumbers in x-direction (nx//2,)
        ky: Wavenumbers in y-direction (ny//2,)
    """
    # Load config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Extract grid parameters
    Lx = config['domain']['Lx']
    Ly = config['domain']['Ly']
    Lz = config['domain']['Lz']
    stretching = config['domain'].get('stretching_type', 'symmetric')

    if stretching == 'double':
        from utils import generate_double_stretched_grid
        _, z_c_t, _, _ = generate_double_stretched_grid(
            config['grid']['nz_canopy'], config['grid']['nz_outer'],
            config['domain']['z_transition'], Lz,
            config['domain'].get('gamma_canopy', 2.0),
            config['domain'].get('gamma_outer', 'auto'))
        z_c = z_c_t[1:-1].cpu().numpy()          # interior points
    elif stretching == 'hybrid':
        from utils import generate_hybrid_grid
        _, z_c_t, _, _ = generate_hybrid_grid(
            config['grid']['nz_uniform'], config['grid']['nz_stretched'],
            config['domain']['z_transition'], Lz,
            config['domain'].get('gamma_stretched', 1.8))
        z_c = z_c_t[1:-1].cpu().numpy()
    else:
        # symmetric tanh (legacy numpy implementation)
        gamma = config['flow']['gamma']
        k = np.linspace(0, nz, nz+1)
        xi = (2 * k / nz) - 1
        z_f_np = 0.5 * Lz * (1 + np.tanh(gamma * xi) / np.tanh(gamma))
        z_c = 0.5 * (z_f_np[:-1] + z_f_np[1:])  # (nz,) interior points


    # Compute wavenumbers
    dx = Lx / nx
    dy = Ly / ny
    kx = 2 * np.pi * np.fft.rfftfreq(nx, d=dx)[1:]  # Skip DC component
    ky = 2 * np.pi * np.fft.rfftfreq(ny, d=dy)[1:]

    return z_c, kx, ky


#: Reference datasets bundled with the package (see scripts/fetch_reference_data.py).
REFERENCE_DATASETS = {
    'mkm180': 'Moser et al. (1999), $Re_\\tau=178$',
    'mkm590': 'Moser et al. (1999), $Re_\\tau=587$',
    'lm550': 'Lee \\& Moser (2015), $Re_\\tau=543$',
}


def load_reference(name):
    """Load a bundled reference DNS profile, or a user-supplied CSV path.

    Returns a dict of numpy arrays keyed z_delta, z_plus, U_plus, uu_plus,
    vv_plus, ww_plus, uw_plus -- already remapped to torChannel's axes (z is
    wall-normal, w is the wall-normal velocity), plus a 'label' for the legend.

    These are CLOSED-channel profiles. Overlaid on an open channel they should
    agree in the near-wall region; a difference toward the centreline is
    physical, not an error -- a symmetry/free-slip top suppresses the
    large-scale motions that cross a closed channel's centreline.
    """
    if name in REFERENCE_DATASETS:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'torchannel', 'data', 'reference', f'{name}.csv')
        label = REFERENCE_DATASETS[name]
        if not os.path.exists(path):
            # Installed rather than in a clone: fall back to package data.
            try:
                import torchannel
                path = os.path.join(os.path.dirname(torchannel.__file__),
                                    'data', 'reference', f'{name}.csv')
            except ImportError:
                pass
    else:
        path = name
        label = os.path.splitext(os.path.basename(path))[0]

    if not os.path.exists(path):
        raise SystemExit(
            f"Reference data not found: {path}\n"
            f"Available bundled datasets: {', '.join(sorted(REFERENCE_DATASETS))}\n"
            f"Regenerate them with: python scripts/fetch_reference_data.py")

    # Parsed explicitly rather than with genfromtxt(names=True): the '#' header
    # block contains commas, which makes genfromtxt mis-detect the column names.
    header, rows = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if header is None:
                header = [c.strip() for c in line.split(',')]
                continue
            rows.append([float(v) for v in line.split(',')])
    if header is None or not rows:
        raise SystemExit(f"Reference file has no data: {path}")

    data = np.asarray(rows, dtype=float)
    ref = {name: data[:, i] for i, name in enumerate(header)}
    ref['label'] = label
    return ref


def plot_mean_velocity(z_c, U_mean, u_tau, nu, ax_outer, ax_inner, ref=None):
    """Plot mean velocity profile in outer and inner coordinates."""
    # Outer coordinates: U vs z
    ax_outer.plot(U_mean, z_c, 'k-', linewidth=1.5)
    ax_outer.set_xlabel(r'$U$')
    ax_outer.set_ylabel(r'$z$')
    ax_outer.grid(True, alpha=0.3)

    # Inner coordinates: U+ vs z+
    z_plus = compute_wall_coordinates(z_c, u_tau, nu)
    U_plus = U_mean / u_tau

    # Asymptotes: viscous sublayer U+ = z+ and the log law with the standard
    # kappa = 0.41, B = 5.2. Drawn first so the data sits on top.
    zp_line = np.logspace(np.log10(0.5), np.log10(max(z_plus.max(), 1.0)), 100)
    ax_inner.semilogx(zp_line, zp_line, ':', color='0.55', linewidth=1.0,
                      label=r'$U^+=z^+$')
    ax_inner.semilogx(zp_line, np.log(zp_line) / 0.41 + 5.2, '--', color='0.55',
                      linewidth=1.0, label=r'$\frac{1}{0.41}\ln z^+ + 5.2$')

    if ref is not None:
        ax_inner.semilogx(ref['z_plus'], ref['U_plus'], 'o', color='tab:red',
                          markersize=3, markerfacecolor='none', markevery=2,
                          linewidth=0, label=ref['label'])

    ax_inner.semilogx(z_plus, U_plus, 'k-', linewidth=1.5, label='torChannel')
    ax_inner.set_xlabel(r'$z^+$')
    ax_inner.set_ylabel(r'$U^+$')
    ax_inner.set_xlim([0.5, z_plus.max()])
    ax_inner.set_ylim([0, max(U_plus.max(), 1.0) * 1.15])
    ax_inner.grid(True, alpha=0.3, which='both')
    ax_inner.legend(fontsize=7, loc='upper left')


def plot_reynolds_stresses_normal(z_c, uu, vv, ww, u_tau, nu, ax, ref=None):
    """Plot normal Reynolds stresses (uu, vv, ww) vs z+."""
    z_plus = compute_wall_coordinates(z_c, u_tau, nu)

    ax.plot(z_plus, uu / u_tau**2, 'r-', linewidth=1.5, label=r"$\langle u'u' \rangle^+$")
    ax.plot(z_plus, vv / u_tau**2, 'g-', linewidth=1.5, label=r"$\langle v'v' \rangle^+$")
    ax.plot(z_plus, ww / u_tau**2, 'b-', linewidth=1.5, label=r"$\langle w'w' \rangle^+$")

    if ref is not None:
        # Same colour per component, open symbols, so the pairing is obvious.
        for key, colour in (('uu_plus', 'r'), ('vv_plus', 'g'), ('ww_plus', 'b')):
            ax.plot(ref['z_plus'], ref[key], 'o', color=colour, markersize=3,
                    markerfacecolor='none', markevery=2, linewidth=0)
        # One legend entry for all three reference series.
        ax.plot([], [], 'o', color='0.35', markersize=3, markerfacecolor='none',
                linewidth=0, label=ref['label'])

    ax.set_xlabel(r'$z^+$')
    ax.set_ylabel(r'Reynolds stress$^+$')
    ax.set_xlim([0, z_plus.max()])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)


def plot_shear_vorticity(z_c, uw, dUdz, u_tau, nu, ax_uw, ax_omega, ref=None):
    """Plot shear stress and mean vorticity (dU/dz) in square subplots.

    The last (topmost) point is dropped: the one-sided dU/dz there depends on
    the top boundary condition (e.g. free-slip) and is not meaningful."""
    z_c, uw, dUdz = z_c[:-1], uw[:-1], dUdz[:-1]
    z_plus = compute_wall_coordinates(z_c, u_tau, nu)

    # Shear stress: -<u'w'> / u_tau^2
    if ref is not None:
        ax_uw.plot(ref['z_plus'], -ref['uw_plus'], 'o', color='tab:red',
                   markersize=3, markerfacecolor='none', markevery=2,
                   linewidth=0, label=ref['label'])
    ax_uw.plot(z_plus, -uw / u_tau**2, 'k-', linewidth=1.5, label='torChannel')
    ax_uw.set_xlabel(r'$z^+$')
    ax_uw.set_ylabel(r"$-\langle u'w' \rangle^+$")
    ax_uw.set_xlim([0, z_plus.max()])
    ax_uw.grid(True, alpha=0.3)
    if ref is not None:
        ax_uw.legend(fontsize=7)

    # Mean vorticity: omega_y ≈ dU/dz (since dW/dx = 0 statistically)
    # Inner scaling: omega_y * nu / u_tau^2
    omega_y_plus = dUdz * nu / u_tau**2
    ax_omega.plot(z_plus, omega_y_plus, 'k-', linewidth=1.5)
    ax_omega.set_xlabel(r'$z^+$')
    ax_omega.set_ylabel(r'$\mathrm{d}U/\mathrm{d}z^+$')
    ax_omega.set_xlim([0, z_plus.max()])
    ax_omega.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    ax_omega.grid(True, alpha=0.3)


def plot_total_stress_decomposition(z_c, uw, dUdz, u_tau, nu, ax):
    """
    Plot total stress with decomposition into Reynolds stress and viscous stress components.

    The total stress in channel flow follows a linear profile: tau_total = 1 - z/delta
    This can be decomposed into:
    - Reynolds stress: -<u'w'>^+ (orange line)
    - Viscous stress: nu * dU/dz / u_tau^2 (teal line)
    - Total stress: sum of the above (thick black line)

    Note: omega_y_mean ≈ dU/dz since dW/dx = 0 statistically in homogeneous flow
    The last (topmost) point is dropped: the one-sided dU/dz there depends on
    the top boundary condition and is not meaningful.
    """
    z_c, uw, dUdz = z_c[:-1], uw[:-1], dUdz[:-1]
    z_plus = compute_wall_coordinates(z_c, u_tau, nu)

    # Reynolds stress component: -<u'w'> / u_tau^2
    reynolds_stress_plus = -uw / u_tau**2

    # Viscous stress component: nu * dU/dz / u_tau^2
    viscous_stress_plus = dUdz * nu / u_tau**2

    # Total stress: sum of Reynolds and viscous components
    total_stress_plus = reynolds_stress_plus + viscous_stress_plus

    # Plot components with smaller linewidth
    ax.plot(z_plus, reynolds_stress_plus, color='orange', linewidth=1.2,
            label=r"$-\langle u'w' \rangle^+$ (Reynolds)")
    ax.plot(z_plus, viscous_stress_plus, color='teal', linewidth=1.2,
            label=r'$\nu \, \mathrm{d}U/\mathrm{d}z^+$ (Viscous)')

    # Plot total stress with thicker line
    ax.plot(z_plus, total_stress_plus, 'k-', linewidth=2.5,
            label=r'Total stress')

    ax.set_xlabel(r'$z^+$')
    ax.set_ylabel(r'Stress$^+$')
    ax.set_xlim([0, z_plus.max()])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', framealpha=0.9)


def plot_2d_spectra_wavelength(kx, ky, premult_uu, premult_vv, premult_ww, premult_uw,
                                ax_uu, ax_vv, ax_ww, ax_uw, u_tau, nu):
    """Plot 2D premultiplied spectra using wavelengths in wall units (lambda+ = lambda * u_tau / nu)."""
    # Convert wavenumbers to wavelengths in physical units
    lambda_x = 2 * np.pi / kx
    lambda_y = 2 * np.pi / ky

    # Convert to wall units: lambda+ = lambda * (u_tau / nu)
    lambda_x_plus = lambda_x * u_tau / nu
    lambda_y_plus = lambda_y * u_tau / nu

    # Create meshgrid for contour plots
    LAMBDA_X_PLUS, LAMBDA_Y_PLUS = np.meshgrid(lambda_x_plus, lambda_y_plus, indexing='ij')

    # Common contour levels (log-spaced from max down to 90% energy threshold)
    def get_levels(data, n_levels=10, energy_fraction=0.80):
        """
        Create logarithmically-spaced contour levels from maximum down to a level
        that retains the specified fraction of total energy.

        Args:
            data: 2D spectral data
            n_levels: Number of contour levels
            energy_fraction: Fraction of energy to retain (default: 0.90 for 90%)
        """
        vmax = data.max()

        # Sort data in descending order to compute cumulative energy
        data_sorted = np.sort(data.flatten())[::-1]
        cumulative_energy = np.cumsum(data_sorted)
        total_energy = cumulative_energy[-1]

        # Find threshold value that captures energy_fraction of total energy
        idx_threshold = np.searchsorted(cumulative_energy, energy_fraction * total_energy)
        vmin_energy = data_sorted[min(idx_threshold, len(data_sorted)-1)]

        # Ensure vmin is positive and not too close to vmax
        vmin = np.maximum(vmin_energy, vmax * 1e-3)
        vmin = np.maximum(vmin, 1e-10)  # Absolute minimum for numerical stability

        return np.logspace(np.log10(vmin), np.log10(vmax), n_levels)

    # Helper to plot peak marker
    def plot_peak(ax, data, x_mesh, y_mesh, find_min=False):
        if find_min:
            # Find global minimum (most negative peak for uw)
            idx = np.unravel_index(np.argmin(data), data.shape)
            marker = 'v'  # Triangle down for minimum
        else:
            # Find global maximum
            idx = np.unravel_index(np.argmax(data), data.shape)
            marker = '^'  # Triangle up for maximum
            
        x_peak = x_mesh[idx]
        y_peak = y_mesh[idx]
        
        # Plot tiny triangle with black edge for visibility
        ax.plot(x_peak, y_peak, marker, color='white', markeredgecolor='black', 
                markeredgewidth=0.5, markersize=6, zorder=10)

    # Plot u'u' premultiplied spectrum
    levels_uu = get_levels(premult_uu)
    ax_uu.contourf(LAMBDA_X_PLUS, LAMBDA_Y_PLUS, premult_uu, levels=levels_uu,
                   norm=mpl.colors.LogNorm(), cmap='hot')
    plot_peak(ax_uu, premult_uu, LAMBDA_X_PLUS, LAMBDA_Y_PLUS)
    ax_uu.set_xlabel(r'$\lambda_x^+$')
    ax_uu.set_ylabel(r'$\lambda_y^+$')
    ax_uu.set_title(r'$k_x k_y E_{uu}(k_x, k_y)$ at $z^+ \approx 15$', fontsize=9)
    ax_uu.set_xscale('log')
    ax_uu.set_yscale('log')

    # Plot v'v' premultiplied spectrum
    levels_vv = get_levels(premult_vv)
    ax_vv.contourf(LAMBDA_X_PLUS, LAMBDA_Y_PLUS, premult_vv, levels=levels_vv,
                   norm=mpl.colors.LogNorm(), cmap='hot')
    plot_peak(ax_vv, premult_vv, LAMBDA_X_PLUS, LAMBDA_Y_PLUS)
    ax_vv.set_xlabel(r'$\lambda_x^+$')
    ax_vv.set_ylabel(r'$\lambda_y^+$')
    ax_vv.set_title(r'$k_x k_y E_{vv}(k_x, k_y)$ at $z^+ \approx 15$', fontsize=9)
    ax_vv.set_xscale('log')
    ax_vv.set_yscale('log')

    # Plot w'w' premultiplied spectrum
    levels_ww = get_levels(premult_ww)
    ax_ww.contourf(LAMBDA_X_PLUS, LAMBDA_Y_PLUS, premult_ww, levels=levels_ww,
                   norm=mpl.colors.LogNorm(), cmap='hot')
    plot_peak(ax_ww, premult_ww, LAMBDA_X_PLUS, LAMBDA_Y_PLUS)
    ax_ww.set_xlabel(r'$\lambda_x^+$')
    ax_ww.set_ylabel(r'$\lambda_y^+$')
    ax_ww.set_title(r'$k_x k_y E_{ww}(k_x, k_y)$ at $z^+ \approx 15$', fontsize=9)
    ax_ww.set_xscale('log')
    ax_ww.set_yscale('log')

    # Plot u'w' premultiplied spectrum (can be negative, use symmetric scale)
    vmax_uw = np.abs(premult_uw).max()
    levels_uw = np.linspace(-vmax_uw, vmax_uw, 20)
    ax_uw.contourf(LAMBDA_X_PLUS, LAMBDA_Y_PLUS, premult_uw, levels=levels_uw, cmap='RdBu_r')
    plot_peak(ax_uw, premult_uw, LAMBDA_X_PLUS, LAMBDA_Y_PLUS, find_min=True)
    ax_uw.set_xlabel(r'$\lambda_x^+$')
    ax_uw.set_ylabel(r'$\lambda_y^+$')
    ax_uw.set_title(r'$k_x k_y E_{uw}(k_x, k_y)$ at $z^+ \approx 15$', fontsize=9)
    ax_uw.set_xscale('log')
    ax_uw.set_yscale('log')


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Plot turbulence statistics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('stats_file', help='Path to .npz statistics file')
    parser.add_argument('--Re', type=float, help='Reynolds number (optional, overrides value in stats file)')
    parser.add_argument('--nu', type=float, help='Kinematic viscosity (optional, overrides value in stats file)')
    parser.add_argument('--Re_tau', type=float, help='Friction Reynolds number (optional, used to compute u_tau)')
    parser.add_argument('--output', default=None, help='Output file prefix (default: derived from stats_file, saved in same directory)')
    parser.add_argument('--format', default='pdf', choices=['pdf', 'png', 'both'],
                       help='Output format (default: pdf)')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for PNG output (default: 300)')
    parser.add_argument('--checkpoint', action='store_true',
                       help='Input file is a checkpoint/state file with running sums (will divide by n_samples)')
    parser.add_argument('--config', default=None, help='Config file (required when using --checkpoint)')
    parser.add_argument('--canopy-height', type=float, default=None, dest='canopy_height',
                       help='Canopy height h (open-channel canopy run): normalize by '
                            'u_tau,out from the TOTAL stress at z=h (Monti et al. 2022 '
                            'convention), report Re_tau,in / Re_tau,out and mark z=h')
    parser.add_argument('--reference', default=None, metavar='NAME|PATH',
                       help='Overlay published DNS profiles. Bundled datasets: '
                            + ', '.join(sorted(REFERENCE_DATASETS))
                            + '. Or give a path to a CSV in the same format '
                              '(see scripts/fetch_reference_data.py). NOTE these '
                              'are CLOSED-channel data: on an open channel expect '
                              'agreement near the wall and a physical difference '
                              'toward the centreline.')

    parser.add_argument('--open-channel', action='store_true', dest='open_channel',
                       help='Free-slip/symmetry top wall (delta = Lz, ONE wall). '
                            'Auto-detected from --config when boundary_conditions.'
                            'top_wall.type is neumann, and from the profile itself; '
                            'this flag forces it.')

    args = parser.parse_args()

    ref = load_reference(args.reference) if args.reference else None

    # Open-channel detection. Explicit flag wins; otherwise read the config.
    open_channel = args.open_channel
    if not open_channel and args.config is not None:
        try:
            with open(args.config, 'r') as _fh:
                _cfg = yaml.safe_load(_fh) or {}
            _bc = (_cfg.get('boundary_conditions') or {}).get('top_wall') or {}
            if _bc.get('type') == 'neumann':
                open_channel = True
                print("  Detected open channel from config "
                      "(boundary_conditions.top_wall.type: neumann)")
        except (OSError, yaml.YAMLError):
            pass

    # Validate that --config is provided when using --checkpoint
    if args.checkpoint and args.config is None:
        parser.error("--config is required when using --checkpoint")

    # Determine output directory and prefix
    stats_dir = os.path.dirname(args.stats_file)
    stats_base = os.path.splitext(os.path.basename(args.stats_file))[0]

    if args.output is None:
        # Default: save in same directory as stats file with name derived from it
        if stats_dir:
            output_prefix = os.path.join(stats_dir, f"{stats_base}_plots")
        else:
            output_prefix = f"{stats_base}_plots"
    else:
        # User specified output prefix (use as-is)
        output_prefix = args.output

    # Load statistics
    print(f"Loading statistics from: {args.stats_file}")
    data = np.load(args.stats_file)

    if args.checkpoint:
        print("  Processing checkpoint file (running sums)...")
        n_samples = int(data['n_samples'])
        if n_samples == 0:
            print("Error: n_samples is 0 in checkpoint file.")
            return 1
        
        # Load sums and compute means
        U_mean = data['U_sum'] / n_samples
        uu_mean = data['uu_sum'] / n_samples
        vv_mean = data['vv_sum'] / n_samples
        ww_mean = data['ww_sum'] / n_samples
        uw_mean = data['uw_sum'] / n_samples
        
        # Load 2D spectra sums and compute means
        E_uu_2d = data['E_uu_2d_sum'] / n_samples
        E_vv_2d = data['E_vv_2d_sum'] / n_samples
        E_ww_2d = data['E_ww_2d_sum'] / n_samples
        E_uw_2d = data['E_uw_2d_sum'] / n_samples
        
        # Reconstruct grid from config
        nx = int(data['nx'])
        ny = int(data['ny'])
        nz = int(data['nz'])
        print(f"  Reconstructing grid for nx={nx}, ny={ny}, nz={nz} from config...")
        z_c, kx, ky = reconstruct_grid_from_config(args.config, nx, ny, nz)
        if 'z_c' in data.files:
            # newer state files carry the grid truth; prefer it
            z_c = data['z_c']
        
        # Checkpoint files might not have nu/u_tau, so we rely on args or config later
        # But we can try to get them if they exist (unlikely in simple checkpoint)
        nu_from_file = None
        u_tau_from_file = None
        
    else:
        # Standard statistics file
        # Extract data (z_c should already be interior points only, no ghost cells)
        z_c = data['z_c']
        U_mean = data['U_mean']
        uu_mean = data['uu_mean']
        vv_mean = data['vv_mean']
        ww_mean = data['ww_mean']
        uw_mean = data['uw_mean']
        kx = data['kx']
        ky = data['ky']
        E_uu_2d = data['E_uu_2d']
        E_vv_2d = data['E_vv_2d']
        E_ww_2d = data['E_ww_2d']
        E_uw_2d = data['E_uw_2d']
        n_samples = int(data['n_samples'])
        
        nu_from_file = data.get('nu', None)
        u_tau_from_file = data.get('u_tau', None)

    # dz_f sums to Lz exactly; both file kinds may carry it.
    dz_f_from_file = data['dz_f'] if 'dz_f' in data.files else None

    # Geometry recorded by the solver. Files written before this was added lack
    # these keys, hence the fallbacks below -- but when present they are the
    # truth and override both the config and the auto-detection heuristic.
    Lz_from_file = float(data['Lz']) if 'Lz' in data.files else None
    delta_from_file = float(data['delta']) if 'delta' in data.files else None
    if 'top_wall_bc_type' in data.files:
        bc_recorded = str(data['top_wall_bc_type'])
        if bc_recorded == 'neumann' and not open_channel:
            print("  Open channel: recorded in the statistics file "
                  "(top_wall_bc_type = neumann)")
        open_channel = (bc_recorded == 'neumann')

    # Profile-based sanity check. On a closed channel U_mean[-1] is a near-wall
    # value and so is small; a large one means a free-slip top that was not
    # declared, which would silently inflate u_tau.
    if not open_channel and len(U_mean) > 2:
        if abs(U_mean[-1]) > 0.25 * float(np.max(np.abs(U_mean))):
            print(f"\n  WARNING: U_mean at the top of the domain is "
                  f"{U_mean[-1]:.4f}, which is {U_mean[-1] / np.max(np.abs(U_mean)):.0%} "
                  f"of the profile maximum.\n"
                  f"           That is a free surface, not a no-slip wall. Treating "
                  f"it as a wall folds the\n"
                  f"           free-surface gradient into u_tau and inflates it "
                  f"several-fold. Pass --open-channel\n"
                  f"           (or --config with top_wall.type: neumann) if this "
                  f"is an open-channel run.\n")

    # Compute domain parameters.
    # z_c[0] + z_c[-1] is only equal to Lz on a SYMMETRIC grid; with 'bottom' or
    # 'double' stretching it is not (0.9915 vs 1.0 on the Re_tau=180 open-channel
    # case). dz_f sums to Lz exactly, so prefer it when the file has it.
    # Prefer what the solver recorded; fall back only for older files.
    if Lz_from_file is not None:
        Lz = Lz_from_file
        print(f"  Domain height Lz = {Lz:.6f} (recorded by the solver)")
    elif dz_f_from_file is not None:
        Lz = float(np.sum(dz_f_from_file))
        print(f"  Domain height Lz = {Lz:.6f} (exact, from dz_f)")
    else:
        Lz = z_c[0] + z_c[-1]
        print(f"  Domain height Lz = {Lz:.6f} (ESTIMATED from z_c -- assumes a "
              f"symmetric grid; rerun to record it)")

    # delta is BC-dependent: Lz/2 for a closed channel (two walls), Lz for an
    # open channel, Lz - h for a canopy. The solver already knows which, so use
    # its value rather than re-deriving one. The canopy branch below may still
    # override with the Monti convention when --canopy-height is given.
    if delta_from_file is not None:
        delta = delta_from_file
        print(f"  delta = {delta:.6f} (recorded by the solver)")
    else:
        delta = Lz if open_channel else Lz / 2
        if open_channel:
            print(f"  Open channel (free-slip top): delta = Lz = {delta:.6f}")

    # Compute dU/dz correctly from U_mean (ignore dUdz_mean from file)
    # Note: omega_y_mean ≈ dU/dz since dW/dx = 0 statistically
    # Now using augmented grid with wall points for better accuracy
    dUdz_mean, dUdz_wall_bot, dUdz_wall_top = compute_dUdz(
        U_mean, z_c, Lz, open_channel=open_channel)
    print(f"  Computed dU/dz from U_mean profile (with augmented wall points)")

    # Premultiply spectra: kx * ky * E(kx, ky)
    # Create 2D wavenumber grids
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    premult_uu = KX * KY * E_uu_2d
    premult_vv = KX * KY * E_vv_2d
    premult_ww = KX * KY * E_ww_2d
    premult_uw = KX * KY * E_uw_2d

    # Determine nu (prefer command line, fallback to file, error if neither)
    if args.Re is not None:
        nu = 1.0 / args.Re
        print(f"  Using nu from --Re: {nu:.6e}")
    elif args.nu is not None:
        nu = args.nu
        print(f"  Using nu from --nu: {nu:.6e}")
    elif nu_from_file is not None:
        nu = float(nu_from_file)
        print(f"  Using nu from stats file: {nu:.6e}")
    elif args.config is not None:
        # Try to get nu from config if provided
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            if 'flow' in config and 'Re' in config['flow']:
                nu = 1.0 / config['flow']['Re']
                print(f"  Using nu from config file: {nu:.6e}")
            else:
                print("Error: nu not found in config file.")
                return 1
    else:
        print("Error: nu not found in stats file and no --Re, --nu, or --config provided.")
        return 1

    # ALWAYS compute u_tau from mean velocity gradient AT THE WALL
    # At the wall: omega_y ≈ dU/dz (since dW/dx = 0 statistically)
    # Wall shear stress: tau_wall = nu * dU/dz|_wall
    # Therefore: u_tau^2 = nu * dU/dz|_wall
    # Use the wall derivatives computed from augmented grid
    
    # Bottom wall: dU/dz > 0
    tau_wall_bot = nu * dUdz_wall_bot
    
    # Top wall: dU/dz < 0 (velocity goes from + to 0 as z increases to Lz? No, symmetric profile)
    # Wait, in channel flow U is positive in middle, 0 at walls.
    # At bottom (z=0), U increases, so dU/dz > 0.
    # At top (z=Lz), U decreases to 0, so dU/dz < 0.
    # So shear stress magnitude is |tau|.
    tau_wall_top = nu * abs(dUdz_wall_top)

    u_tau_bot = np.sqrt(tau_wall_bot)
    u_tau_top = np.sqrt(tau_wall_top)

    if open_channel:
        # One wall only. Averaging in a "top wall" value here would fold in the
        # free-surface gradient and inflate u_tau by a large factor.
        u_tau_from_data = u_tau_bot
    else:
        u_tau_from_data = 0.5 * (u_tau_bot + u_tau_top)
    Re_tau_from_data = u_tau_from_data * delta / nu

    # Determine u_tau for plotting/scaling
    # Priority: 1) --Re_tau argument, 2) computed from data
    # (File value is ignored for scaling but printed for comparison)
    
    if args.Re_tau is not None:
        # Use target value from command line
        u_tau = args.Re_tau * nu / delta
        print(f"  Using u_tau from --Re_tau: u_tau = Re_tau * nu / delta = {u_tau:.6e}")
        print(f"  (Lz = {Lz:.6f}, delta = {delta:.6f})")
        print(f"  Target Re_tau = {args.Re_tau:.2f}")
    else:
        # Always use computed value (even if file has one)
        u_tau = u_tau_from_data
        print(f"  Using u_tau computed from Reynolds stress (dU/dz at wall)")
        
        if u_tau_from_file is not None:
             u_tau_file = float(u_tau_from_file)
             Re_tau_file = u_tau_file * delta / nu
             print(f"  (Value in stats file: u_tau = {u_tau_file:.6e}, Re_tau = {Re_tau_file:.2f})")

    # Display computed Re_tau as convergence check
    print(f"\n  Convergence check (from mean velocity gradient dU/dz near wall):")
    print(f"    u_tau (computed) = {u_tau_from_data:.6e}")
    print(f"    Re_tau (computed) = {Re_tau_from_data:.2f}")

    if args.Re_tau is not None or u_tau_from_file is not None:
        # Show comparison if target was specified
        Re_tau_target = args.Re_tau if args.Re_tau is not None else u_tau * delta / nu
        error = abs(Re_tau_from_data - Re_tau_target) / Re_tau_target * 100
        print(f"    Error from target = {error:.2f}%")

    print(f"  Number of samples: {n_samples}")

    # Canopy conventions (Monti et al. 2022): friction velocities from the
    # stress profile. Height convention: H = outer region height (tip to free
    # surface) = Lz - h; total channel height Lz = H + h. Re numbers use H.
    canopy_h = args.canopy_height
    if canopy_h is not None:
        H = Lz - canopy_h
        # Total stress profile: tau(z) = nu dU/dz - <u'w'>
        tau_total = nu * dUdz_mean - uw_mean
        # Re_tau,in: bed shear stress
        u_tau_in = np.sqrt(max(nu * dUdz_wall_bot, 0.0))
        # Re_tau,out: TOTAL stress interpolated at the canopy tip z = h
        tau_tip = float(np.interp(canopy_h, z_c, tau_total))
        u_tau_out = np.sqrt(max(tau_tip, 0.0))
        print(f"\n  Canopy conventions (h = {canopy_h}, H = {H:.4f}):")
        print(f"    u_tau,in  (bed shear)          = {u_tau_in:.6e}  ->  Re_tau,in  = {u_tau_in * H / nu:.1f}")
        print(f"    u_tau,out (total stress at h)  = {u_tau_out:.6e}  ->  Re_tau,out = {u_tau_out * H / nu:.1f}")
        if args.Re_tau is None:
            u_tau = u_tau_out
            print(f"    Normalizing profiles with u_tau,out (pass --Re_tau to override)")

    # Figure 1: Mean velocity profiles (2 subplots)
    fig1 = plt.figure(figsize=(12, 5))
    gs1 = fig1.add_gridspec(1, 2, hspace=0.3, wspace=0.3)

    ax_U_outer = fig1.add_subplot(gs1[0, 0])
    ax_U_inner = fig1.add_subplot(gs1[0, 1])

    plot_mean_velocity(z_c, U_mean, u_tau, nu, ax_U_outer, ax_U_inner, ref=ref)
    fig1.suptitle(f'Mean Velocity Profile ({n_samples} samples)', fontsize=12)
    if canopy_h is not None:
        # outer plot: U on x, z on y -> horizontal tip line; inner plot: z+ on x
        ax_U_outer.axhline(canopy_h, color='gray', linestyle='--', linewidth=1, alpha=0.8)
        ax_U_inner.axvline(canopy_h * u_tau / nu, color='gray', linestyle='--', linewidth=1, alpha=0.8)

    # Figure 2: Normal Reynolds stresses (1 subplot)
    fig2 = plt.figure(figsize=(8, 6))
    ax_stresses = fig2.add_subplot(111)

    plot_reynolds_stresses_normal(z_c, uu_mean, vv_mean, ww_mean, u_tau, nu, ax_stresses, ref=ref)
    fig2.suptitle(f'Normal Reynolds Stresses ({n_samples} samples)', fontsize=12)
    if canopy_h is not None:
        ax_stresses.axvline(canopy_h * u_tau / nu, color='gray', linestyle='--', linewidth=1, alpha=0.8)

    # Figure 3: Shear stress and vorticity (2 square subplots)
    fig3 = plt.figure(figsize=(10, 5))
    gs3 = fig3.add_gridspec(1, 2, hspace=0.3, wspace=0.4)

    ax_uw = fig3.add_subplot(gs3[0, 0])
    ax_omega = fig3.add_subplot(gs3[0, 1])

    plot_shear_vorticity(z_c, uw_mean, dUdz_mean, u_tau, nu, ax_uw, ax_omega, ref=ref)
    fig3.suptitle(f'Shear Stress and Mean Velocity Gradient ({n_samples} samples)', fontsize=12)
    if canopy_h is not None:
        ax_uw.axvline(canopy_h * u_tau / nu, color='gray', linestyle='--', linewidth=1, alpha=0.8)
        ax_omega.axvline(canopy_h * u_tau / nu, color='gray', linestyle='--', linewidth=1, alpha=0.8)

    # Figure 4: 2D premultiplied spectra (4 subplots)
    # Multi-plane mode (canopy runs): one figure per stored plane
    spectra_z = None
    if not args.checkpoint and 'spectra_z' in data.files:
        spectra_z = data['spectra_z']
    elif args.checkpoint and 'spectra_z' in data.files:
        spectra_z = data['spectra_z']

    spectra_figs = []   # list of (fig, filename_tag)
    if spectra_z is not None:
        for i, z_pl in enumerate(spectra_z):
            figS = plt.figure(figsize=(12, 10))
            gsS = figS.add_gridspec(2, 2, hspace=0.35, wspace=0.35)
            axs = [figS.add_subplot(gsS[r, c]) for r in (0, 1) for c in (0, 1)]
            plot_2d_spectra_wavelength(kx, ky, premult_uu[i], premult_vv[i],
                                       premult_ww[i], premult_uw[i],
                                       axs[0], axs[1], axs[2], axs[3], u_tau, nu)
            figS.suptitle(f'2D Premultiplied Spectra at $z = {z_pl:.3f}$ ({n_samples} samples)',
                          fontsize=12)
            spectra_figs.append((figS, f"spectra_2d_z{z_pl:.3f}"))
        fig4 = None
    else:
        fig4 = plt.figure(figsize=(12, 10))
        gs4 = fig4.add_gridspec(2, 2, hspace=0.35, wspace=0.35)

        ax_premult_uu = fig4.add_subplot(gs4[0, 0])
        ax_premult_vv = fig4.add_subplot(gs4[0, 1])
        ax_premult_ww = fig4.add_subplot(gs4[1, 0])
        ax_premult_uw = fig4.add_subplot(gs4[1, 1])

        plot_2d_spectra_wavelength(kx, ky, premult_uu, premult_vv, premult_ww, premult_uw,
                                   ax_premult_uu, ax_premult_vv, ax_premult_ww, ax_premult_uw,
                                   u_tau, nu)
        fig4.suptitle(f'2D Premultiplied Spectra at $z^+ \\approx 15$ ({n_samples} samples)', fontsize=12)

    # Figure 5: Total stress decomposition (1:1 aspect ratio)
    fig5 = plt.figure(figsize=(6, 6))
    ax_total = fig5.add_subplot(111)

    plot_total_stress_decomposition(z_c, uw_mean, dUdz_mean, u_tau, nu, ax_total)
    fig5.suptitle(f'Total Stress Decomposition ({n_samples} samples)', fontsize=12)
    if canopy_h is not None:
        ax_total.axvline(canopy_h * u_tau / nu, color='gray', linestyle='--', linewidth=1, alpha=0.8)

    # Figure 6: skewness profiles (new stats; skip for older files)
    fig6 = None
    if 'uuu_mean' in data.files and not args.checkpoint:
        uuu = data['uuu_mean']
        www = data['www_mean']
        with np.errstate(divide='ignore', invalid='ignore'):
            S_u = uuu / np.maximum(uu_mean, 1e-300) ** 1.5
            S_w = www / np.maximum(ww_mean, 1e-300) ** 1.5
        fig6 = plt.figure(figsize=(6, 6))
        ax6 = fig6.add_subplot(111)
        ax6.plot(S_u, z_c, 'k-', lw=1.5, label=r'$S_u$')
        ax6.plot(S_w, z_c, 'C3-', lw=1.5, label=r'$S_w$')
        ax6.axvline(0, color='gray', lw=0.8)
        if canopy_h is not None:
            ax6.axhline(canopy_h, color='gray', linestyle='--', lw=1, alpha=0.8)
        ax6.set_xlabel(r'skewness')
        ax6.set_ylabel(r'$z$')
        ax6.legend()
        ax6.grid(alpha=0.3)
        fig6.suptitle(f'Velocity Skewness ({n_samples} samples)', fontsize=12)

    # Figure 7: canopy drag force density profile
    fig7 = None
    if ('fx_profile_mean' in data.files and not args.checkpoint
            and np.abs(data['fx_profile_mean']).max() > 0 and 'dz_f' in data.files):
        fx_prof = data['fx_profile_mean']
        A_xy = float(data['Lx']) * float(data['Ly'])
        f_density = -fx_prof / (A_xy * data['dz_f'])   # drag force density (+ = opposing flow)
        fig7 = plt.figure(figsize=(6, 6))
        ax7 = fig7.add_subplot(111)
        ax7.plot(f_density, z_c, 'k-', lw=1.5)
        if canopy_h is not None:
            ax7.axhline(canopy_h, color='gray', linestyle='--', lw=1, alpha=0.8)
            ax7.set_ylim(0, 1.6 * canopy_h)
        ax7.set_xlabel(r'$-\langle f_x \rangle$ (drag density)')
        ax7.set_ylabel(r'$z$')
        ax7.grid(alpha=0.3)
        fig7.suptitle(f'Canopy Drag Profile ({n_samples} samples)', fontsize=12)

    # Save figures
    formats = ['pdf', 'png'] if args.format == 'both' else [args.format]

    print(f"\nSaving plots to: {output_prefix}_*.{formats[0]}")

    for fmt in formats:
        dpi = args.dpi if fmt == 'png' else None

        to_save = [(fig1, 'velocity'), (fig2, 'normal_stresses'),
                   (fig3, 'shear_vorticity'), (fig5, 'total_stress')]
        if fig4 is not None:
            to_save.append((fig4, 'spectra_2d'))
        to_save.extend(spectra_figs)
        if fig6 is not None:
            to_save.append((fig6, 'skewness'))
        if fig7 is not None:
            to_save.append((fig7, 'canopy_drag_profile'))

        for fig, tag in to_save:
            path = f"{output_prefix}_{tag}.{fmt}"
            fig.savefig(path, dpi=dpi, bbox_inches='tight')
            print(f"  Saved: {path}")

    print("\nPlotting completed successfully!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
