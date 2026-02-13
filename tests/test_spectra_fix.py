#!/usr/bin/env python3
"""
Test script to verify the fix for 2D premultiplied spectra.

The issue: When averaging statistics from z+=15 at both walls, we need to account
for channel flow symmetry. The top wall data should be flipped in the spanwise (y)
direction to properly align with the bottom wall before averaging.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from statistics import TurbulenceStats
from utils import load_flow_fields

# LaTeX formatting - EXACTLY matching plot_statistics.py
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

def compute_2d_spectra_original(u_fluct, v_fluct, w_fluct, k_bot, k_top, nx, ny, Lx, Ly, device):
    """
    Original method: Direct averaging (produces spurious double peaks)
    EXACTLY matches statistics.py lines 246-292
    """
    # Bottom wall plane (line 247-249)
    u_bot = u_fluct[:, :, k_bot - 1]  # (nx, ny)
    v_bot = v_fluct[:, :, k_bot - 1]
    w_bot = w_fluct[:, :, k_bot - 1]

    # Top wall plane (line 252-254)
    u_top = u_fluct[:, :, k_top - 1]
    v_top = v_fluct[:, :, k_top - 1]
    w_top = w_fluct[:, :, k_top - 1]

    # Average between walls (NO FLIP - WRONG!) (line 257-259)
    u_plane = 0.5 * (u_bot + u_top)
    v_plane = 0.5 * (v_bot + v_top)
    w_plane = 0.5 * (w_bot + w_top)

    return compute_spectra_from_planes(u_plane, v_plane, w_plane, nx, ny, Lx, Ly, device)


def compute_2d_spectra_fixed(u_fluct, v_fluct, w_fluct, k_bot, k_top, nx, ny, Lx, Ly, device):
    """
    Fixed method: Flip top wall data in y-direction before averaging
    """
    # Bottom wall plane
    u_bot = u_fluct[:, :, k_bot - 1]
    v_bot = v_fluct[:, :, k_bot - 1]
    w_bot = w_fluct[:, :, k_bot - 1]

    # Top wall plane
    u_top = u_fluct[:, :, k_top - 1]
    v_top = v_fluct[:, :, k_top - 1]
    w_top = w_fluct[:, :, k_top - 1]

    # FLIP top wall data in y-direction to account for channel symmetry
    u_top = torch.flip(u_top, dims=[1])  # Flip in y
    v_top = -torch.flip(v_top, dims=[1])  # Flip in y AND change sign (v reverses)
    w_top = torch.flip(w_top, dims=[1])  # Flip in y only

    # Average between walls (now properly aligned)
    u_plane = 0.5 * (u_bot + u_top)
    v_plane = 0.5 * (v_bot + v_top)
    w_plane = 0.5 * (w_bot + w_top)

    return compute_spectra_from_planes(u_plane, v_plane, w_plane, nx, ny, Lx, Ly, device)


def compute_spectra_from_planes(u_plane, v_plane, w_plane, nx, ny, Lx, Ly, device):
    """
    Compute 2D premultiplied spectra from velocity planes
    EXACTLY matches statistics.py lines 261-292
    """
    # Compute 2D FFT (line 262-264)
    u_fft_2d = torch.fft.rfft2(u_plane)  # (nx, ny//2+1)
    v_fft_2d = torch.fft.rfft2(v_plane)
    w_fft_2d = torch.fft.rfft2(w_plane)

    # Compute energy spectra (line 267-270)
    E_uu_2d = torch.abs(u_fft_2d)**2 / (nx * ny)**2
    E_vv_2d = torch.abs(v_fft_2d)**2 / (nx * ny)**2
    E_ww_2d = torch.abs(w_fft_2d)**2 / (nx * ny)**2
    E_uw_2d = (u_fft_2d * torch.conj(w_fft_2d)).real / (nx * ny)**2

    # Wavenumber grids (line 108-109 from __init__)
    kx_2d = torch.arange(nx, device=device, dtype=torch.float32).view(-1, 1) * (2 * np.pi / Lx)
    ky_2d = torch.arange(ny//2+1, device=device, dtype=torch.float32).view(1, -1) * (2 * np.pi / Ly)

    # Premultiply (line 273-276)
    premult_uu = kx_2d * ky_2d * E_uu_2d
    premult_vv = kx_2d * ky_2d * E_vv_2d
    premult_ww = kx_2d * ky_2d * E_ww_2d
    premult_uw = kx_2d * ky_2d * E_uw_2d

    # Fold spectrum (line 280-292)
    nkx = nx // 2
    nky = ny // 2

    premult_uu_sym = premult_uu[1:nkx+1, 1:nky+1] + torch.flip(premult_uu[1:nkx+1, 1:nky+1], dims=[0])
    premult_vv_sym = premult_vv[1:nkx+1, 1:nky+1] + torch.flip(premult_vv[1:nkx+1, 1:nky+1], dims=[0])
    premult_ww_sym = premult_ww[1:nkx+1, 1:nky+1] + torch.flip(premult_ww[1:nkx+1, 1:nky+1], dims=[0])
    premult_uw_sym = premult_uw[1:nkx+1, 1:nky+1] + torch.flip(premult_uw[1:nkx+1, 1:nky+1], dims=[0])

    return {
        'uu': premult_uu_sym[:nkx, :nky].cpu().numpy(),
        'vv': premult_vv_sym[:nkx, :nky].cpu().numpy(),
        'ww': premult_ww_sym[:nkx, :nky].cpu().numpy(),
        'uw': premult_uw_sym[:nkx, :nky].cpu().numpy(),
    }


def plot_comparison(kx, ky, original, fixed, output_file='spectra_comparison.png'):
    """
    Plot side-by-side comparison of original vs fixed spectra
    EXACTLY matching plot_statistics.py style (wavelengths, log scale, hot colormap)
    """
    # Convert wavenumbers to wavelengths (matching plot_statistics.py line 142-143)
    lambda_x = 2 * np.pi / kx
    lambda_y = 2 * np.pi / ky

    # Create meshgrid for contour plots (matching plot_statistics.py line 146)
    LAMBDA_X, LAMBDA_Y = np.meshgrid(lambda_x, lambda_y, indexing='ij')

    # Helper function to get log-spaced levels (matching plot_statistics.py line 149-152)
    def get_levels(data, n_levels=10):
        vmin = np.maximum(data[data > 0].min(), 1e-10) if np.any(data > 0) else 1e-10
        vmax = data.max()
        return np.logspace(np.log10(vmin), np.log10(vmax), n_levels)

    # Create figure with 2 rows × 4 columns
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.35)

    components = ['uu', 'vv', 'ww', 'uw']
    titles = [r"$k_x k_y E_{uu}$", r"$k_x k_y E_{vv}$", r"$k_x k_y E_{ww}$", r"$k_x k_y E_{uw}$"]

    for i, (comp, title) in enumerate(zip(components, titles)):
        # Original (top row)
        ax_orig = fig.add_subplot(gs[0, i])

        if comp == 'uw':
            # u'w' uses symmetric linear scale (matching plot_statistics.py line 185-187)
            vmax_uw = np.abs(original[comp]).max()
            levels = np.linspace(-vmax_uw, vmax_uw, 20)
            ax_orig.contourf(LAMBDA_X, LAMBDA_Y, original[comp], levels=levels, cmap='RdBu_r')
        else:
            # uu, vv, ww use log scale (matching plot_statistics.py line 155-157)
            levels = get_levels(original[comp])
            ax_orig.contourf(LAMBDA_X, LAMBDA_Y, original[comp], levels=levels,
                           norm=mpl.colors.LogNorm(), cmap='hot')

        ax_orig.set_xlabel(r'$\lambda_x$')
        ax_orig.set_ylabel(r'$\lambda_y$')
        ax_orig.set_title(f'{title} (Original - Double peak)', fontsize=9)
        ax_orig.set_xscale('log')
        ax_orig.set_yscale('log')

        # Fixed (bottom row)
        ax_fixed = fig.add_subplot(gs[1, i])

        if comp == 'uw':
            vmax_uw = np.abs(fixed[comp]).max()
            levels = np.linspace(-vmax_uw, vmax_uw, 20)
            ax_fixed.contourf(LAMBDA_X, LAMBDA_Y, fixed[comp], levels=levels, cmap='RdBu_r')
        else:
            levels = get_levels(fixed[comp])
            ax_fixed.contourf(LAMBDA_X, LAMBDA_Y, fixed[comp], levels=levels,
                            norm=mpl.colors.LogNorm(), cmap='hot')

        ax_fixed.set_xlabel(r'$\lambda_x$')
        ax_fixed.set_ylabel(r'$\lambda_y$')
        ax_fixed.set_title(f'{title} (Fixed - Single peak)', fontsize=9)
        ax_fixed.set_xscale('log')
        ax_fixed.set_yscale('log')

    fig.suptitle(r'2D Premultiplied Spectra Comparison at $z^+ \approx 15$', fontsize=12)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to: {output_file}")


def main():
    """
    Test the spectra fix on a field file
    """
    import argparse

    parser = argparse.ArgumentParser(description='Test 2D spectra fix')
    parser.add_argument('field_file', help='Path to field NPZ file')
    parser.add_argument('--output', default='spectra_comparison.png', help='Output plot file')
    args = parser.parse_args()

    print("="*80)
    print("Testing 2D Premultiplied Spectra Fix")
    print("="*80)

    # Load field
    print(f"\nLoading field from: {args.field_file}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fields = load_flow_fields(args.field_file, device=device)

    u = fields['u']
    v = fields['v']
    w = fields['w']
    z_c = fields['z_c']
    Lx = fields['Lx']
    Ly = fields['Ly']

    # Grid dimensions
    nx = u.shape[0] - 1
    ny = u.shape[1] - 2
    nz = w.shape[2] - 1
    Lz = z_c[-1].item()

    print(f"  Grid: nx={nx}, ny={ny}, nz={nz}")
    print(f"  Domain: Lx={Lx:.4f}, Ly={Ly:.4f}, Lz={Lz:.4f}")

    # Parameters for z+ location
    nu = 1.0 / 10000.0  # Adjust based on your Re
    Re_tau_target = 535.0
    z_plus_target = 15.0

    delta = Lz / 2.0
    u_tau_target = Re_tau_target * nu / delta
    z_target_phys = z_plus_target * nu / u_tau_target

    # Find indices
    z_c_interior = z_c[1:nz+1]
    idx_bot = torch.argmin(torch.abs(z_c_interior - z_target_phys)).item()
    k_bot = idx_bot + 1

    idx_top = torch.argmin(torch.abs(z_c_interior - (Lz - z_target_phys))).item()
    k_top = idx_top + 1

    print(f"\nStatistics locations:")
    print(f"  Bottom: k={k_bot}, z={z_c[k_bot]:.6f}, z+={z_c[k_bot]*u_tau_target/nu:.1f}")
    print(f"  Top: k={k_top}, z={z_c[k_top]:.6f}, z+={(Lz-z_c[k_top])*u_tau_target/nu:.1f}")

    # Compute fluctuations EXACTLY matching statistics.py lines 130-170
    print("\nComputing velocity fluctuations...")

    # Extract interior points (line 130-132)
    u_int = u[0:nx+1, 1:ny+1, 1:nz+1]  # (nx+1, ny, nz)
    v_int = v[1:nx+1, 0:ny+1, 1:nz+1]  # (nx, ny+1, nz)
    w_int = w[1:nx+1, 1:ny+1, 0:nz+1]  # (nx, ny, nz+1)

    # Compute cell-centered velocities and mean profiles (line 136-138)
    u_cell_center = 0.5 * (u_int[:-1, :, :] + u_int[1:, :, :])  # (nx, ny, nz)
    U = torch.mean(u_cell_center, dim=(0, 1))  # (nz,)
    u_fluct = u_cell_center - U.view(1, 1, -1)  # (nx, ny, nz)

    # For v (line 163-165)
    v_cell_center = 0.5 * (v_int[:, :-1, :] + v_int[:, 1:, :])  # (nx, ny, nz)
    V = torch.mean(v_cell_center, dim=(0, 1))  # (nz,)
    v_fluct = v_cell_center - V.view(1, 1, -1)

    # For w (line 168-170)
    w_cell_center = 0.5 * (w_int[:, :, :-1] + w_int[:, :, 1:])  # (nx, ny, nz)
    W = torch.mean(w_cell_center, dim=(0, 1))  # (nz,)
    w_fluct = w_cell_center - W.view(1, 1, -1)

    print(f"  u_fluct shape: {u_fluct.shape} (expected: ({nx}, {ny}, {nz}))")
    print(f"  Extracting planes at k_bot={k_bot}, k_top={k_top}")

    # Compute spectra with both methods
    print("\nComputing 2D spectra (original method)...")
    original = compute_2d_spectra_original(u_fluct, v_fluct, w_fluct, k_bot, k_top, nx, ny, Lx, Ly, device)

    print("Computing 2D spectra (fixed method)...")
    fixed = compute_2d_spectra_fixed(u_fluct, v_fluct, w_fluct, k_bot, k_top, nx, ny, Lx, Ly, device)

    # Compare peak values
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    for comp in ['uu', 'vv', 'ww', 'uw']:
        orig_max = original[comp].max()
        fixed_max = fixed[comp].max()
        print(f"{comp}: Original max = {orig_max:.6e}, Fixed max = {fixed_max:.6e}")

    # Compute wavenumber arrays (matching statistics.py line 103-104)
    dx = Lx / nx
    dy = Ly / ny
    kx = 2 * np.pi * np.fft.rfftfreq(nx, d=dx/nx)[1:]  # Skip DC component
    ky = 2 * np.pi * np.fft.rfftfreq(ny, d=dy/ny)[1:]

    # Extract only the first nx//2 and ny//2 elements (matching folded spectrum size)
    kx = kx[:nx//2]
    ky = ky[:ny//2]

    # Plot comparison
    print("\nGenerating comparison plots...")
    plot_comparison(kx, ky, original, fixed, args.output)

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print("\nExpected result:")
    print("  - Original: Should show TWO symmetric peaks (spurious)")
    print("  - Fixed: Should show ONE central peak (correct)")
    print("\nIf the fixed version shows a single peak, the fix is working correctly!")


if __name__ == '__main__':
    main()
