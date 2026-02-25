import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.patches import Circle
from scipy.interpolate import interp1d
import os
import argparse
import yaml

# Configure Matplotlib for LaTeX
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('axes', labelsize=14)
rc('axes', titlesize=16)
rc('xtick', labelsize=12)
rc('ytick', labelsize=12)
rc('legend', fontsize=12)
rc('figure', titlesize=18)
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

def read_nu_from_config(config_path):
    """Read kinematic viscosity from config file.

    nu = 1 / Re_bulk, where Re_bulk is stored as 'Re' in the config.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    Re_bulk = config['flow']['Re']
    nu = 1.0 / Re_bulk
    print(f"Read Re_bulk = {Re_bulk} from config")
    print(f"Computed nu = 1/Re_bulk = {nu:.6e}")
    return nu

def resample_to_uniform_z(data_2d, z_nonuniform, z_min=None, z_max=None, nz_uniform=None):
    """
    Resample 2D data with non-uniform z-coordinate onto uniform z-grid.

    Parameters:
    -----------
    data_2d : ndarray (nz, nx) or (nz, ny)
        Data to resample, with z as first dimension
    z_nonuniform : ndarray (nz,)
        Non-uniform z coordinates
    z_min, z_max : float, optional
        Range for uniform grid (default: use data range)
    nz_uniform : int, optional
        Number of points in uniform grid (default: 3*len(z_nonuniform) for smoother interpolation)

    Returns:
    --------
    data_uniform : ndarray (nz_uniform, nx) or (nz_uniform, ny)
        Resampled data on uniform z-grid
    z_uniform : ndarray (nz_uniform,)
        Uniform z coordinates
    """
    if z_min is None:
        z_min = z_nonuniform.min()
    if z_max is None:
        z_max = z_nonuniform.max()
    if nz_uniform is None:
        # Use 3x resolution for smoother interpolation
        nz_uniform = len(z_nonuniform) * 3

    # Create uniform z-grid
    z_uniform = np.linspace(z_min, z_max, nz_uniform)

    # Interpolate each column (constant x or y) independently
    nz_orig, n_other = data_2d.shape
    data_uniform = np.zeros((nz_uniform, n_other))

    for i in range(n_other):
        # Cubic interpolation along z direction
        interpolator = interp1d(z_nonuniform, data_2d[:, i],
                               kind='cubic', bounds_error=False, fill_value='extrapolate')
        data_uniform[:, i] = interpolator(z_uniform)

    return data_uniform, z_uniform


def load_fields(filepath):
    """Load fields from .npz file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    data = np.load(filepath)
    u = data['u']
    v = data['v']
    w = data['w']
    p = data['p']
    z_c = data['z_c']
    z_f = data['z_f']

    # Handle legacy files without Lx/Ly
    if 'Lx' in data:
        Lx = data['Lx'].item()
        Ly = data['Ly'].item()
    else:
        print("WARNING: Lx/Ly not found in file. Assuming 2pi, pi.")
        Lx = 2 * np.pi
        Ly = np.pi

    # Extract time and step if available
    time = data['time'].item() if 'time' in data else None
    step = data['step'].item() if 'step' in data else None

    print(f"Loaded fields from {filepath}")
    print(f"Shapes: u={u.shape}, v={v.shape}, w={w.shape}, p={p.shape}")
    print(f"Domain: Lx={Lx:.3f}, Ly={Ly:.3f}")
    if time is not None:
        print(f"Time: t={time:.3f}")
    if step is not None:
        print(f"Step: {step}")

    return u, v, w, p, z_c, z_f, Lx, Ly, time, step

def plot_slices(u, v, w, z_c, Lx, Ly, results_folder, prefix='post', time=None, step=None, sphere_params=None, canopy_h=None):
    """Plot XY, XZ, YZ slices with physical units and colorbars.

    Creates three separate images, each with 3 rows x 1 column showing u, v, w.

    Parameters:
    -----------
    time : float, optional
        Physical simulation time
    step : int, optional
        Iteration/step number
    sphere_params : dict, optional
        Dictionary with 'center' and 'radius' for sphere mask overlay
    """
    nx, ny, nz = u.shape[0]-2, u.shape[1]-2, w.shape[2]

    # Mid-plane indices: use sphere center if sphere_params provided, 
    # use canopy_h for z if provided, else domain center
    if sphere_params is not None:
        # Position slices to pass through sphere center
        xc, yc, zc = sphere_params['center']
        ix = int(xc / Lx * nx)
        iy = int(yc / Ly * ny)
        # For z, need to find the index in the stretched grid
        iz = np.argmin(np.abs(z_c[1:-1] - zc))
    else:
        # Use domain center for x, y
        ix = nx // 2
        iy = ny // 2
        # Use canopy_h for z if provided, else domain center
        if canopy_h is not None:
            iz = np.argmin(np.abs(z_c[1:-1] - canopy_h))
        else:
            iz = nz // 2

    # Coordinate arrays
    x_u = np.linspace(0, Lx, nx+1)
    x_v = np.linspace(0, Lx, nx)
    y_v = np.linspace(0, Ly, ny+1)
    y_u = np.linspace(0, Ly, ny)
    z_u = z_c[1:nz+1]
    z_w = z_c[1:nz]

    # XY slice at z_mid
    u_xy = u[1:nx+1, 1:ny+1, iz+1].T
    v_xy = v[1:nx+1, 1:ny+1, iz+1].T
    w_xy = w[1:nx+1, 1:ny+1, iz].T
    z_mid = sphere_params['center'][2] if sphere_params is not None else z_c[iz+1]

    # XZ slice at y_mid
    u_xz = u[1:nx+1, iy+1, 1:nz+1].T
    v_xz = v[1:nx+1, iy, 1:nz+1].T
    w_xz = w[1:nx+1, iy+1, 1:nz].T
    y_mid = sphere_params['center'][1] if sphere_params is not None else iy * Ly / ny

    # YZ slice at x_mid
    u_yz = u[ix+1, 1:ny+1, 1:nz+1].T
    v_yz = v[ix+1, 1:ny+1, 1:nz+1].T
    w_yz = w[ix+1, 1:ny+1, 1:nz].T
    x_mid = sphere_params['center'][0] if sphere_params is not None else ix * Lx / nx

    # Define coordinate arrays for plotting (needed for sphere masks)
    x_plot = np.linspace(0, Lx, nx)
    y_plot = np.linspace(0, Ly, ny)

    # Compute sphere circle parameters if sphere_params provided
    # NOTE: This assumes the slices pass exactly through the sphere center,
    # so we can draw perfect circles instead of pixelated grid-based masks.
    # For arbitrary slice positions, would need to compute proper circle radii
    # using r_slice = sqrt(R^2 - d^2) where d is distance from sphere center to slice.
    circle_xy, circle_xz, circle_yz = None, None, None
    if sphere_params is not None:
        center = sphere_params['center']
        radius = sphere_params['radius']
        xc, yc, zc = center

        # XY slice at z_mid: circle in (x,y) plane centered at (xc, yc)
        dist_z = abs(z_mid - zc)
        if dist_z <= radius:
            r_xy = np.sqrt(radius**2 - dist_z**2)
            circle_xy = {'center': (xc, yc), 'radius': r_xy}

        # XZ slice at y_mid: circle in (x,z) plane centered at (xc, zc)
        dist_y = abs(y_mid - yc)
        if dist_y <= radius:
            r_xz = np.sqrt(radius**2 - dist_y**2)
            circle_xz = {'center': (xc, zc), 'radius': r_xz}

        # YZ slice at x_mid: circle in (y,z) plane centered at (yc, zc)
        dist_x = abs(x_mid - xc)
        if dist_x <= radius:
            r_yz = np.sqrt(radius**2 - dist_x**2)
            circle_yz = {'center': (yc, zc), 'radius': r_yz}

    # Helper for colorbars
    def plot_field_imshow(ax, data, extent, title, label, cmap='RdBu_r', circle=None):
        im = ax.imshow(data, origin='lower', cmap=cmap, extent=extent, aspect='equal', interpolation='bicubic')
        ax.set_title(title, fontsize=14)
        # Overlay sphere circle if provided
        if circle is not None:
            circ = Circle(circle['center'], circle['radius'],
                         color='gray', alpha=0.9, zorder=10)
            ax.add_patch(circ)
        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.9)
        cbar.set_label(label, fontsize=12)
        return im

    def plot_field_pcolor(ax, x, y, data, title, label, cmap='RdBu_r', circle=None):
        # Use pcolormesh for non-uniform grids (stretched Z)
        # Create 2D meshgrid for proper spacing
        X, Y = np.meshgrid(x, y, indexing='xy')
        # Use 'gouraud' shading for smooth interpolation (similar to bicubic in imshow)
        im = ax.pcolormesh(X, Y, data, cmap=cmap, shading='gouraud')
        ax.set_title(title, fontsize=14)
        ax.set_aspect('equal')
        # Overlay sphere circle if provided
        if circle is not None:
            circ = Circle(circle['center'], circle['radius'],
                         color='gray', alpha=0.9, zorder=10)
            ax.add_patch(circ)
        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.9)
        cbar.set_label(label, fontsize=12)
        return im

    # ========== XY SLICES ==========
    fig_xy, axes_xy = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True)

    # Set figure title with slice info and time
    title_xy = f'$xy$ plane at $z = {z_mid:.3f}$'
    if time is not None:
        title_xy += f', $t = {time:.3f}$'
    fig_xy.suptitle(title_xy, fontsize=16, y=0.995)

    plot_field_imshow(axes_xy[0], u_xy, [0, Lx, 0, Ly], '$u$', '$u$', cmap='plasma', circle=circle_xy)
    axes_xy[0].set_xlabel('$x$')
    axes_xy[0].set_ylabel('$y$')

    plot_field_imshow(axes_xy[1], v_xy, [0, Lx, 0, Ly], '$v$', '$v$', cmap='seismic', circle=circle_xy)
    axes_xy[1].set_xlabel('$x$')
    axes_xy[1].set_ylabel('$y$')

    plot_field_imshow(axes_xy[2], w_xy, [0, Lx, 0, Ly], '$w$', '$w$', cmap='RdBu_r', circle=circle_xy)
    axes_xy[2].set_xlabel('$x$')
    axes_xy[2].set_ylabel('$y$')

    save_path_xy = os.path.join(results_folder, f'{prefix}_slices_xy.png')
    fig_xy.savefig(save_path_xy, dpi=150, bbox_inches='tight')
    plt.close(fig_xy)
    print(f"Saved XY slices to {save_path_xy}")

    # ========== XZ SLICES ==========
    fig_xz, axes_xz = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)

    # Set figure title with slice info and time
    title_xz = f'$xz$ plane at $y = {y_mid:.3f}$'
    if time is not None:
        title_xz += f', $t = {time:.3f}$'
    fig_xz.suptitle(title_xz, fontsize=16, y=0.995)

    # Resample data onto uniform z-grid for smooth bicubic interpolation
    z_min, z_max = z_u.min(), z_u.max()
    u_xz_uniform, z_uniform_u = resample_to_uniform_z(u_xz, z_u, z_min=z_min, z_max=z_max)
    v_xz_uniform, _ = resample_to_uniform_z(v_xz, z_u, z_min=z_min, z_max=z_max)

    z_min_w, z_max_w = z_w.min(), z_w.max()
    w_xz_uniform, z_uniform_w = resample_to_uniform_z(w_xz, z_w, z_min=z_min_w, z_max=z_max_w)

    # Plot with imshow for consistent high-quality rendering
    # Data is already (nz, nx) after resampling - no transpose needed
    plot_field_imshow(axes_xz[0], u_xz_uniform, [0, Lx, z_min, z_max], '$u$', '$u$', cmap='plasma', circle=circle_xz)
    axes_xz[0].set_xlabel('$x$')
    axes_xz[0].set_ylabel('$z$')

    plot_field_imshow(axes_xz[1], v_xz_uniform, [0, Lx, z_min, z_max], '$v$', '$v$', cmap='seismic', circle=circle_xz)
    axes_xz[1].set_xlabel('$x$')
    axes_xz[1].set_ylabel('$z$')

    plot_field_imshow(axes_xz[2], w_xz_uniform, [0, Lx, z_min_w, z_max_w], '$w$', '$w$', cmap='RdBu_r', circle=circle_xz)
    axes_xz[2].set_xlabel('$x$')
    axes_xz[2].set_ylabel('$z$')

    save_path_xz = os.path.join(results_folder, f'{prefix}_slices_xz.png')
    fig_xz.savefig(save_path_xz, dpi=150, bbox_inches='tight')
    plt.close(fig_xz)
    print(f"Saved XZ slices to {save_path_xz}")

    # ========== YZ SLICES ==========
    fig_yz, axes_yz = plt.subplots(3, 1, figsize=(6, 8), constrained_layout=True)

    # Set figure title with slice info and time
    title_yz = f'$yz$ plane at $x = {x_mid:.3f}$'
    if time is not None:
        title_yz += f', $t = {time:.3f}$'
    fig_yz.suptitle(title_yz, fontsize=16, y=0.995)

    # Resample data onto uniform z-grid for smooth bicubic interpolation
    u_yz_uniform, _ = resample_to_uniform_z(u_yz, z_u, z_min=z_min, z_max=z_max)
    v_yz_uniform, _ = resample_to_uniform_z(v_yz, z_u, z_min=z_min, z_max=z_max)
    w_yz_uniform, _ = resample_to_uniform_z(w_yz, z_w, z_min=z_min_w, z_max=z_max_w)

    # Plot with imshow for consistent high-quality rendering
    # Data is already (nz, ny) after resampling - no transpose needed
    plot_field_imshow(axes_yz[0], u_yz_uniform, [0, Ly, z_min, z_max], '$u$', '$u$', cmap='plasma', circle=circle_yz)
    axes_yz[0].set_xlabel('$y$')
    axes_yz[0].set_ylabel('$z$')

    plot_field_imshow(axes_yz[1], v_yz_uniform, [0, Ly, z_min, z_max], '$v$', '$v$', cmap='seismic', circle=circle_yz)
    axes_yz[1].set_xlabel('$y$')
    axes_yz[1].set_ylabel('$z$')

    plot_field_imshow(axes_yz[2], w_yz_uniform, [0, Ly, z_min_w, z_max_w], '$w$', '$w$', cmap='RdBu_r', circle=circle_yz)
    axes_yz[2].set_xlabel('$y$')
    axes_yz[2].set_ylabel('$z$')

    save_path_yz = os.path.join(results_folder, f'{prefix}_slices_yz.png')
    fig_yz.savefig(save_path_yz, dpi=150, bbox_inches='tight')
    plt.close(fig_yz)
    print(f"Saved YZ slices to {save_path_yz}")

def plot_profiles(u, z_c, results_folder, prefix='post'):
    """Plot mean velocity profile with LaTeX formatting."""
    nx, ny, nz = u.shape[0]-2, u.shape[1]-2, u.shape[2]-2

    # Compute mean profile (average over x, y)
    u_mean = np.mean(u[1:nx+1, 1:ny+1, 1:nz+1], axis=(0, 1))
    z_plot = z_c[1:nz+1]

    plt.figure(figsize=(6, 8))
    plt.plot(u_mean, z_plot, 'k-', linewidth=1.5, label='Mean Flow')
    plt.plot(u_mean, z_plot, 'ko', markersize=3, markevery=2, label='Grid Points')

    plt.xlabel(r'$\langle u \rangle$')
    plt.ylabel(r'$z / \delta$')
    plt.title('Mean Streamwise Velocity Profile')
    plt.legend(frameon=False)
    plt.grid(True, linestyle='--', alpha=0.6)

    save_path = os.path.join(results_folder, f'{prefix}_profile.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved profile to {save_path}")

def plot_timeseries(results_folder, prefix='post', t_threshold=50.0, nu=1e-4):
    """Plot time series of u_bulk, u_tau, and forcing from timeseries.csv.

    Also computes averaged statistics for t > t_threshold:
    - Average u_tau (direct)
    - Average u_tau from sqrt(forcing)
    - Re_tau from both measures

    delta is hardcoded to 1.0 (channel half-height).
    """
    import csv

    delta = 1.0  # Channel half-height (hardcoded)

    npz_file = os.path.join(results_folder, 'timeseries.npz')
    csv_file = os.path.join(results_folder, 'timeseries.csv')

    if os.path.exists(npz_file):
        print(f"Loading time series from {npz_file}")
        data = np.load(npz_file)
        time_data = data['time']
        u_bulk_data = data['u_bulk']
        u_tau_data = data['u_tau']
        forcing_data = data['forcing']
    elif os.path.exists(csv_file):
        print(f"Loading time series from {csv_file}")
        # Read data from CSV
        time_data = []
        u_bulk_data = []
        u_tau_data = []
        forcing_data = []

        with open(csv_file, 'r') as f:
            import csv
            reader = csv.DictReader(f)
            for row in reader:
                time_data.append(float(row['time']))
                u_bulk_data.append(float(row['u_bulk']))
                u_tau_data.append(float(row['u_tau']))
                forcing_data.append(float(row['forcing']))
        
        # Convert to numpy arrays
        time_data = np.array(time_data)
        u_bulk_data = np.array(u_bulk_data)
        u_tau_data = np.array(u_tau_data)
        forcing_data = np.array(forcing_data)
    else:
        print(f"WARNING: No time series file found (checked timeseries.npz and timeseries.csv)")
        return
    u_tau_squared = u_tau_data**2

    # Compute averaged statistics for t > t_threshold
    mask = time_data > t_threshold
    if np.any(mask):
        print(f"\n{'='*60}")
        print(f"Computing averaged statistics for t > {t_threshold}")
        print(f"{'='*60}")

        # Average u_tau (direct)
        u_tau_avg = np.mean(u_tau_data[mask])
        print(f"Average u_tau (direct):           {u_tau_avg:.6e}")

        # Average u_tau from sqrt(forcing)
        u_tau_from_forcing = np.sqrt(np.mean(forcing_data[mask]))
        print(f"Average u_tau from sqrt(forcing): {u_tau_from_forcing:.6e}")

        # Re_tau from both measures
        re_tau_direct = u_tau_avg * delta / nu
        re_tau_forcing = u_tau_from_forcing * delta / nu
        print(f"\nRe_tau (from direct u_tau):       {re_tau_direct:.2f}")
        print(f"Re_tau (from sqrt(forcing)):      {re_tau_forcing:.2f}")

        # Relative difference
        rel_diff = abs(re_tau_direct - re_tau_forcing) / re_tau_direct * 100
        print(f"Relative difference:              {rel_diff:.2f}%")
        print(f"{'='*60}\n")
    else:
        print(f"WARNING: No data points found for t > {t_threshold}")

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    # Top left: u_bulk vs time
    axes[0, 0].plot(time_data, u_bulk_data, 'b-', linewidth=1.5, label=r'$u_{\rm bulk}$')
    axes[0, 0].set_xlabel(r'$t$')
    axes[0, 0].set_ylabel(r'$u_{\rm bulk}$')
    axes[0, 0].set_title('Bulk Velocity vs Time')
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)
    axes[0, 0].legend(frameon=False)

    # Top right: u_tau vs time
    axes[0, 1].plot(time_data, u_tau_data, 'r-', linewidth=1.5, label=r'$u_\tau$')
    axes[0, 1].set_xlabel(r'$t$')
    axes[0, 1].set_ylabel(r'$u_\tau$')
    axes[0, 1].set_title('Friction Velocity vs Time')
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)
    axes[0, 1].legend(frameon=False)

    # Bottom left: forcing vs time
    axes[1, 0].plot(time_data, forcing_data, 'g-', linewidth=1.5, label=r'$f_x$')
    axes[1, 0].set_xlabel(r'$t$')
    axes[1, 0].set_ylabel(r'$f_x$')
    axes[1, 0].set_title('Forcing vs Time')
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)
    axes[1, 0].legend(frameon=False)

    # Bottom right: u_tau^2 and forcing on same axis (forcing behind, u_tau^2 on top)
    axes[1, 1].plot(time_data, forcing_data, '-', color='orange', linewidth=1.5, label=r'$f_x$')
    axes[1, 1].plot(time_data, u_tau_squared, '-', color='teal', linewidth=1.5, label=r'$u_\tau^2$')
    axes[1, 1].set_xlabel(r'$t$')
    axes[1, 1].set_ylabel(r'$u_\tau^2$ and $f_x$')
    axes[1, 1].set_title(r'$u_\tau^2$ and Forcing vs Time')
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)
    axes[1, 1].legend(frameon=False)

    save_path = os.path.join(results_folder, f'{prefix}_timeseries.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved time series plots to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Post-process DNS simulation results.")
    parser.add_argument('file_or_dir', type=str, nargs='?', default=None,
                        help='Path to timeseries file (with --timeseries-only) or field .npz file')
    parser.add_argument('--file', type=str, default=None, help='Path to field .npz file (deprecated, use positional argument)')
    parser.add_argument('--out', type=str, default=None, help='Output folder (default: same as input directory)')
    parser.add_argument('--timeseries-only', action='store_true', help='Only plot time series data')
    parser.add_argument('--t-threshold', type=float, default=50.0, help='Time threshold for averaging statistics (default: 50.0)')
    parser.add_argument('--config', type=str, default=None, help='Path to config file (optional with --timeseries-only)')
    parser.add_argument('--Re', type=float, default=None, help='Reynolds number (1/nu) to override config file value')
    parser.add_argument('--sphere', action='store_true', help='Overlay sphere mask on slices (reads center/radius from config)')
    parser.add_argument('--canopy_h', type=float, default=None, help='Z-level for XY slice (e.g., 0.25 for canopy top)')
    args = parser.parse_args()

    # Handle timeseries-only mode
    if args.timeseries_only:
        # Determine timeseries file path
        if args.file_or_dir:
            timeseries_path = args.file_or_dir
        elif args.out:
            # Assume timeseries.npz or timeseries.csv is in --out directory
            timeseries_path = os.path.join(args.out, 'timeseries.npz')
            if not os.path.exists(timeseries_path):
                timeseries_path = os.path.join(args.out, 'timeseries.csv')
        else:
            # Default to results folder
            timeseries_path = os.path.join('results', 'timeseries.npz')
            if not os.path.exists(timeseries_path):
                timeseries_path = os.path.join('results', 'timeseries.csv')

        # Determine output directory for plots
        if args.out:
            results_folder = args.out
        else:
            # Use directory containing timeseries file
            results_folder = os.path.dirname(timeseries_path)
            if not results_folder:
                results_folder = '.'

        # Determine nu
        if args.Re is not None:
            nu = 1.0 / args.Re
            print(f"Using Reynolds number from command line: Re = {args.Re}")
            print(f"Computed nu = 1/Re = {nu:.6e}")
        elif args.config is not None:
            nu = read_nu_from_config(args.config)
        else:
            # Use default value
            nu = 1e-4
            print(f"Using default nu = {nu:.6e} (specify --Re or --config to override)")

        # Plot time series
        plot_timeseries(results_folder, t_threshold=args.t_threshold, nu=nu)

    else:
        # Full post-processing mode (requires config)
        # Determine field file path
        if args.file_or_dir:
            field_file = args.file_or_dir
        elif args.file:
            field_file = args.file
        else:
            field_file = 'results/fields.npz'

        # Determine output directory
        results_folder = args.out if args.out else os.path.dirname(field_file)
        if not results_folder:
            results_folder = '.'

        # Config is required for full processing
        config_file = args.config if args.config else 'config.yaml'

        # Read nu from config file or use command line override
        if args.Re is not None:
            nu = 1.0 / args.Re
            print(f"Using Reynolds number from command line: Re = {args.Re}")
            print(f"Computed nu = 1/Re = {nu:.6e}")
        else:
            nu = read_nu_from_config(config_file)

        # Plot time series if available
        plot_timeseries(results_folder, t_threshold=args.t_threshold, nu=nu)

        # Load sphere parameters if requested
        sphere_params = None
        if args.sphere:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            print("WARNING: --sphere flag is no longer supported (IBM removed)")

        # Plot field data
        u, v, w, p, z_c, z_f, Lx, Ly, time, step = load_fields(field_file)
        plot_slices(u, v, w, z_c, Lx, Ly, results_folder, time=time, step=step, 
                   sphere_params=sphere_params, canopy_h=args.canopy_h)
        plot_profiles(u, z_c, results_folder)

if __name__ == "__main__":
    main()
