#!/usr/bin/env python3
"""
3D Flow Structure Visualization using Q-criterion
Visualizes vortex structures around immersed obstacles using isosurfaces
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
from skimage import measure
import argparse
import yaml
import os

# Configure Matplotlib for LaTeX
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('axes', labelsize=14)
rc('axes', titlesize=16)
rc('xtick', labelsize=12)
rc('ytick', labelsize=12)
rc('legend', fontsize=12)
rc('figure', titlesize=18)


def load_fields(filepath):
    """Load velocity fields from .npz file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    data = np.load(filepath)
    u = data['u']
    v = data['v']
    w = data['w']
    z_c = data['z_c']
    z_f = data['z_f']

    if 'Lx' in data:
        Lx = data['Lx'].item()
        Ly = data['Ly'].item()
    else:
        print("WARNING: Lx/Ly not found in file. Assuming 2pi, pi.")
        Lx = 2 * np.pi
        Ly = np.pi

    time = data['time'].item() if 'time' in data else None
    step = data['step'].item() if 'step' in data else None

    print(f"Loaded fields from {filepath}")
    print(f"Shapes: u={u.shape}, v={v.shape}, w={w.shape}")
    print(f"Domain: Lx={Lx:.3f}, Ly={Ly:.3f}, Lz={z_c[-1]:.3f}")
    if time is not None:
        print(f"Time: t={time:.3f}")

    return u, v, w, z_c, z_f, Lx, Ly, time, step


def interpolate_to_cell_centers(u, v, w, nx, ny, nz):
    """
    Interpolate staggered velocity components to cell centers.

    From initflow.py field structure:
    - u: (nx+1, ny+2, nz+2) - no ghost in x (staggered), ghosts in y,z
    - v: (nx+2, ny+1, nz+2) - ghosts in x,z, no ghost in y (staggered)
    - w: (nx+2, ny+2, nz+1) - ghosts in x,y, no ghost in z (staggered)

    Returns cell-centered velocities on (nx, ny, nz) grid
    """
    # Extract interior points using slicing (no explicit indices)
    # u: remove ghosts in y,z only (x has no ghosts)
    u_interior = u[:, 1:-1, 1:-1]  # (nx+1, ny, nz)
    u_center = 0.5 * (u_interior[:-1, :, :] + u_interior[1:, :, :])  # (nx, ny, nz)

    # v: remove ghosts in x,z only (y has no ghosts)
    v_interior = v[1:-1, :, 1:-1]  # (nx, ny+1, nz)
    v_center = 0.5 * (v_interior[:, :-1, :] + v_interior[:, 1:, :])  # (nx, ny, nz)

    # w: remove ghosts in x,y only (z has no ghosts)
    w_interior = w[1:-1, 1:-1, :]  # (nx, ny, nz+1)
    w_center = 0.5 * (w_interior[:, :, :-1] + w_interior[:, :, 1:])  # (nx, ny, nz)

    return u_center, v_center, w_center


def compute_velocity_gradients(u, v, w, nx, ny, nz, dx, dy, dz_c):
    """
    Compute velocity gradients on collocated grid.

    Returns 9 gradient components:
    du/dx, du/dy, du/dz,
    dv/dx, dv/dy, dv/dz,
    dw/dx, dw/dy, dw/dz
    """
    # First interpolate to cell centers
    u_int, v_int, w_int = interpolate_to_cell_centers(u, v, w, nx, ny, nz)

    # Get actual shape after interpolation
    nx_int, ny_int, nz_int = u_int.shape

    # Initialize gradient arrays with actual interpolated dimensions
    dudx = np.zeros_like(u_int)
    dudy = np.zeros_like(u_int)
    dudz = np.zeros_like(u_int)

    dvdx = np.zeros_like(v_int)
    dvdy = np.zeros_like(v_int)
    dvdz = np.zeros_like(v_int)

    dwdx = np.zeros_like(w_int)
    dwdy = np.zeros_like(w_int)
    dwdz = np.zeros_like(w_int)

    # Central differences in x and y (uniform grids)
    # Interior points
    if nx_int > 2:
        dudx[1:-1, :, :] = (u_int[2:, :, :] - u_int[:-2, :, :]) / (2*dx)
        dvdx[1:-1, :, :] = (v_int[2:, :, :] - v_int[:-2, :, :]) / (2*dx)
        dwdx[1:-1, :, :] = (w_int[2:, :, :] - w_int[:-2, :, :]) / (2*dx)
        # Boundaries
        dudx[0, :, :] = (u_int[1, :, :] - u_int[0, :, :]) / dx
        dudx[-1, :, :] = (u_int[-1, :, :] - u_int[-2, :, :]) / dx
        dvdx[0, :, :] = (v_int[1, :, :] - v_int[0, :, :]) / dx
        dvdx[-1, :, :] = (v_int[-1, :, :] - v_int[-2, :, :]) / dx
        dwdx[0, :, :] = (w_int[1, :, :] - w_int[0, :, :]) / dx
        dwdx[-1, :, :] = (w_int[-1, :, :] - w_int[-2, :, :]) / dx

    if ny_int > 2:
        dudy[:, 1:-1, :] = (u_int[:, 2:, :] - u_int[:, :-2, :]) / (2*dy)
        dvdy[:, 1:-1, :] = (v_int[:, 2:, :] - v_int[:, :-2, :]) / (2*dy)
        dwdy[:, 1:-1, :] = (w_int[:, 2:, :] - w_int[:, :-2, :]) / (2*dy)
        # Boundaries
        dudy[:, 0, :] = (u_int[:, 1, :] - u_int[:, 0, :]) / dy
        dudy[:, -1, :] = (u_int[:, -1, :] - u_int[:, -2, :]) / dy
        dvdy[:, 0, :] = (v_int[:, 1, :] - v_int[:, 0, :]) / dy
        dvdy[:, -1, :] = (v_int[:, -1, :] - v_int[:, -2, :]) / dy
        dwdy[:, 0, :] = (w_int[:, 1, :] - w_int[:, 0, :]) / dy
        dwdy[:, -1, :] = (w_int[:, -1, :] - w_int[:, -2, :]) / dy

    # Central differences in z (non-uniform grid)
    if nz_int > 2:
        for k in range(1, nz_int-1):
            dz_m = dz_c[k]
            dz_p = dz_c[k+1]
            dudz[:, :, k] = (u_int[:, :, k+1] - u_int[:, :, k-1]) / (dz_m + dz_p)
            dvdz[:, :, k] = (v_int[:, :, k+1] - v_int[:, :, k-1]) / (dz_m + dz_p)
            dwdz[:, :, k] = (w_int[:, :, k+1] - w_int[:, :, k-1]) / (dz_m + dz_p)
        # Boundaries
        dudz[:, :, 0] = (u_int[:, :, 1] - u_int[:, :, 0]) / dz_c[1]
        dudz[:, :, -1] = (u_int[:, :, -1] - u_int[:, :, -2]) / dz_c[-1]
        dvdz[:, :, 0] = (v_int[:, :, 1] - v_int[:, :, 0]) / dz_c[1]
        dvdz[:, :, -1] = (v_int[:, :, -1] - v_int[:, :, -2]) / dz_c[-1]
        dwdz[:, :, 0] = (w_int[:, :, 1] - w_int[:, :, 0]) / dz_c[1]
        dwdz[:, :, -1] = (w_int[:, :, -1] - w_int[:, :, -2]) / dz_c[-1]

    return dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz


def compute_q_criterion(dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz):
    """
    Compute Q-criterion: Q = 0.5 * (||Omega||^2 - ||S||^2)

    Where:
    Omega is the rotation rate tensor (antisymmetric part of velocity gradient)
    S is the strain rate tensor (symmetric part of velocity gradient)

    Q > 0 indicates vortex-dominated regions
    """
    # Strain rate tensor S_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
    S11 = dudx
    S22 = dvdy
    S33 = dwdz
    S12 = 0.5 * (dudy + dvdx)
    S13 = 0.5 * (dudz + dwdx)
    S23 = 0.5 * (dvdz + dwdy)

    # Rotation rate tensor Omega_ij = 0.5 * (du_i/dx_j - du_j/dx_i)
    Omega12 = 0.5 * (dudy - dvdx)
    Omega13 = 0.5 * (dudz - dwdx)
    Omega23 = 0.5 * (dvdz - dwdy)

    # ||S||^2 = sum of all S_ij^2
    S_squared = S11**2 + S22**2 + S33**2 + 2*(S12**2 + S13**2 + S23**2)

    # ||Omega||^2 = sum of all Omega_ij^2
    # Note: Omega is antisymmetric, so Omega_ii = 0
    Omega_squared = 2*(Omega12**2 + Omega13**2 + Omega23**2)

    # Q-criterion
    Q = 0.5 * (Omega_squared - S_squared)

    return Q


def create_sphere_mask(nx, ny, nz, Lx, Ly, z_c, center, radius):
    """
    Create boolean mask for points inside sphere.

    Returns:
    --------
    mask : ndarray (nx, ny, nz)
        True for points inside sphere, False outside
    """
    # Create coordinate arrays
    x = np.linspace(Lx/(2*nx), Lx - Lx/(2*nx), nx)
    y = np.linspace(Ly/(2*ny), Ly - Ly/(2*ny), ny)
    z = z_c[1:-1]  # Interior grid points

    # Create 3D meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Distance from sphere center
    xc, yc, zc = center
    dist = np.sqrt((X - xc)**2 + (Y - yc)**2 + (Z - zc)**2)

    # Mask interior points (distance < radius)
    mask = dist <= radius

    return mask


def plot_q_isosurface(Q, x, y, z, q_level, mask=None, sphere_params=None,
                     output_file='q_criterion.png', title=None, elev=20, azim=45):
    """
    Plot Q-criterion isosurface in 3D with height-based coloring.

    Parameters:
    -----------
    Q : ndarray (nx, ny, nz)
        Q-criterion field
    x, y, z : 1D arrays
        Coordinate arrays
    q_level : float
        Isovalue for Q-criterion surface
    mask : ndarray (nx, ny, nz), optional
        Boolean mask (True = masked/hidden)
    sphere_params : dict, optional
        Sphere center and radius for visualization
    elev, azim : float
        Camera elevation and azimuth angles
    """
    # Apply mask if provided
    if mask is not None:
        Q_masked = Q.copy()
        Q_masked[mask] = -np.inf  # Set masked values to very negative
    else:
        Q_masked = Q

    print(f"Q-criterion range: [{np.min(Q):.2e}, {np.max(Q):.2e}]")
    print(f"Isosurface level: Q = {q_level:.2e}")

    # Use marching cubes to find isosurface
    try:
        verts, faces, normals, values = measure.marching_cubes(
            Q_masked, level=q_level, spacing=(x[1]-x[0], y[1]-y[0], (z[-1]-z[0])/(len(z)-1))
        )
    except (ValueError, RuntimeError) as e:
        print(f"ERROR: Could not generate isosurface at Q={q_level:.2e}")
        print(f"       {e}")
        print("       Try adjusting --q-level")
        return

    # Transform vertices to physical coordinates
    verts[:, 0] = verts[:, 0] + x[0]
    verts[:, 1] = verts[:, 1] + y[0]
    verts[:, 2] = verts[:, 2] + z[0]

    print(f"Generated isosurface: {len(verts)} vertices, {len(faces)} faces")

    # Create 3D plot
    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection='3d')

    # Compute face colors based on height (z-coordinate)
    # Use average z of the three vertices of each face
    face_z = np.mean(verts[faces, 2], axis=1)

    # Normalize to [0, 1] for colormap
    z_min, z_max = z[0], z[-1]
    face_colors_normalized = (face_z - z_min) / (z_max - z_min)

    # Use viridis colormap
    cmap = cm.get_cmap('viridis')
    face_colors = cmap(face_colors_normalized)

    # Create mesh for isosurface with height-based coloring
    mesh = Poly3DCollection(verts[faces], alpha=0.85, edgecolor='none')
    mesh.set_facecolors(face_colors)
    ax.add_collection3d(mesh)

    # Add colorbar for height
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array(face_z)
    mappable.set_clim(z_min, z_max)
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1, aspect=20)
    cbar.set_label(r'Height $z$', fontsize=14)

    # Add sphere if parameters provided
    if sphere_params is not None:
        xc, yc, zc = sphere_params['center']
        radius = sphere_params['radius']

        # Draw solid sphere surface
        u_sphere = np.linspace(0, 2*np.pi, 40)
        v_sphere = np.linspace(0, np.pi, 30)
        x_sphere = xc + radius * np.outer(np.cos(u_sphere), np.sin(v_sphere))
        y_sphere = yc + radius * np.outer(np.sin(u_sphere), np.sin(v_sphere))
        z_sphere = zc + radius * np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere))
        ax.plot_surface(x_sphere, y_sphere, z_sphere,
                       color='gray', alpha=1.0, edgecolor='black', linewidth=0.2, shade=True)

    # Set limits and invert x-axis to fix flipped direction
    ax.set_xlim(x[-1], x[0])  # Inverted for proper orientation
    ax.set_ylim(y[0], y[-1])
    ax.set_zlim(z[0], z[-1])

    # Labels with LaTeX
    ax.set_xlabel(r'$x$', fontsize=14)
    ax.set_ylabel(r'$y$', fontsize=14)
    ax.set_zlabel(r'$z$', fontsize=14)

    # Title with LaTeX
    if title:
        ax.set_title(title, fontsize=16)
    else:
        ax.set_title(r'Q-criterion Isosurface: $Q = ' + f'{q_level:.2e}' + r'$', fontsize=16)

    # Set viewing angle
    ax.view_init(elev=elev, azim=azim)

    # Equal aspect ratio
    ax.set_box_aspect([x[-1]-x[0], y[-1]-y[0], z[-1]-z[0]])

    # Save
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    print(f"Saved 3D visualization to {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize 3D flow structures using Q-criterion isosurfaces"
    )
    parser.add_argument('field_file', type=str,
                       help='Path to field .npz file')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file (for sphere parameters)')
    parser.add_argument('--q-level', type=float, default=None,
                       help='Q-criterion isovalue (default: auto-compute from percentile)')
    parser.add_argument('--percentile', type=float, default=99.0,
                       help='Percentile for auto Q-level (default: 99.0)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output filename (default: q_criterion.png)')
    parser.add_argument('--elev', type=float, default=20,
                       help='Camera elevation angle (default: 20)')
    parser.add_argument('--azim', type=float, default=45,
                       help='Camera azimuth angle (default: 45)')
    parser.add_argument('--no-sphere', action='store_true',
                       help='Do not overlay sphere wireframe')
    parser.add_argument('--no-mask', action='store_true',
                       help='Do not mask sphere interior')

    args = parser.parse_args()

    # Load fields
    print("="*70)
    print("3D FLOW STRUCTURE VISUALIZATION - Q-CRITERION")
    print("="*70)
    u, v, w, z_c, z_f, Lx, Ly, time, step = load_fields(args.field_file)

    # Infer grid dimensions from field shapes
    # From initflow.py: u=(nx+1, ny+2, nz+2), v=(nx+2, ny+1, nz+2), w=(nx+2, ny+2, nz+1)
    nx = u.shape[0] - 1  # u has nx+1 in x (staggered)
    ny = u.shape[1] - 2  # u has ny+2 (ghost cells in y)
    nz = w.shape[2] - 1  # w has nz+1 in z (staggered)

    dx = Lx / nx
    dy = Ly / ny
    dz_c = z_c[1:] - z_c[:-1]

    print(f"\nGrid dimensions: nx={nx}, ny={ny}, nz={nz}")

    # Load config for sphere parameters
    sphere_params = None
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

    else:
        print(f"WARNING: Config file not found: {args.config}")

    # Compute velocity gradients
    print("\nComputing velocity gradients...")
    dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz = \
        compute_velocity_gradients(u, v, w, nx, ny, nz, dx, dy, dz_c)

    # Compute Q-criterion
    print("Computing Q-criterion...")
    Q = compute_q_criterion(dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz)

    # Create sphere mask
    mask = None
    if sphere_params is not None and not args.no_mask:
        print("Creating sphere mask...")
        mask = create_sphere_mask(nx, ny, nz, Lx, Ly, z_c,
                                 sphere_params['center'], sphere_params['radius'])
        print(f"  Masked {np.sum(mask)} / {mask.size} points "
              f"({100*np.sum(mask)/mask.size:.1f}%)")

    # Determine Q-level
    if args.q_level is not None:
        q_level = args.q_level
    else:
        # Auto-compute from percentile
        if mask is not None:
            Q_valid = Q[~mask]
        else:
            Q_valid = Q.ravel()
        q_level = np.percentile(Q_valid, args.percentile)
        print(f"\nAuto-computed Q-level from {args.percentile}th percentile: {q_level:.2e}")

    # Create coordinate arrays
    x = np.linspace(dx/2, Lx - dx/2, nx)
    y = np.linspace(dy/2, Ly - dy/2, ny)
    z = z_c[1:-1]

    # Output filename
    if args.output is None:
        output_file = args.field_file.replace('.npz', '_q_criterion.png')
        if output_file == args.field_file:
            output_file = 'q_criterion.png'
    else:
        output_file = args.output

    # Title
    if time is not None:
        title = f'Q-criterion Isosurface at $t = {time:.3f}$'
    else:
        title = None

    # Plot
    print("\nGenerating 3D isosurface...")
    plot_q_isosurface(
        Q, x, y, z,
        q_level=q_level,
        mask=mask,
        sphere_params=sphere_params if not args.no_sphere else None,
        output_file=output_file,
        title=title,
        elev=args.elev,
        azim=args.azim
    )

    print("\n" + "="*70)
    print("Done!")
    print("="*70)


if __name__ == "__main__":
    main()
