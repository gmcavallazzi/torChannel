import torch
import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to path to import solver modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from operators import diffusion_u, advection_u
from utils import generate_grid

def test_advection_diffusion():
    print("=== Testing Advection and Diffusion Operators ===")
    
    # Grid parameters
    nx, ny, nz = 64, 64, 64
    Lx, Ly, Lz = 1.0, 1.0, 1.0
    gamma = 0.1 # Nearly uniform grid
    nu = 0.01   # Viscosity
    
    print(f"Grid: {nx}x{ny}x{nz}, Gamma: {gamma}, Nu: {nu}")
    
    # Generate grid
    z_f, z_c, dz_f, dz_c = generate_grid(gamma, nz, Lz)
    dx = Lx / nx
    dy = Ly / ny
    
    # Coordinates
    x_face = torch.linspace(0, Lx, nx+1)
    x_center = 0.5 * (x_face[:-1] + x_face[1:])
    
    y_face = torch.linspace(0, Ly, ny+1)
    y_center = 0.5 * (y_face[:-1] + y_face[1:])
    
    # Create meshgrids for u, v, w locations
    # u: (x_face, y_center, z_center)
    # v: (x_center, y_face, z_center)
    # w: (x_center, y_center, z_face)
    
    # We only need u-grid for testing diffusion_u and advection_u (output location)
    # But we need v and w fields for advection_u input.
    
    # Define analytical functions
    # u = sin(kx*x) * sin(ky*y) * sin(kz*z)
    # v = sin(kx*x) * sin(ky*y) * sin(kz*z)
    # w = sin(kx*x) * sin(ky*y) * sin(kz*z)
    kx = 2 * torch.pi / Lx
    ky = 2 * torch.pi / Ly
    kz = torch.pi / Lz
    
    def get_field(x, y, z):
        # x, y, z are tensors of shape (nx, ny, nz) or broadcastable
        return torch.sin(kx * x) * torch.sin(ky * y) * torch.sin(kz * z)
    
    # Initialize fields with ghost cells
    u = torch.zeros(nx+1, ny+2, nz+2)
    v = torch.zeros(nx+2, ny+1, nz+2)
    w = torch.zeros(nx+2, ny+2, nz+1)
    
    # Fill u (interior + boundary)
    # u grid: x[0..nx], y[1..ny], z[1..nz]
    # Note: u has shape (nx+1, ny+2, nz+2). 
    # Physical domain is u[0:nx+1, 1:ny+1, 1:nz+1]
    # We must fill ghosts too for correct derivatives at boundaries
    
    # Full grids for u
    X_u_full, Y_u_full, Z_u_full = torch.meshgrid(x_face, y_center, z_c, indexing='ij')
    # z_c has length nz+2. x_face nx+1. y_center ny.
    # u shape (nx+1, ny+2, nz+2).
    # We need to handle y-ghosts too.
    # y_center has length ny.
    # Let's create full coordinate arrays including ghosts.
    
    # y_center_full: add ghosts
    dy = y_face[1] - y_face[0]
    y_center_full = torch.cat([torch.tensor([y_center[0] - dy]), y_center, torch.tensor([y_center[-1] + dy])])
    
    X_u, Y_u, Z_u = torch.meshgrid(x_face, y_center_full, z_c, indexing='ij')
    u = get_field(X_u, Y_u, Z_u)
    
    # Fill v
    # v grid: x_center (needs ghosts), y_face, z_c
    dx = x_face[1] - x_face[0]
    x_center_full = torch.cat([torch.tensor([x_center[0] - dx]), x_center, torch.tensor([x_center[-1] + dx])])
    
    X_v, Y_v, Z_v = torch.meshgrid(x_center_full, y_face, z_c, indexing='ij')
    v = get_field(X_v, Y_v, Z_v)
    
    # Fill w
    # w grid: x_center, y_center, z_f (needs ghosts?)
    # w is at z-faces. z_f has length nz+1.
    # w shape (nx+2, ny+2, nz+1).
    # z_f covers 0..Lz.
    # We need x and y ghosts.
    
    X_w, Y_w, Z_w = torch.meshgrid(x_center_full, y_center_full, z_f, indexing='ij')
    w = get_field(X_w, Y_w, Z_w)
    
    # --- Test Diffusion U ---
    print("\nTesting Diffusion U...")
    diff_u_num = diffusion_u(u, nx, ny, nz, dx, dy, dz_c, dz_f, nu)
    
    # Analytical Diffusion: nu * lap(u)
    # lap(u) = -(kx^2 + ky^2 + kz^2) * u
    # Evaluate at interior u-nodes
    u_int = u[1:nx, 1:ny+1, 1:nz+1]
    lap_u_exact = -(kx**2 + ky**2 + kz**2) * u_int
    diff_u_exact = nu * lap_u_exact
    
    # Compare only interior u[1:nx, 1:ny+1, 1:nz+1]
    diff_u_num_int = diff_u_num[1:nx, 1:ny+1, 1:nz+1]
    
    err_diff = torch.abs(diff_u_num_int - diff_u_exact)
    l2_diff = torch.sqrt(torch.mean(err_diff**2))
    print(f"Diffusion L2 Error: {l2_diff:.6e}")
    
    # --- Test Advection U ---
    print("\nTesting Advection U...")
    adv_u_num = advection_u(u, v, w, nx, ny, nz, dx, dy, dz_f)
    
    # Analytical Advection (Conservative Form): d(uu)/dx + d(vu)/dy + d(wu)/dz
    # u = sin(kx x) sin(ky y) sin(kz z)
    # v = ...
    # w = ...
    
    # d(uu)/dx = d/dx (sin^2(kx x) ...) = kx sin(2 kx x) sin^2(ky y) sin^2(kz z)
    # d(vu)/dy = d/dy (sin^2(ky y) ...) = ky sin(2 ky y) sin^2(kx x) sin^2(kz z)
    # d(wu)/dz = d/dz (sin^2(kz z) ...) = kz sin(2 kz z) sin^2(kx x) sin^2(ky y)
    
    # Evaluate at u-nodes (interior)
    # u-nodes: x_face[1:nx], y_center, z_c[1:nz+1]
    
    X_int = X_u[1:nx, 1:ny+1, 1:nz+1]
    Y_int = Y_u[1:nx, 1:ny+1, 1:nz+1]
    Z_int = Z_u[1:nx, 1:ny+1, 1:nz+1]
    
    duu_dx = kx * torch.sin(2 * kx * X_int) * torch.sin(ky * Y_int)**2 * torch.sin(kz * Z_int)**2
    dvu_dy = ky * torch.sin(2 * ky * Y_int) * torch.sin(kx * X_int)**2 * torch.sin(kz * Z_int)**2
    dwu_dz = kz * torch.sin(2 * kz * Z_int) * torch.sin(kx * X_int)**2 * torch.sin(ky * Y_int)**2
    
    adv_u_exact_int = duu_dx + dvu_dy + dwu_dz
    
    adv_u_num_int = adv_u_num[1:nx, 1:ny+1, 1:nz+1]
    
    err_adv = torch.abs(adv_u_num_int - adv_u_exact_int)
    l2_adv = torch.sqrt(torch.mean(err_adv**2))
    print(f"Advection L2 Error: {l2_adv:.6e}")
    
    # Plot slices
    test_results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(test_results_dir, exist_ok=True)
    
    iz = nz // 2
    
    # Diffusion Plot
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(diff_u_num_int[:, :, iz].T, origin='lower')
    plt.title('Numerical Diffusion')
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow(diff_u_exact[:, :, iz].T, origin='lower')
    plt.title('Analytical Diffusion')
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow(err_diff[:, :, iz].T, origin='lower')
    plt.title('Error')
    plt.colorbar()
    plt.savefig(os.path.join(test_results_dir, 'test_diffusion_u.png'))
    plt.close()
    
    # Advection Plot
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(adv_u_num_int[:, :, iz].T, origin='lower')
    plt.title('Numerical Advection')
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow(adv_u_exact_int[:, :, iz].T, origin='lower')
    plt.title('Analytical Advection')
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow(err_adv[:, :, iz].T, origin='lower')
    plt.title('Error')
    plt.colorbar()
    plt.savefig(os.path.join(test_results_dir, 'test_advection_u.png'))
    plt.close()
    
    print(f"Plots saved to {test_results_dir}")

if __name__ == "__main__":
    test_advection_diffusion()
