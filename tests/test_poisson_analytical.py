import torch
import sys
import os

# Add parent directory to path to import solver modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from projection import build_poisson_matrix, solve_poisson
from utils import generate_grid

def test_analytical_poisson():
    print("=== Testing Direct Poisson Solver with Analytical Solution ===")
    
    # Grid parameters
    # NOTE: Direct solver uses dense matrix O(N^2) memory. 
    # 64^3 is too large (~270GB RAM). Reduced to 16^3 for validation.
    nx, ny, nz = 16, 16, 16 
    Lx, Ly, Lz = 1.0, 1.0, 1.0
    gamma = 1.5 # Stretched grid
    
    print(f"Grid: {nx}x{ny}x{nz}")
    print(f"Domain: {Lx}x{Ly}x{Lz}")
    
    # Generate grid
    z_f, z_c, dz_f, dz_c = generate_grid(gamma, nz, Lz)
    dx = Lx / nx
    dy = Ly / ny
    
    # Create meshgrid for analytical solution
    # x, y are uniform, z is stretched (z_c)
    x = torch.linspace(dx/2, Lx - dx/2, nx)
    y = torch.linspace(dy/2, Ly - dy/2, ny)
    
    # Create 3D meshgrid (indexing='ij' for x,y,z order)
    X, Y, Z = torch.meshgrid(x, y, z_c[1:-1], indexing='ij')
    
    # Analytical solution: p = cos(2*pi*x) * cos(2*pi*y) * cos(pi*z)
    # This satisfies Periodic in x,y and Neumann in z (dp/dz ~ sin(pi*z) = 0 at z=0,1)
    # Note: We assume L=1 for simplicity in the formula
    p_exact = torch.cos(2 * torch.pi * X / Lx) * torch.cos(2 * torch.pi * Y / Ly) * torch.cos(torch.pi * Z / Lz)
    
    # Analytical Laplacian (Source term f)
    # lap(p) = -( (2pi/Lx)^2 + (2pi/Ly)^2 + (pi/Lz)^2 ) * p
    k_x = 2 * torch.pi / Lx
    k_y = 2 * torch.pi / Ly
    k_z = torch.pi / Lz
    f = -(k_x**2 + k_y**2 + k_z**2) * p_exact
    
    # Build Poisson matrix
    print(f"Building Poisson matrix ({nx*ny*nz}x{nx*ny*nz})...")
    A = build_poisson_matrix(nx, ny, nz, dx, dy, dz_c, dz_f)
    print("Poisson matrix built.")
    
    # Solve Ap = f
    print("Solving Poisson equation...")
    # solve_poisson expects 'div' which is usually div/dt. Here we pass 'f'.
    # It returns p with ghost cells.
    p_num_padded = solve_poisson(A, f, nx, ny, nz)
    
    # Extract interior
    p_num = p_num_padded[1:-1, 1:-1, 1:-1]
    
    # Since pressure is unique up to a constant, align means
    p_exact_mean = torch.mean(p_exact)
    p_num_mean = torch.mean(p_num)
    p_num_aligned = p_num - p_num_mean + p_exact_mean
    
    # Compute error
    error = torch.abs(p_num_aligned - p_exact)
    l2_error = torch.sqrt(torch.mean(error**2))
    linf_error = torch.max(error)
    
    print(f"L2 Error: {l2_error:.6e}")
    print(f"L_inf Error: {linf_error:.6e}")
    
    # Check if error is acceptable (e.g., < 1.5e-2 for this resolution 16^3)
    # Expected error is O(dx^2) ~ (1/16)^2 ~ 0.004. L_inf might be slightly higher.
    if l2_error < 1.5e-2:
        print("TEST PASSED: Error is within acceptable limits.")
    else:
        print("TEST FAILED: Error is too high.")

    # Plot slices
    import matplotlib.pyplot as plt
    
    # Create results directory for tests if it doesn't exist
    test_results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(test_results_dir, exist_ok=True)
    
    # Slice indices (middle of the domain)
    iz = nz // 2
    iy = ny // 2
    ix = nx // 2
    
    # XY slice at middle Z
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(p_num_aligned[:, :, iz].T, origin='lower', extent=[0, Lx, 0, Ly])
    plt.colorbar()
    plt.title(f'Numerical (z={z_c[iz+1]:.2f})')
    
    plt.subplot(1, 3, 2)
    plt.imshow(p_exact[:, :, iz].T, origin='lower', extent=[0, Lx, 0, Ly])
    plt.colorbar()
    plt.title('Analytical')
    
    plt.subplot(1, 3, 3)
    plt.imshow(error[:, :, iz].T, origin='lower', extent=[0, Lx, 0, Ly])
    plt.colorbar()
    plt.title('Error')
    
    plt.savefig(os.path.join(test_results_dir, 'validation_slice_xy.png'))
    plt.close()
    
    # XZ slice at middle Y
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(p_num_aligned[:, iy, :].T, origin='lower', extent=[0, Lx, 0, Lz], aspect='auto')
    plt.colorbar()
    plt.title(f'Numerical (y={y[iy]:.2f})')
    
    plt.subplot(1, 3, 2)
    plt.imshow(p_exact[:, iy, :].T, origin='lower', extent=[0, Lx, 0, Lz], aspect='auto')
    plt.colorbar()
    plt.title('Analytical')
    
    plt.subplot(1, 3, 3)
    plt.imshow(error[:, iy, :].T, origin='lower', extent=[0, Lx, 0, Lz], aspect='auto')
    plt.colorbar()
    plt.title('Error')
    
    plt.savefig(os.path.join(test_results_dir, 'validation_slice_xz.png'))
    plt.close()
    
    print(f"Validation plots saved to {test_results_dir}")

if __name__ == "__main__":
    test_analytical_poisson()
