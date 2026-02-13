import torch
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import generate_grid
from projection import build_poisson_matrix, solve_poisson
from projection_fft import initialize_fft_solver, solve_poisson_fft

# Set double precision
torch.set_default_dtype(torch.float64)

def test_fft_vs_direct():
    print("="*60)
    print("TEST: FFT Solver vs Direct Solver")
    print("="*60)
    
    # Test parameters
    nx, ny, nz = 8, 8, 16
    Lx, Ly, Lz = 0.1, 0.1, 2.0
    dx, dy = Lx/nx, Ly/ny
    
    # Generate grid
    gamma = 1.5
    z_f, z_c, dz_f, dz_c = generate_grid(gamma, nz, Lz)
    
    print(f"Grid: nx={nx}, ny={ny}, nz={nz}")
    
    # 1. Build Direct Solver Matrix
    print("\nBuilding Direct Solver Matrix...")
    A = build_poisson_matrix(nx, ny, nz, dx, dy, dz_c, dz_f)
    
    # 2. Initialize FFT Solver
    print("Initializing FFT Solver...")
    fft_data = initialize_fft_solver(nx, ny, nz, dx, dy, dz_c, dz_f)
    
    # 3. Create a deterministic divergence field
    # Use a smooth function instead of random noise to ensure test reliability
    x = torch.linspace(dx/2, Lx-dx/2, nx)
    y = torch.linspace(dy/2, Ly-dy/2, ny)
    z = z_c[1:-1] # Interior z points
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    
    # A simple smooth function that satisfies periodic BCs in x,y
    div = torch.sin(2*torch.pi*X/Lx) * torch.cos(2*torch.pi*Y/Ly) * torch.sin(torch.pi*Z/Lz)
    
    # Ensure mean is zero for compatibility
    div = div - torch.mean(div)
    
    print(f"Divergence field: min={torch.min(div):.6e}, max={torch.max(div):.6e}, mean={torch.mean(div):.6e}")
    
    # 4. Solve with Direct Solver
    print("\nSolving with Direct Solver...")
    p_direct = solve_poisson(A, div, nx, ny, nz)
    
    # 5. Solve with FFT Solver
    print("Solving with FFT Solver...")
    p_fft = solve_poisson_fft(div, fft_data)
    
    # 6. Compare Results
    p_direct_centered = p_direct - torch.mean(p_direct)
    p_fft_centered = p_fft - torch.mean(p_fft)
    
    # Check interior only
    p_direct_int = p_direct_centered[1:nx+1, 1:ny+1, 1:nz+1]
    p_fft_int = p_fft_centered[1:nx+1, 1:ny+1, 1:nz+1]
    diff_int = torch.abs(p_direct_int - p_fft_int)
    max_diff_int = torch.max(diff_int)
    
    print(f"\nComparison (p - mean(p)):")
    print(f"  Max difference (interior): {max_diff_int:.6e}")
    
    # 7. Check Residuals: ||A*p - b||
    # Extract interior
    p_direct_int_raw = p_direct[1:nx+1, 1:ny+1, 1:nz+1]
    p_fft_int_raw = p_fft[1:nx+1, 1:ny+1, 1:nz+1]
    
    # Permute to match matrix indexing (i varies fastest)
    p_direct_flat = p_direct_int_raw.permute(2, 1, 0).flatten()
    p_fft_flat = p_fft_int_raw.permute(2, 1, 0).flatten()
    
    # Prepare b
    b_flat = div.permute(2, 1, 0).flatten()
    
    # Compute residuals
    res_direct = torch.norm(torch.matmul(A, p_direct_flat) - b_flat)
    res_fft = torch.norm(torch.matmul(A, p_fft_flat) - b_flat)
    
    print(f"\nResiduals ||Ax - b||:")
    print(f"  Direct Solver: {res_direct:.6e}")
    print(f"  FFT Solver:    {res_fft:.6e}")
    
    if max_diff_int < 1e-9:
        print("\n✓ PASS: Solvers match to high precision.")
    else:
        print("\n✗ FAILURE: Solvers disagree.")

    # ============================================================
    # ANALYTICAL SOLUTION TEST
    # ============================================================
    print("\n" + "="*60)
    print("TEST: Analytical Solution")
    print("="*60)
    
    # Define analytical pressure: p = cos(2pi x/Lx) * cos(2pi y/Ly) * cos(pi z/Lz)
    # This satisfies Neumann BCs at z=0 and z=Lz (dp/dz ~ sin(pi z/Lz) = 0)
    # And periodic in x, y
    
    x = torch.linspace(dx/2, Lx-dx/2, nx)
    y = torch.linspace(dy/2, Ly-dy/2, ny)
    # z_c has ghost cells (length nz+2). We need interior only for RHS generation.
    z_c_int = z_c[1:-1]
    
    X, Y, Z = torch.meshgrid(x, y, z_c_int, indexing='ij')
    
    kx = 2 * torch.pi / Lx
    ky = 2 * torch.pi / Ly
    kz = torch.pi / Lz
    
    p_exact = torch.cos(kx * X) * torch.cos(ky * Y) * torch.cos(kz * Z)
    
    # Analytical Laplacian
    # lap(p) = -(kx^2 + ky^2 + kz^2) * p
    lap_p_exact = -(kx**2 + ky**2 + kz**2) * p_exact
    
    # Solve A * p = lap_p_exact
    # Note: In our code, we solve A * p = div / dt. So here 'div' is lap_p_exact * dt.
    # Let's just pass lap_p_exact as 'div' and assume dt=1 for this test.
    rhs = lap_p_exact
    
    print("Solving for analytical RHS...")
    p_num_direct = solve_poisson(A, rhs, nx, ny, nz)
    p_num_fft = solve_poisson_fft(rhs, fft_data)
    
    # Center solutions
    p_exact_centered = p_exact - torch.mean(p_exact)
    p_num_direct_centered = p_num_direct[1:nx+1, 1:ny+1, 1:nz+1] - torch.mean(p_num_direct[1:nx+1, 1:ny+1, 1:nz+1])
    p_num_fft_centered = p_num_fft[1:nx+1, 1:ny+1, 1:nz+1] - torch.mean(p_num_fft[1:nx+1, 1:ny+1, 1:nz+1])
    
    # Compare
    err_direct = torch.max(torch.abs(p_num_direct_centered - p_exact_centered))
    err_fft = torch.max(torch.abs(p_num_fft_centered - p_exact_centered))
    
    print(f"Max Error (Direct vs Exact): {err_direct:.6e}")
    print(f"Max Error (FFT vs Exact):    {err_fft:.6e}")
    
    # Plotting
    print("\nPlotting comparison...")
    # Take a slice at x=Lx/2, y=Ly/2 (indices nx//2, ny//2)
    idx_x = nx // 2
    idx_y = ny // 2
    
    z_np = z_c_int.numpy()
    p_exact_line = p_exact_centered[idx_x, idx_y, :].numpy()
    p_direct_line = p_num_direct_centered[idx_x, idx_y, :].numpy()
    p_fft_line = p_num_fft_centered[idx_x, idx_y, :].numpy()
    
    plt.figure(figsize=(10, 6))
    plt.plot(z_np, p_exact_line, 'k-', linewidth=2, label='Exact')
    plt.plot(z_np, p_direct_line, 'r--', linewidth=2, label='Direct Solver')
    plt.plot(z_np, p_fft_line, 'b:', linewidth=2, label='FFT Solver')
    
    plt.xlabel('z')
    plt.ylabel('Pressure (centered)')
    plt.title(f'Pressure Profile Comparison at x={x[idx_x]:.2f}, y={y[idx_y]:.2f}')
    plt.legend()
    plt.grid(True)
    plt.savefig('comparison_analytical.png')
    print("Saved plot to comparison_analytical.png")

if __name__ == "__main__":
    test_fft_vs_direct()
