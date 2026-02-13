"""
Test script to validate workspace preallocation in projection_fft.py

This test compares results between:
1. Current version (allocates each call)
2. Preallocated workspace version

Checks:
- Numerical accuracy (should match to machine precision)
- Multiple consecutive solves (catches workspace reuse bugs)
- GPU-CPU synchronization issues
- Divergence quality
"""

import torch
import numpy as np
from projection_fft import initialize_fft_solver, solve_poisson_fft
from utils import compute_divergence

def create_test_divergence(nx, ny, nz, device='cpu'):
    """Create a test divergence field with known structure"""
    # Create a smooth divergence field (easier to debug)
    x = torch.linspace(0, 2*np.pi, nx, device=device)
    y = torch.linspace(0, 2*np.pi, ny, device=device)
    z = torch.linspace(0, 1, nz, device=device)

    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

    # Smooth test pattern: sum of sines
    div = (torch.sin(2*X) * torch.cos(3*Y) * torch.sin(np.pi*Z) +
           torch.cos(X) * torch.sin(2*Y) * torch.cos(np.pi*Z))

    return div

def solve_poisson_fft_preallocated(div, fft_data):
    """
    PREALLOCATED VERSION for testing
    Uses workspace tensors to avoid allocations
    """
    nx = fft_data['nx']
    ny = fft_data['ny']
    nz = fft_data['nz']
    tri_a = fft_data['tri_a']
    tri_b = fft_data['tri_b']
    tri_c = fft_data['tri_c']

    # Get preallocated workspaces
    workspace_p = fft_data['workspace_p']

    # FFT in x and y directions
    div_hat = torch.fft.rfft2(div, dim=(0, 1))

    # div_hat shape is (nx, ny//2+1, nz)
    nkx, nky = div_hat.shape[0], div_hat.shape[1]

    # Pin zero mode to resolve Neumann BC singularity
    div_hat[0, 0, :] = 0.0

    # Flatten batch dimensions
    div_hat_flat = div_hat.reshape(-1, nz)
    tri_a_flat = tri_a.reshape(-1, nz)
    tri_b_flat = tri_b.reshape(-1, nz)
    tri_c_flat = tri_c.reshape(-1, nz)

    # Solve all tridiagonal systems in parallel
    from projection_fft import solve_tridiagonal
    p_hat_flat = solve_tridiagonal(tri_a_flat, tri_b_flat, tri_c_flat, div_hat_flat)

    # Reshape back
    p_hat = p_hat_flat.reshape(nkx, nky, nz)

    # Inverse FFT
    p_interior = torch.fft.irfft2(p_hat, s=(nx, ny), dim=(0, 1))

    # CRITICAL: Clear workspace before reuse
    workspace_p.zero_()

    # Fill interior
    workspace_p[1:nx+1, 1:ny+1, 1:nz+1] = p_interior

    # Fused boundary conditions
    workspace_p[[0, nx+1], :, :] = workspace_p[[nx, 1], :, :]
    workspace_p[:, [0, ny+1], :] = workspace_p[:, [ny, 1], :]
    workspace_p[:, :, [0, nz+1]] = workspace_p[:, :, [1, nz]]

    return workspace_p

def initialize_fft_solver_with_workspace(nx, ny, nz, dx, dy, dz_c, dz_f):
    """
    Initialize FFT solver with preallocated workspace
    """
    # Call original initialization
    from projection_fft import initialize_fft_solver as init_original
    fft_data = init_original(nx, ny, nz, dx, dy, dz_c, dz_f)

    # Add workspace tensor
    device = dz_c.device
    fft_data['workspace_p'] = torch.zeros(nx+2, ny+2, nz+2, device=device)

    return fft_data

def test_preallocation(nx=64, ny=64, nz=32, device='cpu', n_iterations=5):
    """
    Main test function

    Args:
        nx, ny, nz: Grid size
        device: 'cpu' or 'cuda'
        n_iterations: Number of consecutive solves to test
    """
    print(f"=" * 70)
    print(f"Testing Workspace Preallocation")
    print(f"Grid: {nx}×{ny}×{nz}")
    print(f"Device: {device}")
    print(f"Iterations: {n_iterations}")
    print(f"=" * 70)

    # Setup grid
    Lx, Ly, Lz = 2*np.pi, 2*np.pi, 1.0
    dx = Lx / nx
    dy = Ly / ny

    # Uniform z-grid for simplicity
    dz_f = torch.ones(nz, device=device) * (Lz / nz)
    dz_c = torch.ones(nz+1, device=device) * (Lz / nz)

    # Initialize solvers
    print("\n1. Initializing solvers...")
    fft_data_original = initialize_fft_solver(nx, ny, nz, dx, dy, dz_c, dz_f)
    fft_data_preallocated = initialize_fft_solver_with_workspace(nx, ny, nz, dx, dy, dz_c, dz_f)
    print("   ✓ Both solvers initialized")

    # Run multiple iterations
    max_error = 0.0
    all_passed = True

    for iteration in range(n_iterations):
        print(f"\n2. Iteration {iteration + 1}/{n_iterations}")

        # Create test divergence (different each iteration)
        div = create_test_divergence(nx, ny, nz, device=device) * (iteration + 1)
        print(f"   Divergence range: [{div.min():.6e}, {div.max():.6e}]")

        # Solve with original version
        p_original = solve_poisson_fft(div.clone(), fft_data_original)

        # Solve with preallocated version
        p_preallocated = solve_poisson_fft_preallocated(div.clone(), fft_data_preallocated)

        # Force GPU synchronization if using CUDA
        if device == 'cuda':
            torch.cuda.synchronize()

        # Check for NaN values first
        has_nan_original = torch.isnan(p_original).any().item()
        has_nan_preallocated = torch.isnan(p_preallocated).any().item()

        if has_nan_original or has_nan_preallocated:
            print(f"   ❌ FAILED: NaN detected!")
            print(f"   - Original has NaN: {has_nan_original}")
            print(f"   - Preallocated has NaN: {has_nan_preallocated}")
            all_passed = False
            continue

        # Compare results
        abs_diff = torch.abs(p_original - p_preallocated)
        rel_diff = abs_diff / (torch.abs(p_original) + 1e-15)

        max_abs_error = abs_diff.max().item()
        max_rel_error = rel_diff.max().item()
        mean_abs_error = abs_diff.mean().item()

        max_error = max(max_error, max_abs_error)

        print(f"   Max absolute error: {max_abs_error:.6e}")
        print(f"   Mean absolute error: {mean_abs_error:.6e}")
        print(f"   Max relative error: {max_rel_error:.6e}")

        # Check if results match (strict tolerance for pressure)
        tolerance_abs = 1e-10  # Very strict
        tolerance_rel = 1e-8

        if max_abs_error > tolerance_abs or max_rel_error > tolerance_rel:
            print(f"   ❌ FAILED: Errors exceed tolerance!")
            all_passed = False

            # Detailed diagnostics
            print(f"\n   Diagnostics:")
            print(f"   - Original pressure range: [{p_original.min():.6e}, {p_original.max():.6e}]")
            print(f"   - Preallocated pressure range: [{p_preallocated.min():.6e}, {p_preallocated.max():.6e}]")

            # Find location of max error
            max_idx = torch.argmax(abs_diff)
            max_loc = torch.unravel_index(max_idx, abs_diff.shape)
            print(f"   - Max error location: {max_loc}")
            print(f"   - Original value: {p_original[max_loc]:.6e}")
            print(f"   - Preallocated value: {p_preallocated[max_loc]:.6e}")
        else:
            print(f"   ✓ PASSED: Results match within tolerance")

    print(f"\n" + "=" * 70)
    if all_passed:
        print(f"✓ ALL TESTS PASSED")
        print(f"  Maximum error across all iterations: {max_error:.6e}")
    else:
        print(f"❌ TESTS FAILED")
        print(f"  Preallocated version produces different results!")
    print(f"=" * 70)

    return all_passed

def test_workspace_reuse_bug():
    """
    Specific test for workspace reuse corruption
    Tests if workspace is properly cleared between calls
    """
    print(f"\n" + "=" * 70)
    print(f"Testing Workspace Reuse (Corruption Detection)")
    print(f"=" * 70)

    nx, ny, nz = 32, 32, 16
    device = 'cpu'

    # Setup
    dx = dy = 2*np.pi / nx
    dz_f = torch.ones(nz, device=device) * (1.0 / nz)
    dz_c = torch.ones(nz+1, device=device) * (1.0 / nz)

    fft_data = initialize_fft_solver_with_workspace(nx, ny, nz, dx, dy, dz_c, dz_f)

    # First solve with non-zero divergence
    div1 = torch.randn(nx, ny, nz, device=device)
    p1 = solve_poisson_fft_preallocated(div1, fft_data)

    # Second solve with ZERO divergence
    div2 = torch.zeros(nx, ny, nz, device=device)
    p2 = solve_poisson_fft_preallocated(div2, fft_data)

    # Check for NaN
    if torch.isnan(p1).any() or torch.isnan(p2).any():
        print(f"\n❌ NaN detected in workspace reuse test!")
        print(f"  First solve has NaN: {torch.isnan(p1).any().item()}")
        print(f"  Second solve has NaN: {torch.isnan(p2).any().item()}")
        return False

    # Pressure for zero divergence should be near zero (up to constant)
    p2_normalized = p2 - p2.mean()
    max_p2 = torch.abs(p2_normalized).max().item()

    print(f"\nTest: Solve with zero divergence after non-zero solve")
    print(f"  First solve: max |p| = {torch.abs(p1).max().item():.6e}")
    print(f"  Second solve (zero div): max |p - mean(p)| = {max_p2:.6e}")

    if max_p2 < 1e-10:
        print(f"  ✓ PASSED: Workspace properly cleared")
        return True
    else:
        print(f"  ❌ FAILED: Workspace contaminated from previous solve")
        return False

if __name__ == "__main__":
    # Test on CPU first
    print("\n" + "=" * 70)
    print("PHASE 1: CPU Tests")
    print("=" * 70)

    cpu_passed = test_preallocation(nx=64, ny=64, nz=32, device='cpu', n_iterations=5)
    cpu_reuse_passed = test_workspace_reuse_bug()

    # Test on GPU if available
    if torch.cuda.is_available():
        print("\n\n" + "=" * 70)
        print("PHASE 2: GPU Tests")
        print("=" * 70)

        gpu_passed = test_preallocation(nx=128, ny=128, nz=64, device='cuda', n_iterations=5)

        print("\n\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        print(f"CPU Tests: {'✓ PASSED' if cpu_passed and cpu_reuse_passed else '❌ FAILED'}")
        print(f"GPU Tests: {'✓ PASSED' if gpu_passed else '❌ FAILED'}")
        print("=" * 70)
    else:
        print("\n\nGPU not available, skipping GPU tests")
        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)
        print(f"CPU Tests: {'✓ PASSED' if cpu_passed and cpu_reuse_passed else '❌ FAILED'}")
        print("=" * 70)
