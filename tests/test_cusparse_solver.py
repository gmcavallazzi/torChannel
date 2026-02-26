"""
Test cuSPARSE tridiagonal solver against Thomas algorithm.

Compares solve_shared_coeffs and solve_per_system against the JIT Thomas solver.
Skips gracefully if cuSPARSE is not available.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
torch.set_default_dtype(torch.float64)


def test_cusparse_available():
    """Check if cuSPARSE is available."""
    if not torch.cuda.is_available():
        print("CUDA not available — skipping cuSPARSE tests")
        return False

    from tridiagonal_cusparse import is_available
    if not is_available():
        print("cuSPARSE library not found — skipping cuSPARSE tests")
        return False

    print("cuSPARSE is available")
    return True


def test_shared_coeffs():
    """Test solve_shared_coeffs matches Thomas algorithm."""
    from operators import solve_tridiagonal_batch
    from tridiagonal_cusparse import CuSparseTridiagonalSolver

    device = 'cuda'
    solver = CuSparseTridiagonalSolver()

    n = 64
    batch_size = 1024
    torch.manual_seed(123)

    # Create a well-conditioned tridiagonal system (diagonally dominant)
    a = -0.3 * torch.ones(n, device=device)
    b = 2.0 * torch.ones(n, device=device)
    c = -0.3 * torch.ones(n, device=device)
    a[0] = 0.0
    c[-1] = 0.0
    d = torch.randn(batch_size, n, device=device)

    # Thomas algorithm
    x_thomas = solve_tridiagonal_batch(a, b, c, d.clone())

    # cuSPARSE
    x_cusparse = solver.solve_shared_coeffs(a, b, c, d.clone())

    diff = torch.max(torch.abs(x_thomas - x_cusparse)).item()
    rel_diff = diff / (torch.max(torch.abs(x_thomas)).item() + 1e-30)
    print(f"  solve_shared_coeffs: max abs diff = {diff:.2e}, rel diff = {rel_diff:.2e}")
    assert diff < 1e-10, f"cuSPARSE shared_coeffs does not match Thomas! diff={diff:.2e}"
    return True


def test_per_system():
    """Test solve_per_system matches Thomas algorithm for Poisson-like systems."""
    from projection_fft import solve_tridiagonal
    from tridiagonal_cusparse import CuSparseTridiagonalSolver

    device = 'cuda'
    solver = CuSparseTridiagonalSolver()

    n = 64
    batch_size = 512
    torch.manual_seed(456)

    # Create per-system tridiagonal coefficients (like FFT Poisson solver)
    a = -0.3 * torch.ones(batch_size, n, device=device)
    b = 2.0 + 0.1 * torch.rand(batch_size, n, device=device)  # varying diagonal
    c = -0.3 * torch.ones(batch_size, n, device=device)
    a[:, 0] = 0.0
    c[:, -1] = 0.0
    d = torch.randn(batch_size, n, device=device)

    # Thomas algorithm
    x_thomas = solve_tridiagonal(a.clone(), b.clone(), c.clone(), d.clone())

    # cuSPARSE
    x_cusparse = solver.solve_per_system(a.clone(), b.clone(), c.clone(), d.clone())

    diff = torch.max(torch.abs(x_thomas - x_cusparse)).item()
    rel_diff = diff / (torch.max(torch.abs(x_thomas)).item() + 1e-30)
    print(f"  solve_per_system: max abs diff = {diff:.2e}, rel diff = {rel_diff:.2e}")
    assert diff < 1e-10, f"cuSPARSE per_system does not match Thomas! diff={diff:.2e}"
    return True


def test_implicit_diffusion_integration():
    """Test that cuSPARSE produces same result when used through solve_implicit_diffusion_*_ext."""
    from operators import (solve_implicit_diffusion_u_vectorized,
                           solve_implicit_diffusion_u_ext)
    from utils import generate_grid
    from solver import apply_bc_all
    from tridiagonal_cusparse import CuSparseTridiagonalSolver

    device = 'cuda'
    solver = CuSparseTridiagonalSolver()

    nx, ny, nz = 32, 32, 16
    gamma, Lz = 2.0, 2.0
    nu = 1e-4
    dt = 0.001

    z_f, z_c, dz_f, dz_c = generate_grid(gamma, nz, Lz, device=device)

    torch.manual_seed(42)
    u = torch.randn(nx+1, ny+2, nz+2, device=device)
    v = torch.randn(nx+2, ny+1, nz+2, device=device)
    w = torch.randn(nx+2, ny+2, nz+1, device=device)
    apply_bc_all(u, v, w, 'dirichlet')

    # Thomas
    u_thomas = u.clone()
    u_thomas = solve_implicit_diffusion_u_vectorized(u_thomas, dt, nx, ny, nz, dz_c, dz_f, nu,
                                                      top_wall_bc_type='dirichlet')

    # cuSPARSE via _ext
    u_cusparse = u.clone()
    u_cusparse = solve_implicit_diffusion_u_ext(u_cusparse, dt, nx, ny, nz, dz_c, dz_f, nu,
                                                 top_wall_bc_type='dirichlet',
                                                 trid_solver=solver)

    diff = torch.max(torch.abs(u_thomas - u_cusparse)).item()
    rel_diff = diff / (torch.max(torch.abs(u_thomas)).item() + 1e-30)
    print(f"  implicit_diffusion_u via cuSPARSE: max abs diff = {diff:.2e}, rel diff = {rel_diff:.2e}")
    assert diff < 1e-10, f"cuSPARSE implicit diffusion does not match Thomas! diff={diff:.2e}"
    return True


if __name__ == '__main__':
    print("Running cuSPARSE tridiagonal solver tests...")

    if not test_cusparse_available():
        print("\nAll cuSPARSE tests skipped (not available)")
        sys.exit(0)

    passed = 0
    failed = 0

    for test in [test_shared_coeffs, test_per_system, test_implicit_diffusion_integration]:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
