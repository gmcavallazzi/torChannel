"""
Test that vectorized implicit diffusion matches the original loop-based version.

Tests solve_implicit_diffusion_u/v/w with both Dirichlet and Neumann top-wall BCs,
comparing original (loop) and vectorized implementations to machine precision.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
torch.set_default_dtype(torch.float64)

from utils import generate_grid
from solver import apply_bc_all


def make_test_fields(nx, ny, nz, device='cpu'):
    """Create random test velocity fields with ghost cells."""
    torch.manual_seed(42)
    u = torch.randn(nx+1, ny+2, nz+2, device=device)
    v = torch.randn(nx+2, ny+1, nz+2, device=device)
    w = torch.randn(nx+2, ny+2, nz+1, device=device)
    return u, v, w


def test_vectorized_diffusion_u(top_wall_bc_type='dirichlet', device='cpu'):
    """Test vectorized solve_implicit_diffusion_u matches original."""
    from operators import solve_implicit_diffusion_u

    nx, ny, nz = 32, 32, 16
    gamma, Lz = 2.0, 2.0
    nu = 1e-4
    dt = 0.001
    theta = 0.5

    z_f, z_c, dz_f, dz_c = generate_grid(gamma, nz, Lz, device=device)

    # Create test fields and apply BCs
    u, v, w = make_test_fields(nx, ny, nz, device=device)
    apply_bc_all(u, v, w, top_wall_bc_type)

    # Run original (loop-based) version
    u_orig = u.clone()
    u_orig = solve_implicit_diffusion_u(u_orig, dt, nx, ny, nz, dz_c, dz_f, nu,
                                         theta=theta, top_wall_bc_type=top_wall_bc_type)

    # Run vectorized version
    from operators import solve_implicit_diffusion_u_vectorized
    u_vec = u.clone()
    u_vec = solve_implicit_diffusion_u_vectorized(u_vec, dt, nx, ny, nz, dz_c, dz_f, nu,
                                                    theta=theta, top_wall_bc_type=top_wall_bc_type)

    # Compare
    diff = torch.max(torch.abs(u_orig - u_vec)).item()
    rel_diff = diff / (torch.max(torch.abs(u_orig)).item() + 1e-30)
    print(f"  solve_implicit_diffusion_u ({top_wall_bc_type}): max abs diff = {diff:.2e}, rel diff = {rel_diff:.2e}")
    assert diff < 1e-12, f"Vectorized u does not match original! diff={diff:.2e}"
    return True


def test_vectorized_diffusion_v(top_wall_bc_type='dirichlet', device='cpu'):
    """Test vectorized solve_implicit_diffusion_v matches original."""
    from operators import solve_implicit_diffusion_v

    nx, ny, nz = 32, 32, 16
    gamma, Lz = 2.0, 2.0
    nu = 1e-4
    dt = 0.001
    theta = 0.5

    z_f, z_c, dz_f, dz_c = generate_grid(gamma, nz, Lz, device=device)

    u, v, w = make_test_fields(nx, ny, nz, device=device)
    apply_bc_all(u, v, w, top_wall_bc_type)

    # Original
    v_orig = v.clone()
    v_orig = solve_implicit_diffusion_v(v_orig, dt, nx, ny, nz, dz_c, dz_f, nu,
                                         theta=theta, top_wall_bc_type=top_wall_bc_type)

    # Vectorized
    from operators import solve_implicit_diffusion_v_vectorized
    v_vec = v.clone()
    v_vec = solve_implicit_diffusion_v_vectorized(v_vec, dt, nx, ny, nz, dz_c, dz_f, nu,
                                                    theta=theta, top_wall_bc_type=top_wall_bc_type)

    diff = torch.max(torch.abs(v_orig - v_vec)).item()
    rel_diff = diff / (torch.max(torch.abs(v_orig)).item() + 1e-30)
    print(f"  solve_implicit_diffusion_v ({top_wall_bc_type}): max abs diff = {diff:.2e}, rel diff = {rel_diff:.2e}")
    assert diff < 1e-12, f"Vectorized v does not match original! diff={diff:.2e}"
    return True


def test_vectorized_diffusion_w(device='cpu'):
    """Test vectorized solve_implicit_diffusion_w matches original."""
    from operators import solve_implicit_diffusion_w

    nx, ny, nz = 32, 32, 16
    gamma, Lz = 2.0, 2.0
    nu = 1e-4
    dt = 0.001
    theta = 0.5

    z_f, z_c, dz_f, dz_c = generate_grid(gamma, nz, Lz, device=device)

    u, v, w = make_test_fields(nx, ny, nz, device=device)
    apply_bc_all(u, v, w, 'dirichlet')

    # Original
    w_orig = w.clone()
    w_orig = solve_implicit_diffusion_w(w_orig, dt, nx, ny, nz, dz_c, dz_f, nu, theta=theta)

    # Vectorized
    from operators import solve_implicit_diffusion_w_vectorized
    w_vec = w.clone()
    w_vec = solve_implicit_diffusion_w_vectorized(w_vec, dt, nx, ny, nz, dz_c, dz_f, nu, theta=theta)

    diff = torch.max(torch.abs(w_orig - w_vec)).item()
    rel_diff = diff / (torch.max(torch.abs(w_orig)).item() + 1e-30)
    print(f"  solve_implicit_diffusion_w: max abs diff = {diff:.2e}, rel diff = {rel_diff:.2e}")
    assert diff < 1e-12, f"Vectorized w does not match original! diff={diff:.2e}"
    return True


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running vectorized diffusion tests on {device}...")

    passed = 0
    failed = 0

    for bc in ['dirichlet', 'neumann']:
        try:
            test_vectorized_diffusion_u(bc, device)
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

        try:
            test_vectorized_diffusion_v(bc, device)
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    try:
        test_vectorized_diffusion_w(device)
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
