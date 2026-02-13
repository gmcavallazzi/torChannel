"""
Test script to verify enhanced fused kernels produce correct results.

This script compares the output of the new fused kernels with the original
separate kernel implementations to ensure numerical correctness.
"""

import torch
import numpy as np
from utils import generate_grid
import operators
from operators import (
    advection_u, advection_v, advection_w,
    diffusion_u, diffusion_v, diffusion_w,
    diffusion_xy_u, diffusion_xy_v, diffusion_xy_w,
    compute_momentum_rhs_fused_v2,
    compute_momentum_rhs_fused_imex
)

def test_fused_kernel_v2():
    """Test the enhanced fused kernel v2 for AB2 scheme."""
    print("=" * 80)
    print("Testing compute_momentum_rhs_fused_v2 (AB2 scheme)")
    print("=" * 80)

    # Setup test grid
    nx, ny, nz = 32, 32, 32
    Lx, Ly, Lz = 2.67, 0.8, 2.0
    gamma = 2.5
    nu = 1.0 / 2870.0

    dx = Lx / nx
    dy = Ly / ny
    z_f, z_c, dz_f, dz_c = generate_grid(Lz, nz, gamma)

    device = torch.device('cpu')  # Test on CPU for reproducibility
    dtype = torch.float64

    # Convert grid to torch tensors
    dz_f = torch.tensor(dz_f, device=device, dtype=dtype)
    dz_c = torch.tensor(dz_c, device=device, dtype=dtype)

    # Create random velocity field
    torch.manual_seed(42)
    u = torch.randn(nx+1, ny+2, nz+2, device=device, dtype=dtype) * 0.1
    v = torch.randn(nx+2, ny+1, nz+2, device=device, dtype=dtype) * 0.1
    w = torch.randn(nx+2, ny+2, nz+1, device=device, dtype=dtype) * 0.1

    # Test 1: Compare with separate kernels (original implementation)
    print("\n1. Computing RHS using separate kernels (original)...")

    # Advection
    adv_u_orig = advection_u(u, v, w, nx, ny, nz, dx, dy, dz_f)
    adv_v_orig = advection_v(u, v, w, nx, ny, nz, dx, dy, dz_f)
    adv_w_orig = advection_w(u, v, w, nx, ny, nz, dx, dy, dz_c)

    # Diffusion
    diff_u_orig = diffusion_u(u, nx, ny, nz, dx, dy, dz_c, dz_f, nu)
    diff_v_orig = diffusion_v(v, nx, ny, nz, dx, dy, dz_c, dz_f, nu)
    diff_w_orig = diffusion_w(w, nx, ny, nz, dx, dy, dz_c, dz_f, nu)

    # Combined RHS: diffusion - advection
    rhs_u_orig = diff_u_orig - adv_u_orig
    rhs_v_orig = diff_v_orig - adv_v_orig
    rhs_w_orig = diff_w_orig - adv_w_orig

    print("2. Computing RHS using fused kernel v2...")
    rhs_u_fused, rhs_v_fused, rhs_w_fused = compute_momentum_rhs_fused_v2(
        u, v, w, nx, ny, nz, dx, dy, dz_c, dz_f, nu
    )

    # Compare results
    print("\n3. Comparing results...")

    # Compute differences
    diff_u = torch.abs(rhs_u_fused - rhs_u_orig)
    diff_v = torch.abs(rhs_v_fused - rhs_v_orig)
    diff_w = torch.abs(rhs_w_fused - rhs_w_orig)

    # Find interior points (where RHS is actually computed)
    u_interior = diff_u[1:nx, 1:ny+1, 1:nz+1]
    v_interior = diff_v[1:nx+1, 1:ny, 1:nz+1]
    w_interior = diff_w[1:nx+1, 1:ny+1, 1:nz]

    print(f"\nU-component:")
    print(f"  Max absolute difference: {u_interior.max().item():.2e}")
    print(f"  Mean absolute difference: {u_interior.mean().item():.2e}")
    print(f"  Max RHS value: {torch.abs(rhs_u_orig[1:nx, 1:ny+1, 1:nz+1]).max().item():.2e}")

    print(f"\nV-component:")
    print(f"  Max absolute difference: {v_interior.max().item():.2e}")
    print(f"  Mean absolute difference: {v_interior.mean().item():.2e}")
    print(f"  Max RHS value: {torch.abs(rhs_v_orig[1:nx+1, 1:ny, 1:nz+1]).max().item():.2e}")

    print(f"\nW-component:")
    print(f"  Max absolute difference: {w_interior.max().item():.2e}")
    print(f"  Mean absolute difference: {w_interior.mean().item():.2e}")
    print(f"  Max RHS value: {torch.abs(rhs_w_orig[1:nx+1, 1:ny+1, 1:nz]).max().item():.2e}")

    # Check if differences are within acceptable tolerance
    tol = 1e-12  # Very strict tolerance for double precision
    u_pass = u_interior.max().item() < tol
    v_pass = v_interior.max().item() < tol
    w_pass = w_interior.max().item() < tol

    print("\n" + "=" * 80)
    if u_pass and v_pass and w_pass:
        print("✓ PASSED: Fused kernel v2 matches original implementation!")
    else:
        print("✗ FAILED: Significant differences detected!")
    print("=" * 80)

    return u_pass and v_pass and w_pass


def test_fused_kernel_imex():
    """Test the fused IMEX kernel."""
    print("\n\n")
    print("=" * 80)
    print("Testing compute_momentum_rhs_fused_imex (IMEX scheme)")
    print("=" * 80)

    # Setup test grid
    nx, ny, nz = 32, 32, 32
    Lx, Ly, Lz = 2.67, 0.8, 2.0
    gamma = 2.5
    nu = 1.0 / 2870.0

    dx = Lx / nx
    dy = Ly / ny
    z_f, z_c, dz_f, dz_c = generate_grid(Lz, nz, gamma)

    device = torch.device('cpu')
    dtype = torch.float64

    dz_f = torch.tensor(dz_f, device=device, dtype=dtype)
    dz_c = torch.tensor(dz_c, device=device, dtype=dtype)

    # Create random velocity field
    torch.manual_seed(42)
    u = torch.randn(nx+1, ny+2, nz+2, device=device, dtype=dtype) * 0.1
    v = torch.randn(nx+2, ny+1, nz+2, device=device, dtype=dtype) * 0.1
    w = torch.randn(nx+2, ny+2, nz+1, device=device, dtype=dtype) * 0.1

    # Test: Compare with separate kernels
    print("\n1. Computing RHS using separate kernels (original IMEX)...")

    # Advection (full 3D)
    adv_u_orig = advection_u(u, v, w, nx, ny, nz, dx, dy, dz_f)
    adv_v_orig = advection_v(u, v, w, nx, ny, nz, dx, dy, dz_f)
    adv_w_orig = advection_w(u, v, w, nx, ny, nz, dx, dy, dz_c)

    # Diffusion (XY only, no Z)
    diff_xy_u_orig = diffusion_xy_u(u, nx, ny, nz, dx, dy, nu)
    diff_xy_v_orig = diffusion_xy_v(v, nx, ny, nz, dx, dy, nu)
    diff_xy_w_orig = diffusion_xy_w(w, nx, ny, nz, dx, dy, nu)

    # Combined RHS: diffusion_xy - advection
    rhs_u_orig = diff_xy_u_orig - adv_u_orig
    rhs_v_orig = diff_xy_v_orig - adv_v_orig
    rhs_w_orig = diff_xy_w_orig - adv_w_orig

    print("2. Computing RHS using fused IMEX kernel...")
    rhs_u_fused, rhs_v_fused, rhs_w_fused = compute_momentum_rhs_fused_imex(
        u, v, w, nx, ny, nz, dx, dy, dz_c, dz_f, nu
    )

    # Compare results
    print("\n3. Comparing results...")

    diff_u = torch.abs(rhs_u_fused - rhs_u_orig)
    diff_v = torch.abs(rhs_v_fused - rhs_v_orig)
    diff_w = torch.abs(rhs_w_fused - rhs_w_orig)

    # Interior points (corrected for IMEX scheme)
    u_interior = diff_u[1:nx, 1:ny+1, 1:nz+1]  # U is at [1:nx, 1:ny+1, 1:nz+1]
    v_interior = diff_v[1:nx+1, 1:ny, 1:nz+1]  # V is at [1:nx+1, 1:ny, 1:nz+1]
    w_interior = diff_w[1:nx+1, 1:ny+1, 1:nz]

    print(f"\nU-component:")
    print(f"  Max absolute difference: {u_interior.max().item():.2e}")
    print(f"  Mean absolute difference: {u_interior.mean().item():.2e}")
    print(f"  Max RHS value: {torch.abs(rhs_u_orig[1:nx, 1:ny+1, 1:nz+1]).max().item():.2e}")

    print(f"\nV-component:")
    print(f"  Max absolute difference: {v_interior.max().item():.2e}")
    print(f"  Mean absolute difference: {v_interior.mean().item():.2e}")
    print(f"  Max RHS value: {torch.abs(rhs_v_orig[1:nx+1, 1:ny, 1:nz+1]).max().item():.2e}")

    print(f"\nW-component:")
    print(f"  Max absolute difference: {w_interior.max().item():.2e}")
    print(f"  Mean absolute difference: {w_interior.mean().item():.2e}")
    print(f"  Max RHS value: {torch.abs(rhs_w_orig[1:nx+1, 1:ny+1, 1:nz]).max().item():.2e}")

    # Check tolerance
    tol = 1e-12
    u_pass = u_interior.max().item() < tol
    v_pass = v_interior.max().item() < tol
    w_pass = w_interior.max().item() < tol

    print("\n" + "=" * 80)
    if u_pass and v_pass and w_pass:
        print("✓ PASSED: Fused IMEX kernel matches original implementation!")
    else:
        print("✗ FAILED: Significant differences detected!")
    print("=" * 80)

    return u_pass and v_pass and w_pass


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TESTING ENHANCED FUSED GPU KERNELS")
    print("=" * 80)

    # Run tests
    test1_pass = test_fused_kernel_v2()
    test2_pass = test_fused_kernel_imex()

    # Summary
    print("\n\n")
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Fused kernel v2 (AB2):   {'✓ PASSED' if test1_pass else '✗ FAILED'}")
    print(f"Fused IMEX kernel:       {'✓ PASSED' if test2_pass else '✗ FAILED'}")
    print("=" * 80)

    if test1_pass and test2_pass:
        print("\n✓ All tests passed! Kernels are numerically correct.")
    else:
        print("\n✗ Some tests failed. Please review the implementation.")
