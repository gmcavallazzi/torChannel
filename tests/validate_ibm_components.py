"""
Validate IBM components against MATLAB reference implementation.
Tests each function individually with known inputs and expected outputs.
"""

import torch
import numpy as np
import yaml

def test_roma_kernel():
    """Test Roma kernel against MATLAB phi_roma.m"""
    from ibm.rkpm_kernel import roma_kernel_1d
    
    print("="*60)
    print("TEST 1: Roma Kernel (phi_roma.m)")
    print("="*60)
    
    # Test points
    r_test = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    h = 1.0  # Kernel width
    phi = roma_kernel_1d(r_test, h)
    
    print(f"r = {r_test.numpy()}")
    print(f"h = {h}")
    print(f"phi(r) = {phi.numpy()}")
    print(f"Expected: phi(0)=1.0, phi(2)=0.0, sum should integrate to 1")
    print(f"Sum of phi for r in [-2,2] with dr=0.01: {torch.sum(roma_kernel_1d(torch.linspace(-2, 2, 401), h)) * 0.01:.6f}")
    print()

def test_sphere_geometry():
    """Test sphere geometry against MATLAB"""
    from ibm.geometry import Sphere
    
    print("="*60)
    print("TEST 2: Sphere Geometry")
    print("="*60)
    
    # Create sphere
    dx = 0.0625  # Grid spacing for auto-resolution
    sphere = Sphere(center=[0.5, 0.5, 0.5], radius=0.1, dx=dx, device='cpu')
    
    print(f"Number of points: {sphere.n_points}")
    print(f"Radius: {sphere.radius}")
    print(f"Expected surface area: {4 * np.pi * sphere.radius**2:.6f}")
    print(f"Computed surface area: {torch.sum(sphere.dS).item():.6f}")
    print(f"Relative error: {abs(torch.sum(sphere.dS).item() - 4*np.pi*sphere.radius**2) / (4*np.pi*sphere.radius**2) * 100:.2f}%")
    
    # Check a few points are on sphere
    distances = torch.sqrt(torch.sum((sphere.x_lag - sphere.center.view(1, 3))**2, dim=1))
    print(f"Distance from center: min={torch.min(distances):.6f}, max={torch.max(distances):.6f}, mean={torch.mean(distances):.6f}")
    print(f"All points on sphere? {torch.allclose(distances, torch.tensor(sphere.radius), rtol=1e-3)}")
    print()

def test_rkpm_preprocessing():
    """Test RKPM preprocessing with simple grid"""
    from ibm.geometry import Sphere
    from ibm.preprocessing import preprocess_ibm_rkpm
    
    print("="*60)
    print("TEST 3: RKPM Preprocessing")
    print("="*60)
    
    # Simple uniform grid
    nx, ny, nz = 16, 16, 16
    Lx, Ly, Lz = 1.0, 1.0, 1.0
    dx, dy = Lx/nx, Ly/ny
    
    # Uniform z-grid
    z_f = torch.linspace(0, Lz, nz+1)
    z_c = 0.5 * (z_f[:-1] + z_f[1:])
    dz_f = torch.diff(z_f)
    dz_c = torch.diff(z_c)
    dz_c = torch.cat([dz_c, dz_c[-1:]])  # Pad to match nz
    
    # Create sphere at center
    sphere = Sphere(center=[0.5, 0.5, 0.5], radius=0.1, dx=dx, device='cpu')
    
    # Build simple u-grid (just test u-component)
    x = torch.linspace(0, Lx, nx+1)
    y = (torch.arange(-1, ny+1) + 0.5) * dy
    z_inner = z_c
    z = torch.cat([z_inner[0:1]-dz_c[0], z_inner, z_inner[-1:]+dz_c[-1]])
    
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    u_grid = torch.stack([X, Y, Z], dim=-1)
    
    print(f"Grid: {nx}x{ny}x{nz}")
    print(f"dx={dx:.4f}, dy={dy:.4f}, dz_min={torch.min(dz_f):.4f}")
    print(f"Sphere: center={sphere.center.numpy()}, R={sphere.radius}")
    print(f"Sphere points: {sphere.n_points}")
    
    # Preprocess (just u-component for now)
    from ibm.preprocessing import find_support_staggered, compute_rkpm_b_coefficients, compute_epsilon_global
    
    # Find support
    support_u, support_count_u, h_u = find_support_staggered(
        sphere.x_lag, u_grid, dx, dy, dz_f
    )
    
    print(f"\nSupport counts: min={torch.min(support_count_u)}, max={torch.max(support_count_u)}, mean={torch.mean(support_count_u.float()):.1f}")
    
    # Compute b coefficients
    b_coef_u = compute_rkpm_b_coefficients(
        sphere.x_lag, u_grid, support_u, support_count_u, h_u, dx, dy, dz_f
    )
    
    print(f"b_coef shape: {b_coef_u.shape}")
    print(f"b_coef stats: min={torch.min(b_coef_u):.4e}, max={torch.max(b_coef_u):.4e}")
    
    # DEBUG: Check grid shape
    print(f"\nu_grid shape: {u_grid.shape}")
    print(f"Expected: [nx+1, ny+2, nz+2, 3] = [{nx+1}, {ny+2}, {nz+2}, 3]")
    
    # Compute epsilon - CORRECT argument order:
    # compute_epsilon_global(x_lag, x_grid, support_indices, support_count, b_coef, h, dS, dx, dy, dz)
    epsilon_u = compute_epsilon_global(
        sphere.x_lag, u_grid, support_u, support_count_u,
        b_coef_u, h_u, sphere.dS, dx, dy, dz_f
    )
    
    print(f"\nEpsilon stats:")
    print(f"  min={torch.min(epsilon_u):.4e}")
    print(f"  max={torch.max(epsilon_u):.4e}")
    print(f"  mean={torch.mean(epsilon_u):.4e}")
    print(f"  Any NaN? {torch.isnan(epsilon_u).any()}")
    print(f"  Any Inf? {torch.isinf(epsilon_u).any()}")
    
    # Check if epsilon values are reasonable (should be O(1))
    if torch.max(torch.abs(epsilon_u)) > 1e10:
        print(f"  WARNING: Epsilon values are HUGE! This will cause instability.")
    elif torch.max(torch.abs(epsilon_u)) < 1e-10:
        print(f"  WARNING: Epsilon values are TINY! This may indicate a problem.")
    else:
        print(f"  Epsilon values appear reasonable (O(1))")
    
    print()

def test_interpolation_spreading():
    """Test interpolation and spreading operators"""
    from ibm.geometry import Sphere
    from ibm.preprocessing import preprocess_ibm_rkpm
    from ibm.operators import interpolate_to_lagrangian, spread_to_eulerian
    
    print("="*60)
    print("TEST 4: Interpolation and Spreading")
    print("="*60)
    
    # Simple uniform grid
    nx, ny, nz = 16, 16, 16
    Lx, Ly, Lz = 1.0, 1.0, 1.0
    dx, dy = Lx/nx, Ly/ny
    
    z_f = torch.linspace(0, Lz, nz+1)
    z_c = 0.5 * (z_f[:-1] + z_f[1:])
    dz_f = torch.diff(z_f)
    dz_c = torch.diff(z_c)
    dz_c = torch.cat([dz_c, dz_c[-1:]])
    
    sphere = Sphere(center=[0.5, 0.5, 0.5], radius=0.1, dx=dx, device='cpu')
    
    # Build u-grid
    x = torch.linspace(0, Lx, nx+1)
    y = (torch.arange(-1, ny+1) + 0.5) * dy
    z_inner = z_c
    z = torch.cat([z_inner[0:1]-dz_c[0], z_inner, z_inner[-1:]+dz_c[-1]])
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    u_grid = torch.stack([X, Y, Z], dim=-1)
    
    # Dummy grids for v, w (not used in this test)
    v_grid = u_grid.clone()
    w_grid = u_grid.clone()
    
    # Preprocess
    ibm_data = preprocess_ibm_rkpm(sphere, u_grid, v_grid, w_grid, dx, dy, dz_f, dz_c, 'cpu')
    
    # Test with constant field
    u_field = torch.ones(nx+1, ny+2, nz+2)
    
    u_lag = interpolate_to_lagrangian(
        u_field, sphere.x_lag, ibm_data['u_grid'],
        ibm_data['support_u'], ibm_data['support_count_u'],
        ibm_data['b_coef_u'], ibm_data['h_u'], dx, dy, dz_f
    )
    
    print(f"Constant field test:")
    print(f"  Input: u_field = 1.0 everywhere")
    print(f"  Interpolated to Lagrangian: min={torch.min(u_lag):.6f}, max={torch.max(u_lag):.6f}, mean={torch.mean(u_lag):.6f}")
    print(f"  Expected: ~1.0 everywhere (partition of unity)")
    print(f"  Error: {torch.max(torch.abs(u_lag - 1.0)):.6e}")
    
    # Spread back
    u_spread = spread_to_eulerian(
        u_lag, ibm_data['epsilon_u'], sphere.dS, sphere.x_lag,
        ibm_data['u_grid'], ibm_data['support_u'],
        ibm_data['support_count_u'], ibm_data['b_coef_u'],
        ibm_data['h_u'], nx+1, ny+2, nz+2, dx, dy, dz_f
    )
    
    print(f"\nSpread back to Eulerian:")
    print(f"  min={torch.min(u_spread):.6e}, max={torch.max(u_spread):.6e}")
    print(f"  Non-zero values: {torch.sum(torch.abs(u_spread) > 1e-10).item()} / {u_spread.numel()}")
    
    print()

if __name__ == "__main__":
    print("\n" + "="*60)
    print("IBM COMPONENT VALIDATION")
    print("="*60 + "\n")
    
    test_roma_kernel()
    test_sphere_geometry()
    test_rkpm_preprocessing()
    test_interpolation_spreading()
    
    print("="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
