import torch
import numpy as np

from ibm.geometry import Sphere
from ibm.preprocessing import find_support_staggered, compute_rkpm_b_coefficients, compute_epsilon_global
from ibm.operators import interpolate_to_lagrangian, spread_to_eulerian

def test_operators_consistency():
    """Test interpolate -> spread consistency for constant field."""
    # Setup small grid
    nx, ny, nz = 10, 10, 10
    dx, dy = 0.1, 0.1
    dz = torch.full((nz,), 0.1)
    
    x = torch.linspace(0, nx*dx, nx)
    y = torch.linspace(0, ny*dy, ny)
    z = torch.linspace(0, nz*0.1, nz)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    x_grid = torch.stack([X, Y, Z], dim=-1)
    
    # Create sphere
    center = [0.5, 0.5, 0.5]
    radius = 0.2
    sphere = Sphere(center, radius, n_points=50)
    
    # Preprocess
    sup, cnt, h = find_support_staggered(sphere.x_lag, x_grid, dx, dy, dz)
    b_coef = compute_rkpm_b_coefficients(sphere.x_lag, x_grid, sup, cnt, h, dx, dy, dz)
    epsilon = compute_epsilon_global(sphere.x_lag, x_grid, sup, cnt, b_coef, h, sphere.dS, dx, dy, dz)
    
    # Test 1: Interpolate constant field f=1
    f_eul = torch.ones((nx, ny, nz))
    f_lag = interpolate_to_lagrangian(f_eul, sphere.x_lag, x_grid, sup, cnt, b_coef, h, dx, dy, dz)
    
    # Should be close to 1 (partition of unity)
    # Note: RKPM delta ensures polynomial reproduction.
    # sum delta * dV should be 1.
    assert torch.allclose(f_lag, torch.ones_like(f_lag), atol=1e-2)
    
    # Test 2: Spread constant field f_lag=1
    # This is less obvious what it should be.
    # But if we spread f_lag=1, we get a field concentrated on the surface.
    
    # Test 3: Consistency check
    # <Spread(f_lag), g_eul> = <f_lag, Interpolate(g_eul)> ?
    # This is the adjoint property.
    # Spread: f_eul_j = sum_i f_lag_i * eps_i * dS_i * delta_ji
    # Interpolate: g_lag_i = sum_j g_eul_j * delta_ji * dV_j
    # Inner product Eul: sum_j f_eul_j * g_eul_j * dV_j
    # = sum_j (sum_i f_lag_i * eps_i * dS_i * delta_ji) * g_eul_j * dV_j
    # = sum_i f_lag_i * eps_i * dS_i * (sum_j g_eul_j * delta_ji * dV_j)
    # = sum_i f_lag_i * eps_i * dS_i * g_lag_i
    # = <f_lag, g_lag>_weighted
    # where weight is eps_i * dS_i.
    
    g_eul = torch.randn((nx, ny, nz))
    f_lag = torch.randn(sphere.n_points)
    
    f_spread = spread_to_eulerian(f_lag, epsilon, sphere.dS, sphere.x_lag, x_grid, sup, cnt, b_coef, h, nx, ny, nz, dx, dy, dz)
    g_interp = interpolate_to_lagrangian(g_eul, sphere.x_lag, x_grid, sup, cnt, b_coef, h, dx, dy, dz)
    
    # Compute inner products
    # Eul inner product: sum f * g * dV
    # Since dV is constant here (except dz, but dz is constant 0.1)
    dV = dx * dy * 0.1
    ip_eul = torch.sum(f_spread * g_eul) * dV
    
    # Lag inner product: sum f * g * eps * dS
    ip_lag = torch.sum(f_lag * g_interp * epsilon * sphere.dS)
    
    # Should be equal
    assert torch.abs(ip_eul - ip_lag) < 1e-4

if __name__ == "__main__":
    test_operators_consistency()
    print("All tests passed!")
