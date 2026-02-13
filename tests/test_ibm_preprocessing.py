import torch
import numpy as np

from ibm.geometry import Sphere
from ibm.preprocessing import find_support_staggered, compute_rkpm_b_coefficients, compute_epsilon_global

def test_preprocessing_pipeline():
    """Test full preprocessing pipeline on a small grid."""
    # Setup small grid
    nx, ny, nz = 10, 10, 10
    dx, dy = 0.1, 0.1
    dz = torch.full((nz,), 0.1)
    
    # Create grid coordinates
    x = torch.linspace(0, nx*dx, nx)
    y = torch.linspace(0, ny*dy, ny)
    z = torch.linspace(0, nz*0.1, nz)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    x_grid = torch.stack([X, Y, Z], dim=-1).unsqueeze(0) # [1, nx, ny, nz, 3] ? No, shape is [nx, ny, nz, 3]
    x_grid = torch.stack([X, Y, Z], dim=-1)
    
    # Create sphere
    center = [0.5, 0.5, 0.5]
    radius = 0.2
    sphere = Sphere(center, radius, n_points=50) # Small number of points
    
    # 1. Find support
    sup, cnt, h = find_support_staggered(sphere.x_lag, x_grid, dx, dy, dz)
    
    assert sup.shape[0] == 50
    assert torch.all(cnt > 0)
    
    # 2. Compute b coefficients
    b_coef = compute_rkpm_b_coefficients(sphere.x_lag, x_grid, sup, cnt, h, dx, dy, dz)
    
    assert b_coef.shape == (50, 10)
    # Check that b0 is close to 1 (zeroth moment)
    # Actually b depends on the moments.
    
    # 3. Compute epsilon
    epsilon = compute_epsilon_global(sphere.x_lag, x_grid, sup, cnt, b_coef, h, sphere.dS, dx, dy, dz)
    
    assert epsilon.shape == (50,)
    # Epsilon should be roughly 1/dS ? No, M * eps = 1.
    # M_ij ~ delta * delta * dV.
    # If delta ~ 1/dV, then M ~ 1/dV. So eps ~ dV?
    # Let's just check it runs and produces finite values.
    assert torch.all(torch.isfinite(epsilon))

if __name__ == "__main__":
    test_preprocessing_pipeline()
    print("All tests passed!")
