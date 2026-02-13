import torch
import numpy as np

from ibm.rkpm_kernel import roma_kernel_1d, rkpm_delta_3d

def test_roma_kernel_partition_of_unity():
    """Verify ∑ φ(x_i) = 1 for uniform grid."""
    h = 1.0
    # Create a grid that covers the support [-2h, 2h] well
    # For partition of unity, we need sum_i phi(x - x_i) = 1
    # or equivalently sum_i phi(x_i - x) = 1/h * dx ?
    # Actually, the property is sum_i phi(r_i) * h = 1 for r_i on grid
    
    # Let's test integral property first: integral phi(r) dr = 1
    x = torch.linspace(-3*h, 3*h, 10000)
    dx = x[1] - x[0]
    phi = roma_kernel_1d(x, h)
    integral = torch.sum(phi) * dx
    assert torch.abs(integral - 1.0) < 1e-4

    # Test partition of unity on a grid
    # If we have a grid with spacing h, sum of phi(x_grid - x_p) should be 1/h
    # Wait, the formula has 1/(8h) factor.
    # The standard property is sum phi(x_i - x) = 1/h.
    # Let's check specific values.
    # If r=0, phi = (3 + 1)/8h = 0.5/h
    # If r=h, phi = (3-2 + 1)/8h = 0.25/h (using first formula)
    # If r=h, phi = (5-2 - sqrt(-7+12-4))/8h = (3-1)/8h = 0.25/h (using second formula)
    # So phi(0) + phi(h) + phi(-h) = 0.5/h + 0.25/h + 0.25/h = 1/h.
    # Yes, for grid aligned with particle, sum is 1/h.
    
    # Let's test random offset
    offset = 0.3 * h
    # Grid points at k*h
    # r_k = k*h - offset
    # We sum k=-2, -1, 0, 1, 2 (since support is 2h)
    k = torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float32)
    r = k * h - offset
    phi = roma_kernel_1d(r, h)
    total = torch.sum(phi)
    assert torch.abs(total - 1.0/h) < 1e-6

def test_roma_kernel_compact_support():
    """Verify φ(r) = 0 for |r| > 2h."""
    h = 1.0
    assert roma_kernel_1d(torch.tensor(2.001*h), h) == 0.0
    assert roma_kernel_1d(torch.tensor(-2.001*h), h) == 0.0

def test_rkpm_delta_3d_shape():
    """Verify output shape of 3D delta."""
    ne = 10
    x_eul = torch.randn(ne, 3)
    x_lag = torch.randn(3)
    h = torch.tensor([0.1, 0.1, 0.1])
    b_coef = torch.zeros(10)
    b_coef[0] = 1.0 # Constant term
    
    delta = rkpm_delta_3d(x_eul, x_lag, h, b_coef)
    assert delta.shape == (ne,)

if __name__ == "__main__":
    test_roma_kernel_partition_of_unity()
    test_roma_kernel_compact_support()
    test_rkpm_delta_3d_shape()
    print("All tests passed!")
