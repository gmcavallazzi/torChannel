"""
Debug script to visualize Roma kernel and partition of unity
"""

import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ibm.rkpm_kernel import roma_kernel_1d

def test_different_h_dx_ratios():
    """Test partition of unity for different h/dx ratios."""

    print("Testing different h/dx ratios for partition of unity")
    print("="*70)

    dx = 1.0
    h_values = [0.5*dx, 0.75*dx, dx, 1.25*dx, 1.5*dx, 2.0*dx]

    # Single test point
    x_test = torch.tensor(0.5)

    # Grid points
    x_grid = torch.arange(-10.0, 11.0, dx)

    for h in h_values:
        kernel_sum = 0.0
        for x_i in x_grid:
            r = x_test - x_i
            kernel_sum += roma_kernel_1d(r, h).item() * dx

        error = abs(kernel_sum - 1.0)
        print(f"h = {h:.3f} (h/dx = {h/dx:.2f}): sum = {kernel_sum:.6f}, error = {error:.6e}")

    print("\n" + "="*70)
    print("Testing without dx multiplication (discrete sum)")
    print("="*70)

    for h in h_values:
        kernel_sum = 0.0
        for x_i in x_grid:
            r = x_test - x_i
            kernel_sum += roma_kernel_1d(r, h).item()  # No dx multiplication

        error = abs(kernel_sum - 1.0)
        print(f"h = {h:.3f} (h/dx = {h/dx:.2f}): sum = {kernel_sum:.6f}, error = {error:.6e}")


def visualize_kernel_values():
    """Print kernel values to understand the shape."""

    print("\n" + "="*70)
    print("Kernel values for h=dx=1.0")
    print("="*70)

    h = 1.0
    dx = 1.0

    # Evaluate at grid points
    r_values = torch.arange(-3.0, 3.1, dx)

    print(f"\n{'r':>6} | {'r/h':>6} | {'δ(r)':>12} | {'δ(r)*dx':>12}")
    print("-"*50)

    total_sum = 0.0
    total_sum_dx = 0.0

    for r in r_values:
        delta_val = roma_kernel_1d(r, h).item()
        delta_val_dx = delta_val * dx
        total_sum += delta_val
        total_sum_dx += delta_val_dx
        print(f"{r:6.2f} | {r/h:6.2f} | {delta_val:12.8f} | {delta_val_dx:12.8f}")

    print("-"*50)
    print(f"{'TOTAL':>6} |        | {total_sum:12.8f} | {total_sum_dx:12.8f}")
    print(f"{'ERROR':>6} |        | {abs(total_sum-1.0):12.8f} | {abs(total_sum_dx-1.0):12.8f}")


def check_kernel_integral():
    """Compute continuous integral to verify normalization."""

    print("\n" + "="*70)
    print("Continuous integral verification")
    print("="*70)

    h = 1.0

    # Fine discretization for integration
    r_values = torch.linspace(-2*h, 2*h, 10000)
    dr = r_values[1] - r_values[0]

    kernel_values = roma_kernel_1d(r_values, h)
    integral = torch.sum(kernel_values) * dr

    print(f"h = {h}")
    print(f"Integration range: [{-2*h:.2f}, {2*h:.2f}]")
    print(f"Number of points: {len(r_values)}")
    print(f"dr = {dr.item():.6f}")
    print(f"Integral = {integral.item():.10f}")
    print(f"Expected = 1.0")
    print(f"Error = {abs(integral.item() - 1.0):.10e}")


if __name__ == "__main__":
    test_different_h_dx_ratios()
    visualize_kernel_values()
    check_kernel_integral()
