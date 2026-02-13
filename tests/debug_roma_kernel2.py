"""
Debug partition of unity with multiple test points like the actual test
"""

import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ibm.rkpm_kernel import roma_kernel_1d

dx = 1.0
h = 1.5 * dx

# Test points (same as in test)
x_test = torch.linspace(-5, 5, 100)

# Grid points (same as in test)
x_grid = torch.arange(-10.0, 11.0, dx)

print(f"dx = {dx}")
print(f"h = {h} (h/dx = {h/dx})")
print(f"Number of test points: {len(x_test)}")
print(f"Number of grid points: {len(x_grid)}")
print(f"Support radius: 2h = {2*h}")
print()

max_error = 0.0
errors = []

# Find worst case
worst_x = None
worst_sum = None

for x in x_test:
    kernel_sum = 0.0
    for x_i in x_grid:
        r = x - x_i
        kernel_sum += roma_kernel_1d(r, h).item() * dx

    error = abs(kernel_sum - 1.0)
    errors.append(error)

    if error > max_error:
        max_error = error
        worst_x = x.item()
        worst_sum = kernel_sum

print(f"Max error: {max_error:.10e}")
print(f"Mean error: {sum(errors)/len(errors):.10e}")
print(f"Worst case at x = {worst_x:.6f}, sum = {worst_sum:.10f}")
print()

# Analyze worst case in detail
print(f"Analyzing worst case at x = {worst_x}")
print(f"{'x_i':>8} | {'r':>8} | {'r/h':>8} | {'δ(r)':>12} | {'δ(r)*dx':>12}")
print("-" * 60)

kernel_sum = 0.0
for x_i in x_grid:
    r_val = worst_x - x_i.item()
    if abs(r_val) <= 2*h + 0.1:  # Only print points near support
        delta_val = roma_kernel_1d(torch.tensor(r_val), h).item()
        delta_val_dx = delta_val * dx
        kernel_sum += delta_val_dx
        print(f"{x_i.item():8.2f} | {r_val:8.4f} | {r_val/h:8.4f} | {delta_val:12.8f} | {delta_val_dx:12.8f}")

print("-" * 60)
print(f"{'TOTAL':>8} |          |          | {kernel_sum/dx:12.8f} | {kernel_sum:12.8f}")
print(f"{'ERROR':>8} |          |          |              | {abs(kernel_sum - 1.0):12.8f}")
