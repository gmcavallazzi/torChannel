"""
Quick test to diagnose which optimization broke stability
"""
import torch
import numpy as np

# Test current version
print("Testing CURRENT optimized version...")
from projection_fft import initialize_fft_solver, solve_poisson_fft

nx, ny, nz = 64, 64, 32
device = 'cpu'
dx = dy = 2*np.pi / nx
dz_f = torch.ones(nz, device=device) * (1.0 / nz)
dz_c = torch.ones(nz+1, device=device) * (1.0 / nz)

# Initialize
fft_data = initialize_fft_solver(nx, ny, nz, dx, dy, dz_c, dz_f)
print(f"  Initialized. Has workspace_p: {'workspace_p' in fft_data}")

# Create test divergence
div = torch.randn(nx, ny, nz, device=device) * 0.1

# Solve 3 times (test for contamination)
for i in range(3):
    p = solve_poisson_fft(div.clone(), fft_data)
    has_nan = torch.isnan(p).any().item()
    has_inf = torch.isinf(p).any().item()
    p_max = p.abs().max().item()

    print(f"  Iteration {i+1}: max|p|={p_max:.6e}, NaN={has_nan}, Inf={has_inf}")

    if has_nan or has_inf:
        print("  ❌ INSTABILITY DETECTED!")
        break
else:
    print("  ✓ Stable")

# Check if zero mode is being handled
print(f"\n  Checking zero mode handling...")
print(f"  tri_b[0,0,0] = {fft_data['tri_b'][0,0,0].item():.6f} (should be 1.0)")
print(f"  tri_a[0,0,0] = {fft_data['tri_a'][0,0,0].item():.6f} (should be 0.0)")
