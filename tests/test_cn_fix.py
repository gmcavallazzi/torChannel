"""
Test CN diffusion fix - check if bulk velocity is preserved
"""
import torch
import numpy as np
from operators import solve_implicit_diffusion_u, solve_implicit_diffusion_v, solve_implicit_diffusion_w
from utils import compute_bulk_velocity

# Setup small test case
nx, ny, nz = 32, 32, 16
device = 'cpu'

# Uniform grid for simplicity
dz_f = torch.ones(nz, device=device) * (1.0 / nz)
dz_c = torch.ones(nz+1, device=device) * (1.0 / nz)

# Create velocity field with non-zero bulk velocity
u = torch.randn(nx+2, ny+2, nz+2, device=device) * 0.1 + 1.0  # Mean ~1.0
v = torch.randn(nx+2, ny+2, nz+2, device=device) * 0.1 + 0.5  # Mean ~0.5
w = torch.randn(nx+1, ny+1, nz+1, device=device) * 0.01

# Apply BCs
u[:, :, 0] = -u[:, :, 1]
u[:, :, -1] = -u[:, :, -2]
v[:, :, 0] = -v[:, :, 1]
v[:, :, -1] = -v[:, :, -2]
w[:, :, 0] = 0
w[:, :, -1] = 0

# Compute initial bulk velocities
u_bulk_init = u[1:nx+1, 1:ny+1, 1:nz+1].mean().item()
v_bulk_init = v[1:nx+1, 1:ny+1, 1:nz+1].mean().item()
w_bulk_init = w[1:nx, 1:ny, 1:nz].mean().item()

print(f"Initial bulk velocities:")
print(f"  u_bulk = {u_bulk_init:.8f}")
print(f"  v_bulk = {v_bulk_init:.8f}")
print(f"  w_bulk = {w_bulk_init:.8f}")

# Apply implicit diffusion (CN with theta=0.5)
dt = 0.001
nu = 0.01

u_new = solve_implicit_diffusion_u(u, dt, nx, ny, nz, dz_c, dz_f, nu, theta=0.5)
v_new = solve_implicit_diffusion_v(v, dt, nx, ny, nz, dz_c, dz_f, nu, theta=0.5)
w_new = solve_implicit_diffusion_w(w, dt, nx, ny, nz, dz_c, dz_f, nu, theta=0.5)

# Compute final bulk velocities
u_bulk_final = u_new[1:nx+1, 1:ny+1, 1:nz+1].mean().item()
v_bulk_final = v_new[1:nx+1, 1:ny+1, 1:nz+1].mean().item()
w_bulk_final = w_new[1:nx, 1:ny, 1:nz].mean().item()

print(f"\nFinal bulk velocities:")
print(f"  u_bulk = {u_bulk_final:.8f}")
print(f"  v_bulk = {v_bulk_final:.8f}")
print(f"  w_bulk = {w_bulk_final:.8f}")

# Check changes
u_bulk_change = abs(u_bulk_final - u_bulk_init)
v_bulk_change = abs(v_bulk_final - v_bulk_init)
w_bulk_change = abs(w_bulk_final - w_bulk_init)

print(f"\nBulk velocity changes:")
print(f"  |Δu_bulk| = {u_bulk_change:.8e}")
print(f"  |Δv_bulk| = {v_bulk_change:.8e}")
print(f"  |Δw_bulk| = {w_bulk_change:.8e}")

# Check for NaNs
has_nan = torch.isnan(u_new).any() or torch.isnan(v_new).any() or torch.isnan(w_new).any()
print(f"\nNaN detected: {has_nan}")

# Diffusion should preserve mean velocity (bulk velocity should not change significantly)
# Small changes O(1e-16) are due to floating point errors
tolerance = 1e-10
if u_bulk_change < tolerance and v_bulk_change < tolerance and w_bulk_change < tolerance and not has_nan:
    print("\n✅ TEST PASSED: Bulk velocity preserved!")
else:
    print("\n❌ TEST FAILED: Bulk velocity not preserved!")
    if u_bulk_change >= tolerance:
        print(f"   u_bulk changed by {u_bulk_change:.8e} (exceeds tolerance {tolerance})")
    if v_bulk_change >= tolerance:
        print(f"   v_bulk changed by {v_bulk_change:.8e} (exceeds tolerance {tolerance})")
    if w_bulk_change >= tolerance:
        print(f"   w_bulk changed by {w_bulk_change:.8e} (exceeds tolerance {tolerance})")
