"""
Test CN diffusion fix - check energy decay and stability
"""
import torch
import numpy as np
from operators import solve_implicit_diffusion_u, solve_implicit_diffusion_v, solve_implicit_diffusion_w

# Setup small test case
nx, ny, nz = 32, 32, 16
device = 'cpu'

# Uniform grid for simplicity
dz_f = torch.ones(nz, device=device) * (1.0 / nz)
dz_c = torch.ones(nz+1, device=device) * (1.0 / nz)

# Create velocity field
u = torch.randn(nx+2, ny+2, nz+2, device=device) * 0.5
v = torch.randn(nx+2, ny+2, nz+2, device=device) * 0.5
w = torch.randn(nx+1, ny+1, nz+1, device=device) * 0.1

# Apply BCs
u[:, :, 0] = -u[:, :, 1]
u[:, :, -1] = -u[:, :, -2]
v[:, :, 0] = -v[:, :, 1]
v[:, :, -1] = -v[:, :, -2]
w[:, :, 0] = 0
w[:, :, -1] = 0

# Time-stepping parameters
dt = 0.001
nu = 0.01
n_steps = 10

print("Testing CN diffusion for energy decay and stability...\n")
print(f"Running {n_steps} steps with dt={dt}, nu={nu}")
print(f"Grid: {nx}x{ny}x{nz}")
print(f"theta=0.5 (Crank-Nicolson)\n")

# Track energy over time
energies = []

for step in range(n_steps):
    # Compute kinetic energy
    ke_u = (u[1:nx+1, 1:ny+1, 1:nz+1]**2).sum()
    ke_v = (v[1:nx+1, 1:ny+1, 1:nz+1]**2).sum()
    ke_w = (w[1:nx, 1:ny, 1:nz]**2).sum()
    ke_total = 0.5 * (ke_u + ke_v + ke_w).item()
    energies.append(ke_total)

    # Apply implicit diffusion (CN)
    u = solve_implicit_diffusion_u(u, dt, nx, ny, nz, dz_c, dz_f, nu, theta=0.5)
    v = solve_implicit_diffusion_v(v, dt, nx, ny, nz, dz_c, dz_f, nu, theta=0.5)
    w = solve_implicit_diffusion_w(w, dt, nx, ny, nz, dz_c, dz_f, nu, theta=0.5)

    # Re-apply BCs (mimicking solver loop)
    u[:, :, 0] = -u[:, :, 1]
    u[:, :, -1] = -u[:, :, -2]
    v[:, :, 0] = -v[:, :, 1]
    v[:, :, -1] = -v[:, :, -2]
    w[:, :, 0] = 0
    w[:, :, -1] = 0

    # Check for NaNs/Infs
    has_nan = torch.isnan(u).any() or torch.isnan(v).any() or torch.isnan(w).any()
    has_inf = torch.isinf(u).any() or torch.isinf(v).any() or torch.isinf(w).any()

    print(f"Step {step:2d}: KE = {ke_total:.6e}, NaN={has_nan}, Inf={has_inf}")

    if has_nan or has_inf:
        print("\n❌ INSTABILITY DETECTED!")
        break
else:
    print("\n✅ No NaNs or Infs detected!")

    # Check energy decay
    print("\nEnergy analysis:")
    energy_decrease = [(energies[i] - energies[i+1])/energies[i] * 100
                       for i in range(len(energies)-1)]

    all_decreasing = all(d > 0 for d in energy_decrease)

    print(f"  Initial energy: {energies[0]:.6e}")
    print(f"  Final energy:   {energies[-1]:.6e}")
    print(f"  Total decay:    {(energies[0]-energies[-1])/energies[0]*100:.2f}%")
    print(f"  Monotonic decay: {all_decreasing}")

    if all_decreasing:
        print("\n✅ TEST PASSED: Energy decays monotonically (physically correct)!")
    else:
        print("\n⚠️  WARNING: Energy increased in some steps (may indicate issue)")
        for i, d in enumerate(energy_decrease):
            if d < 0:
                print(f"     Step {i}->{i+1}: energy increased by {-d:.4f}%")
