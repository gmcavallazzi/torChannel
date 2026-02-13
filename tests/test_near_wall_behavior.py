import torch
import sys
sys.path.append('/Users/giorgio.cavallazzi/Library/CloudStorage/OneDrive-City,UniversityofLondon/python_DNS_playground/DNS_homemade')

from solver import ChannelFlow
from utils import compute_bulk_velocity, compute_u_tau, compute_divergence
from operators import diffusion_u

torch.set_default_dtype(torch.float64)

print("="*90)
print("DIAGNOSTIC TEST: Near-Wall Momentum Budget")
print("="*90)

# Create solver
print("\nInitializing simulation...")
solver = ChannelFlow(config_file='config.yaml')

# Analyze from the start
print("\nAnalyzing initial near-wall momentum balance...")
print("="*90)

# Compute diffusion at initial state
diff_u = diffusion_u(solver.u, solver.nx, solver.ny, solver.nz,
                     solver.dx, solver.dy, solver.dz_c, solver.dz_f, solver.nu)

# Compute initial bulk velocity and forcing
u_bulk = compute_bulk_velocity(solver.u, solver.cell_vol_ratio, solver.total_volume)
forcing = (solver.U_bulk - u_bulk) / solver.dt

print(f"Initial u_bulk: {u_bulk:.12f}")
print(f"Target U_bulk: {solver.U_bulk}")
print(f"Initial forcing: {forcing:.6e}")
print(f"dt * forcing (velocity increment per step): {(solver.dt * forcing).item():.6e}")

# Look at first few z-layers near bottom wall
print(f"\nNear bottom wall (first 5 z-layers):")
print(f"  k=0: ghost cell below wall")
print(f"  k=1: first interior (wall-adjacent)")
print(f"  k=2,3,4: next layers")
print(f"\nMomentum budget at each layer (averaged over x,y):")
print(f"{'k':>3} {'u':>12} {'diff_u':>15} {'forcing':>15} {'diff_u/forcing':>15}")
print("-"*65)

for k in range(6):
    if k == 0:
        u_avg = torch.mean(solver.u[1:solver.nx+1, 1:solver.ny+1, k])
        diff_avg = 0.0  # Ghost cell, no diffusion computed here
        ratio = 0.0
    else:
        u_avg = torch.mean(solver.u[1:solver.nx+1, 1:solver.ny+1, k])
        diff_avg = torch.mean(diff_u[:, :, k-1])  # diff_u is interior only, k-1 indexing
        ratio = (diff_avg / forcing).item() if abs(forcing) > 1e-12 else 0.0

    print(f"{k:3d} {u_avg.item():12.6e} {diff_avg if isinstance(diff_avg, float) else diff_avg.item():15.6e} {forcing:15.6e} {ratio:15.3f}")

print("-"*65)

print(f"\nInterpretation:")
print(f"  If |diff_u/forcing| >> 1 near wall:")
print(f"    → Diffusion is removing momentum faster than forcing adds it")
print(f"    → Net effect at wall is NEGATIVE (momentum loss)")
print(f"    → Forcing must increase to compensate for this 'leak'")
print(f"  ")
print(f"  Expected for equilibrium channel flow:")
print(f"    → forcing balances wall shear stress globally")
print(f"    → But near wall, diffusion >> forcing (steep gradients)")
print(f"    → In interior, forcing ≈ diffusion")

# Check total momentum budget (extract interior only)
print(f"\ndiff_u shape: {diff_u.shape}")
print(f"cell_vol_ratio shape: {solver.cell_vol_ratio.shape}")

# diff_u should be interior only (nx, ny, nz), but let's check
if diff_u.shape == solver.cell_vol_ratio.shape:
    total_diffusion = torch.sum(diff_u * solver.cell_vol_ratio)
    total_forcing_effect = forcing * solver.total_volume

    print(f"\nGlobal momentum budget:")
    print(f"  Total diffusion: {total_diffusion.item():.6e}")
    print(f"  Total forcing: {total_forcing_effect:.6e}")
    print(f"  Ratio: {(total_diffusion / total_forcing_effect).item():.3f}")
else:
    print(f"\nShape mismatch - skipping global budget calculation")

print("="*90)
