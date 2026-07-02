"""
Unit test of RigidCanopyIBM.apply_forcing:
- in-place velocity update opposing the local flow (uniform u=1 stream)
- drag sign and bookkeeping: returned force = actual grid momentum change / dt
- v, w untouched when their marker velocities are zero
- slip at markers reduced by the expected factor per shot
- n_iter multidirect iterations reduce slip further
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import torch
from utils import generate_double_stretched_grid
from canopy import RigidCanopyIBM

torch.set_default_dtype(torch.float64)

nx, ny = 96, 72
Lx, Ly, Lz = 2 * math.pi, 1.5 * math.pi, 1.0
h = 0.25
dx, dy = Lx / nx, Ly / ny
z_f, z_c, dz_f, dz_c = generate_double_stretched_grid(24, 40, h, Lz, 2.0, 'auto')
nz = len(dz_f)

cfg = {'h': h, 'n_fil_x': 12, 'n_fil_y': 9, 'placement': 'random_in_tile',
       'seed': 42, 'diameter': 2.2 * dx, 'markers_per_ring': 4}
ibm = RigidCanopyIBM(cfg, nx, ny, nz, dx, dy, Lx, Ly, z_c, z_f, dz_f, dz_c, 'cpu')

failures = []
def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name} {detail}")
    if not cond:
        failures.append(name)

u = torch.ones(nx + 1, ny + 2, nz + 2)
v = torch.zeros(nx + 2, ny + 1, nz + 2)
w = torch.zeros(nx + 2, ny + 2, nz + 1)

dt = 1.5e-3
dt_t = torch.tensor(dt)
gain_t = torch.tensor(ibm.recommended_alpha)

slip0 = ibm.slip_rms(u, v, w).item()
u_before = u.clone()

# grid momentum bookkeeping (u cells: volume dx*dy*dz_f)
vol_u = (dx * dy) * torch.cat([torch.zeros(1), dz_f, torch.zeros(1)]).view(1, 1, -1)
mom_before = (u * vol_u).sum().item()

drag = ibm.apply_forcing(u, v, w, dt_t, gain_t)
mom_after = (u * vol_u).sum().item()
slip1 = ibm.slip_rms(u, v, w).item()

print("\n1. Direct forcing on a uniform stream")
check("u reduced in the canopy", bool((u <= u_before + 1e-14).all()) and slip1 < slip0,
      f"(slip {slip0:.3f} -> {slip1:.3f})")
check("v, w untouched (zero marker velocity)",
      v.abs().max().item() == 0.0 and w.abs().max().item() == 0.0)
check("streamwise force is a drag (negative)", drag[0].item() < 0,
      f"(Fx = {drag[0].item():.4f})")
check("spanwise/vertical force ~ 0 vs drag",
      abs(drag[1].item()) < 0.02 * abs(drag[0].item()) and
      abs(drag[2].item()) < 0.02 * abs(drag[0].item()),
      f"(Fy = {drag[1].item():.2e}, Fz = {drag[2].item():.2e})")

print("\n2. Momentum bookkeeping")
dmom = mom_after - mom_before
err = abs(dmom - drag[0].item() * dt) / abs(dmom)
check("returned Fx = grid momentum change / dt", err < 1e-12, f"(rel err {err:.2e})")

print("\n3. Expected per-shot slip reduction")
# One shot with gain alpha on a uniform stream: marker velocity after forcing
# is (1 - alpha * A 1); rms predicted from the coupling operator directly
pred = (1.0 - gain_t * ibm._apply_coupling(torch.ones(ibm.N_L), 'u'))
pred_rms = pred.square().mean().sqrt().item()
# slip_rms includes v, w (still 0); compare u-only slip
slip1_u = ibm.interpolate(u, 'u').square().mean().sqrt().item()
check("slip matches operator prediction", abs(slip1_u - pred_rms) / pred_rms < 1e-10,
      f"({slip1_u:.6f} vs {pred_rms:.6f})")

print("\n4. Multidirect iterations (n_iter=3)")
cfg3 = dict(cfg, forcing={'n_iter': 3})
ibm3 = RigidCanopyIBM(cfg3, nx, ny, nz, dx, dy, Lx, Ly, z_c, z_f, dz_f, dz_c, 'cpu')
u3 = torch.ones(nx + 1, ny + 2, nz + 2)
v3 = torch.zeros(nx + 2, ny + 1, nz + 2)
w3 = torch.zeros(nx + 2, ny + 2, nz + 1)
drag3 = ibm3.apply_forcing(u3, v3, w3, dt_t, gain_t)
slip3 = ibm3.slip_rms(u3, v3, w3).item()
check("3 iterations beat 1", slip3 < 0.6 * slip1, f"({slip3:.4f} vs {slip1:.4f})")
mom3 = ((u3 - 1.0) * vol_u).sum().item()
err3 = abs(mom3 - drag3[0].item() * dt) / abs(mom3)
check("bookkeeping holds across iterations", err3 < 1e-12, f"(rel err {err3:.2e})")

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("All forcing checks passed.")
