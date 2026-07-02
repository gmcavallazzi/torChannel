"""
Integration test: one rigid filament in a laminar open channel, full IMEX
stepping through ChannelFlow.

Regression guard against the instability of the removed IBM implementation:
- no NaN, kinetic energy bounded over the whole run
- marker slip settles to a small fraction of U_bulk
- post-projection divergence stays at the no-canopy level
- streamwise drag is negative (opposes the flow) and settles
- constant flow rate: bulk velocity held at U_bulk by the PI controller
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import tempfile
import yaml
import torch
from solver import ChannelFlow
from utils import compute_divergence, compute_bulk_velocity

torch.set_default_dtype(torch.float64)

workdir = tempfile.mkdtemp(prefix='ibm_single_')
config = {
    'grid': {'nx': 48, 'ny': 36, 'nz': 40, 'nz_canopy': 16, 'nz_outer': 24},
    'domain': {'Lx': 2 * math.pi, 'Ly': 1.5 * math.pi, 'Lz': 1.0,
               'stretching_type': 'double', 'z_transition': 0.25,
               'gamma_canopy': 1.5, 'gamma_outer': 'auto'},
    'flow': {'Re': 1000.0, 'Re_tau': 180.0, 'U_bulk': 1.0, 'gamma': 2.0},
    'boundary_conditions': {'top_wall': {'type': 'neumann'}},
    'time': {'dt': 2.0e-3, 'n_steps': 400, 't_max': 1000.0, 'CFL_target': 0.5,
             'dt_update_interval': 0, 'scheme': 'IMEX'},
    'initialization': {'type': 'parabolic', 'perturbation_intensity': 0.0},
    'solver': {'type': 'fft'},
    'compute': {'device': 'auto'},
    'output': {'results_folder': workdir, 'n_out': 100, 'n_save': 100000},
    'statistics': {'n_stats': 0},
    'canopy': {
        'enabled': True, 'h': 0.25, 'n_fil_x': 1, 'n_fil_y': 1,
        'placement': 'regular', 'seed': 1, 'markers_per_ring': 4,
        'forcing': {'alpha': 'auto', 'ramp_steps': 50, 'n_iter': 2},
    },
}
cfg_path = os.path.join(workdir, 'config.yaml')
with open(cfg_path, 'w') as f:
    yaml.safe_dump(config, f)

solver = ChannelFlow(config_file=cfg_path)

failures = []
def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name} {detail}")
    if not cond:
        failures.append(name)

n_steps = 400
ke_hist, drag_hist, slip_hist = [], [], []
for step in range(1, n_steps + 1):
    solver.current_step = solver.initial_step + step
    solver.step_imex(solver.dt)
    if step % 50 == 0:
        ke = 0.5 * (solver.u.square().mean() + solver.v.square().mean()
                    + solver.w.square().mean()).item()
        ke_hist.append(ke)
        drag_hist.append(solver.canopy_drag[0].item())
        slip_hist.append(solver.canopy.slip_rms(solver.u, solver.v, solver.w).item())

print(f"\n  KE:    " + " ".join(f"{k:.4f}" for k in ke_hist))
print(f"  dragX: " + " ".join(f"{d:8.4f}" for d in drag_hist))
print(f"  slip:  " + " ".join(f"{s:.4f}" for s in slip_hist))

print("\n1. Stability")
check("no NaN in fields", not bool(torch.isnan(solver.u).any() or
                                    torch.isnan(solver.v).any() or
                                    torch.isnan(solver.w).any()))
check("kinetic energy bounded", max(ke_hist) < 2.0 * ke_hist[0],
      f"(max {max(ke_hist):.4f} vs initial {ke_hist[0]:.4f})")
check("KE settled (last two samples within 2%)",
      abs(ke_hist[-1] - ke_hist[-2]) < 0.02 * ke_hist[-1])

print("\n2. Immersed boundary behaviour")
check("slip small (< 5% U_bulk)", slip_hist[-1] < 0.05,
      f"(slip {slip_hist[-1]:.4f})")
check("drag opposes flow", drag_hist[-1] < 0, f"(Fx {drag_hist[-1]:.4f})")
check("drag settled (within 5%)",
      abs(drag_hist[-1] - drag_hist[-2]) < 0.05 * abs(drag_hist[-1]))

print("\n3. Incompressibility and flow rate")
div = compute_divergence(solver.u, solver.v, solver.w, solver.nx, solver.ny,
                         solver.nz, solver.dx, solver.dy, solver.dz_f)
max_div = div.abs().max().item()
check("max|div| < 1e-9", max_div < 1e-9, f"({max_div:.2e})")
u_bulk = compute_bulk_velocity(solver.u, solver.cell_vol_ratio, solver.total_volume).item()
check("bulk velocity held at U_bulk (constant flow rate)", abs(u_bulk - 1.0) < 0.01,
      f"(u_bulk {u_bulk:.5f})")

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("All single-filament integration checks passed.")
