"""
Integration test: full random-in-tile canopy array (12x9 filaments) in a
laminar open channel, IMEX stepping through ChannelFlow.

Checks the canopy-flow physics wiring:
- flow decelerates INSIDE the canopy, fast flow above (inflectional profile)
- mean spanwise drag ~ 0 (random placement has no preferred y direction)
- constant-flow-rate controller: u_bulk held, forcing rises to positive value
  balancing wall friction + canopy drag
- steady-state streamwise momentum balance: forcing*V ~ |F_canopy| + tau_wall*A
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import tempfile
import yaml
import torch
from solver import ChannelFlow
from utils import compute_bulk_velocity

torch.set_default_dtype(torch.float64)

workdir = tempfile.mkdtemp(prefix='ibm_array_')
config = {
    'grid': {'nx': 48, 'ny': 36, 'nz': 40, 'nz_canopy': 16, 'nz_outer': 24},
    'domain': {'Lx': 2 * math.pi, 'Ly': 1.5 * math.pi, 'Lz': 1.0,
               'stretching_type': 'double', 'z_transition': 0.25,
               'gamma_canopy': 1.5, 'gamma_outer': 'auto'},
    'flow': {'Re': 1000.0, 'Re_tau': 180.0, 'U_bulk': 1.0, 'gamma': 2.0},
    'boundary_conditions': {'top_wall': {'type': 'neumann'}},
    'time': {'dt': 2.0e-3, 'n_steps': 800, 't_max': 1000.0, 'CFL_target': 0.5,
             'dt_update_interval': 0, 'scheme': 'IMEX'},
    'initialization': {'type': 'parabolic', 'perturbation_intensity': 0.0},
    'solver': {'type': 'fft'},
    'compute': {'device': 'auto'},
    'output': {'results_folder': workdir, 'n_out': 100, 'n_save': 100000},
    'statistics': {'n_stats': 0},
    'canopy': {
        'enabled': True, 'h': 0.25, 'n_fil_x': 12, 'n_fil_y': 9,
        'placement': 'random_in_tile', 'seed': 42, 'markers_per_ring': 4,
        'forcing': {'alpha': 'auto', 'ramp_steps': 100, 'n_iter': 2},
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

n_steps = 800
for step in range(1, n_steps + 1):
    solver.current_step = solver.initial_step + step
    solver.step_imex(solver.dt)

check_nan = bool(torch.isnan(solver.u).any() or torch.isnan(solver.v).any())
drag = solver.canopy_drag.cpu()
forcing = float(solver.forcing) if not torch.is_tensor(solver.forcing) else solver.forcing.item()
u_bulk = compute_bulk_velocity(solver.u, solver.cell_vol_ratio, solver.total_volume).item()

# Mean streamwise velocity profile U(z) (interior)
U_z = solver.u[1:solver.nx + 1, 1:solver.ny + 1, 1:solver.nz + 1].mean(dim=(0, 1)).cpu()
z_centers = solver.z_c[1:solver.nz + 1].cpu()
in_canopy = z_centers < 0.20          # well inside the canopy
above = (z_centers > 0.35) & (z_centers < 0.9)
U_in = U_z[in_canopy].mean().item()
U_above = U_z[above].mean().item()

print(f"\n  U inside canopy = {U_in:.4f}, U above = {U_above:.4f}")
print(f"  drag = ({drag[0]:.4f}, {drag[1]:.5f}, {drag[2]:.5f}), forcing = {forcing:.5f}")

print("\n1. Stability and flow rate")
check("no NaN", not check_nan)
check("u_bulk held at 1", abs(u_bulk - 1.0) < 0.01, f"({u_bulk:.5f})")
check("forcing positive (balances drag)", forcing > 0, f"({forcing:.5f})")

print("\n2. Canopy-flow structure")
check("strong deceleration inside canopy", U_in < 0.5 * U_above,
      f"(ratio {U_in/U_above:.3f})")
check("mean spanwise drag ~ 0", abs(drag[1].item()) < 0.05 * abs(drag[0].item()),
      f"(Fy/Fx = {drag[1].item()/drag[0].item():.4f})")

print("\n3. Streamwise momentum balance (steady state)")
# forcing * V = tau_wall * A + |F_drag,x|
V = solver.Lx * solver.Ly * solver.Lz
A = solver.Lx * solver.Ly
dUdz_wall = (U_z[0] / z_centers[0]).item()   # no-slip wall, first cell center
tau_wall = solver.nu * dUdz_wall
lhs = forcing * V
rhs = tau_wall * A + abs(drag[0].item())
err = abs(lhs - rhs) / abs(lhs)
check("forcing*V ~ tau_wall*A + |Fx|", err < 0.1,
      f"(lhs {lhs:.4f} vs rhs {rhs:.4f}, err {err*100:.1f}%)")

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("All canopy-array integration checks passed.")
