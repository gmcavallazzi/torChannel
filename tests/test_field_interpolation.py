"""
Test the 'interpolate' initialization: a saved field on one grid (closed
channel, symmetric stretching, Lz=2) initializes a run on a different grid
(open channel, double-stretched, Lz=1, different nx/ny).

1. Exactness: a field linear in z is reproduced exactly on the new grid
   (trilinear interpolation is exact for linear fields).
2. End-to-end through ChannelFlow: bulk velocity rescaled to U_bulk, initial
   projection leaves a divergence-free field, 20 IMEX steps run NaN-free.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import tempfile
import yaml
import torch
from utils import generate_grid, generate_double_stretched_grid, save_flow_fields, \
    compute_divergence, compute_bulk_velocity
from initflow import initialize_flow_interpolated
from solver import ChannelFlow

torch.set_default_dtype(torch.float64)

workdir = tempfile.mkdtemp(prefix='interp_init_')
failures = []
def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name} {detail}")
    if not cond:
        failures.append(name)

# ---- source: closed channel, Lz=2, symmetric tanh grid, 64x48x64
nxs, nys, nzs = 64, 48, 64
Lxs, Lys, Lzs = 4 * math.pi, 2 * math.pi, 2.0
z_f_s, z_c_s, dz_f_s, dz_c_s = generate_grid(1.8, nzs, Lzs)

u_s = torch.zeros(nxs + 1, nys + 2, nzs + 2)
v_s = torch.zeros(nxs + 2, nys + 1, nzs + 2)
w_s = torch.zeros(nxs + 2, nys + 2, nzs + 1)
p_s = torch.zeros(nxs + 2, nys + 2, nzs + 2)

# linear-in-z u plus a smooth periodic 3D perturbation on all components
a, e = 0.3, 0.9
u_s += a + e * z_c_s.view(1, 1, -1)
xs = torch.arange(nxs + 1).view(-1, 1, 1) * (Lxs / nxs)
ys = (torch.arange(nys + 2).view(1, -1, 1) - 0.5) * (Lys / nys)
u_pert = 0.1 * torch.sin(2 * math.pi * xs / Lxs) * torch.cos(2 * math.pi * ys / Lys) \
    * torch.sin(math.pi * z_c_s.view(1, 1, -1) / Lzs)
xc = (torch.arange(nxs + 2).view(-1, 1, 1) - 0.5) * (Lxs / nxs)
yc = (torch.arange(nys + 2).view(1, -1, 1) - 0.5) * (Lys / nys)
v_s += 0.05 * torch.sin(4 * math.pi * xc / Lxs) * torch.sin(math.pi * z_c_s.view(1, 1, -1) / Lzs)
w_s += 0.05 * torch.cos(2 * math.pi * xc / Lxs) * torch.sin(math.pi * z_f_s.view(1, 1, -1) / Lzs)

src_file = os.path.join(workdir, 'source_field.npz')
save_flow_fields(u_s + u_pert, v_s, w_s, p_s, z_c_s, z_f_s, Lxs, Lys,
                 1000, 12.5, 0.05, 0.003, workdir, 'source_field.npz')

# ---- target: open channel, Lz=1, double-stretched, 48x36
nx, ny = 48, 36
Lx, Ly, Lz = 2 * math.pi, 1.5 * math.pi, 1.0
z_f, z_c, dz_f, dz_c = generate_double_stretched_grid(16, 24, 0.25, Lz, 1.5, 'auto')
nz = len(dz_f)

print("\n1. Linear-field exactness (pure interpolation, no perturbation)")
lin_file = os.path.join(workdir, 'linear_field.npz')
save_flow_fields(u_s, v_s * 0, w_s * 0, p_s, z_c_s, z_f_s, Lxs, Lys,
                 0, 0.0, 0.0, 0.0, workdir, 'linear_field.npz')
u_t, v_t, w_t, p_t = initialize_flow_interpolated(
    lin_file, nx, ny, nz, Lx, Ly, Lz, z_c, z_f, device='cpu')
exact = a + e * z_c.view(1, 1, -1).expand_as(u_t)
# interior only: ghost z centers of the target lie outside the source range
err = (u_t - exact)[:, :, 1:-1].abs().max().item()
check("u = a + e*z reproduced exactly", err < 1e-13, f"(max err {err:.2e})")
check("v, w stay zero", v_t.abs().max().item() == 0.0 and w_t.abs().max().item() == 0.0)

print("\n2. End-to-end ChannelFlow init with perturbed source")
config = {
    'grid': {'nx': nx, 'ny': ny, 'nz': nz, 'nz_canopy': 16, 'nz_outer': 24},
    'domain': {'Lx': Lx, 'Ly': Ly, 'Lz': Lz, 'stretching_type': 'double',
               'z_transition': 0.25, 'gamma_canopy': 1.5, 'gamma_outer': 'auto'},
    'flow': {'Re': 1000.0, 'Re_tau': 180.0, 'U_bulk': 1.0, 'gamma': 2.0},
    'boundary_conditions': {'top_wall': {'type': 'neumann'}},
    'time': {'dt': 2.0e-3, 'n_steps': 20, 't_max': 1000.0, 'CFL_target': 0.5,
             'dt_update_interval': 0, 'scheme': 'IMEX'},
    'initialization': {'type': 'interpolate', 'field_file': src_file,
                       'source_half': 'lower'},
    'solver': {'type': 'fft'},
    'compute': {'device': 'auto'},
    'output': {'results_folder': workdir, 'n_out': 10, 'n_save': 100000},
    'statistics': {'n_stats': 0},
    'canopy': {'enabled': True, 'h': 0.25, 'n_fil_x': 6, 'n_fil_y': 5,
               'placement': 'random_in_tile', 'seed': 7, 'markers_per_ring': 4,
               'forcing': {'alpha': 'auto', 'ramp_steps': 10, 'n_iter': 1}},
}
cfg_path = os.path.join(workdir, 'config.yaml')
with open(cfg_path, 'w') as f:
    yaml.safe_dump(config, f)

solver = ChannelFlow(config_file=cfg_path)

u_bulk0 = compute_bulk_velocity(solver.u, solver.cell_vol_ratio, solver.total_volume).item()
check("bulk velocity rescaled to U_bulk", abs(u_bulk0 - 1.0) < 1e-10, f"({u_bulk0:.8f})")
check("fresh start (time reset)", solver.initial_step == 0 and solver.time == 0.0)

div0 = compute_divergence(solver.u, solver.v, solver.w, solver.nx, solver.ny,
                          solver.nz, solver.dx, solver.dy, solver.dz_f)
check("divergence-free after initial projection", div0.abs().max().item() < 1e-9,
      f"({div0.abs().max().item():.2e})")

for step in range(1, 21):
    solver.current_step = step
    solver.step_imex(solver.dt)
check("20 IMEX steps NaN-free", not bool(torch.isnan(solver.u).any() or
                                          torch.isnan(solver.w).any()))

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("All field-interpolation checks passed.")
