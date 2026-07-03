"""
Wiring test for the canopy-specific statistics:
- multi-plane 2D spectra (spectra_z list) with per-plane accumulators
- skewness third-moment profiles
- canopy drag profile f_x(z) fed from the IBM per-ring force
- state save/load roundtrip with the new keys
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import tempfile
import yaml
import numpy as np
import torch
from solver import ChannelFlow

torch.set_default_dtype(torch.float64)

workdir = tempfile.mkdtemp(prefix='canopy_stats_')
config = {
    'grid': {'nx': 48, 'ny': 36, 'nz': 40, 'nz_canopy': 16, 'nz_outer': 24},
    'domain': {'Lx': 2 * math.pi, 'Ly': 1.5 * math.pi, 'Lz': 1.25,
               'stretching_type': 'double', 'z_transition': 0.25,
               'gamma_canopy': 1.5, 'gamma_outer': 'auto'},
    'flow': {'Re': 1000.0, 'Re_tau': 180.0, 'U_bulk': 1.0, 'gamma': 2.0},
    'boundary_conditions': {'top_wall': {'type': 'neumann'}},
    'time': {'dt': 2.0e-3, 'n_steps': 100, 't_max': 1000.0, 'CFL_target': 0.5,
             'dt_update_interval': 0, 'scheme': 'IMEX'},
    'initialization': {'type': 'vortices', 'perturbation_intensity': 0.1, 'n_vortices': 4},
    'solver': {'type': 'fft'},
    'compute': {'device': 'auto'},
    'output': {'results_folder': workdir, 'n_out': 50, 'n_save': 100000},
    'statistics': {'n_stats': 10, 't_stats': 0.0,
                   'spectra_z': [0.125, 0.25, 0.75],
                   'output_file': 'stats.npz', 'state_file': 'stats_state.npz'},
    'canopy': {'enabled': True, 'h': 0.25, 'n_fil_x': 6, 'n_fil_y': 5,
               'placement': 'random_in_tile', 'seed': 7, 'markers_per_ring': 4,
               'forcing': {'alpha': 'auto', 'ramp_steps': 0, 'n_iter': 2}},
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

# run 60 steps, accumulating every 10; check profile/total consistency per sample
profile_total_err = 0.0
for step in range(1, 61):
    solver.current_step = step
    solver.step_imex(solver.dt)
    if solver.n_stats > 0 and step % solver.n_stats == 0:
        from utils import compute_u_tau
        ut = compute_u_tau(solver.u, solver.z_c, solver.nu,
                           top_wall_bc_type=solver.top_wall_bc_type)
        fx_prof = torch.zeros(solver.nz, dtype=torch.float64, device=solver.device)
        fx_prof[:solver.canopy.n_rings] = solver.canopy.last_fx_rings
        # same-step invariant: profile sums exactly to the total drag
        profile_total_err = max(profile_total_err,
                                abs(fx_prof.sum().item() - solver.canopy_drag[0].item()))
        solver.turbulence_stats.accumulate_statistics(solver.u, solver.v, solver.w, ut,
                                                      fx_profile=fx_prof)

ts = solver.turbulence_stats
stats = ts.finalize_statistics()
nkx, nky = solver.nx // 2, solver.ny // 2

print("\n1. Multi-plane spectra")
check("spectra_z saved", 'spectra_z' in stats,
      f"(z = {stats.get('spectra_z')})")
check("per-plane spectra shape", stats['E_uu_2d'].shape == (3, nkx, nky),
      f"({stats['E_uu_2d'].shape})")
check("spectra positive", bool((stats['E_uu_2d'] >= 0).all()))
check("planes near requested heights (coarse grid)",
      bool(np.allclose(stats['spectra_z'], [0.125, 0.25, 0.75], atol=0.06)))

print("\n2. Skewness moments")
check("uuu/www present and finite",
      np.isfinite(stats['uuu_mean']).all() and np.isfinite(stats['www_mean']).all())
check("nonzero third moments", np.abs(stats['uuu_mean']).max() > 0)

print("\n3. Canopy drag profile")
fx = stats['fx_profile_mean']
n_r = solver.canopy.n_rings
check("force nonzero inside canopy", np.abs(fx[:n_r]).max() > 0,
      f"(max |fx| in-canopy {np.abs(fx[:n_r]).max():.2e})")
check("zero above canopy", bool((fx[n_r:] == 0).all()))
check("profile sums exactly to total drag (same step)", profile_total_err < 1e-12,
      f"(max err {profile_total_err:.2e})")

print("\n4. State roundtrip")
state_path = os.path.join(workdir, 'state.npz')
ts.save_state(state_path)
n_before = ts.n_samples
uuu_before = ts.uuu_sum.clone()
Euu_before = ts.E_uu_2d_sum.clone()
ts.uuu_sum.zero_(); ts.E_uu_2d_sum.zero_(); ts.n_samples = 0
ts.load_state(state_path)
check("n_samples restored", ts.n_samples == n_before)
check("uuu_sum restored", bool(torch.allclose(ts.uuu_sum, uuu_before)))
check("spectra restored", bool(torch.allclose(ts.E_uu_2d_sum, Euu_before)))

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("All canopy statistics checks passed.")
