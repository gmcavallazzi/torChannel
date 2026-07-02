"""
Benchmark: canopy IBM overhead at production resolution (Monti config grid).

Runs the same grid with canopy disabled and enabled, measures steady-state
ms/step after warmup (past torch.compile/JIT compilation), reports overhead.

Usage:
    PYTORCH_JIT=0 TORCHANNEL_COMPILE=1 TORCHANNEL_POISSON_CUDAGRAPH=1 \
        python tests/bench_canopy_overhead.py [n_warmup] [n_bench]
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import math
import tempfile
import yaml
import torch
from solver import ChannelFlow

torch.set_default_dtype(torch.float64)

n_warmup = int(sys.argv[1]) if len(sys.argv) > 1 else 30
n_bench = int(sys.argv[2]) if len(sys.argv) > 2 else 50


def bench(canopy_enabled):
    workdir = tempfile.mkdtemp(prefix='bench_canopy_')
    config = {
        'grid': {'nx': 576, 'ny': 432, 'nz': 300, 'nz_canopy': 100, 'nz_outer': 200},
        'domain': {'Lx': 2 * math.pi, 'Ly': 1.5 * math.pi, 'Lz': 1.0,
                   'stretching_type': 'double', 'z_transition': 0.25,
                   'gamma_canopy': 2.2, 'gamma_outer': 'auto'},
        'flow': {'Re': 6000.0, 'Re_tau': 1157.0, 'U_bulk': 1.0, 'gamma': 2.6},
        'boundary_conditions': {'top_wall': {'type': 'neumann'}},
        'time': {'dt': 0.0005, 'n_steps': 10, 't_max': 1e9, 'CFL_target': 0.3,
                 'dt_update_interval': 0, 'scheme': 'IMEX'},
        'initialization': {'type': 'vortices', 'perturbation_intensity': 0.15,
                           'n_vortices': 6},
        'solver': {'type': 'fft'},
        'compute': {'device': 'auto'},
        'output': {'results_folder': workdir, 'n_out': 1000000, 'n_save': 1000000},
        'statistics': {'n_stats': 0},
        'canopy': {'enabled': canopy_enabled, 'h': 0.25, 'n_fil_x': 48, 'n_fil_y': 36,
                   'placement': 'random_in_tile', 'seed': 20260702, 'diameter': 0.024,
                   'markers_per_ring': 4,
                   'forcing': {'alpha': 'auto', 'ramp_steps': 0, 'n_iter': 2}},
    }
    cfg_path = os.path.join(workdir, 'config.yaml')
    with open(cfg_path, 'w') as f:
        yaml.safe_dump(config, f)

    solver = ChannelFlow(config_file=cfg_path)
    dev = solver.device

    for step in range(1, n_warmup + 1):
        solver.current_step = step
        solver.step_imex(solver.dt)
    if dev.type == 'cuda':
        torch.cuda.synchronize()

    t0 = time.time()
    for step in range(n_warmup + 1, n_warmup + n_bench + 1):
        solver.current_step = step
        solver.step_imex(solver.dt)
    if dev.type == 'cuda':
        torch.cuda.synchronize()
    ms = (time.time() - t0) / n_bench * 1000.0

    del solver
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
    return ms


print(f"\nBenchmark: 576x432x300, {n_warmup} warmup + {n_bench} timed steps\n")
ms_base = bench(False)
print(f"\n  baseline (no canopy):  {ms_base:8.2f} ms/step")
ms_canopy = bench(True)
overhead = (ms_canopy - ms_base) / ms_base * 100.0
print(f"\n  baseline (no canopy):  {ms_base:8.2f} ms/step")
print(f"  with canopy (n_iter=2): {ms_canopy:8.2f} ms/step")
print(f"  overhead: {ms_canopy - ms_base:+.2f} ms/step ({overhead:+.1f}%)")

ok = overhead < 40.0
print("\nPASS" if ok else "\nFAIL (overhead above 40%)")
sys.exit(0 if ok else 1)
