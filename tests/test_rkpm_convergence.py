"""
RKPM interpolation convergence: with linear reproducing conditions the
interpolation of a smooth field converges at 2nd order under grid refinement.
The canopy (filament positions, diameter, seed) is held fixed in physical
space; only the grid is refined.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import torch
from utils import generate_double_stretched_grid
from canopy import RigidCanopyIBM

torch.set_default_dtype(torch.float64)

Lx, Ly, Lz = 2 * math.pi, 1.5 * math.pi, 1.0
h = 0.25


def smooth(x, y, z):
    return (torch.sin(2 * math.pi * x / Lx) * torch.cos(4 * math.pi * y / Ly)
            * torch.sin(0.5 * math.pi * z / h))


def interp_error(nx, ny, nz_canopy, nz_outer, diameter):
    dx, dy = Lx / nx, Ly / ny
    z_f, z_c, dz_f, dz_c = generate_double_stretched_grid(
        nz_canopy, nz_outer, h, Lz, 2.0, 'auto')
    nz = len(dz_f)
    cfg = {'h': h, 'n_fil_x': 12, 'n_fil_y': 9, 'placement': 'random_in_tile',
           'seed': 42, 'diameter': diameter, 'markers_per_ring': 4,
           'rkpm': {'normalize': False}}
    ibm = RigidCanopyIBM(cfg, nx, ny, nz, dx, dy, Lx, Ly, z_c, z_f, dz_f, dz_c, 'cpu')

    # u-component field on its staggered nodes
    x = torch.arange(nx + 1, dtype=torch.float64) * dx
    y = (torch.arange(ny + 2, dtype=torch.float64) - 0.5) * dy
    f = smooth(x.view(-1, 1, 1), y.view(1, -1, 1), z_c.view(1, 1, -1))

    vals = ibm.interpolate(f, 'u')
    exact = smooth(ibm.x_lag, ibm.y_lag, ibm.z_lag)
    return (vals - exact).abs().max().item(), (vals - exact).square().mean().sqrt().item()


d = 2.2 * (Lx / 96)  # fixed physical diameter across resolutions
err_max_c, err_rms_c = interp_error(96, 72, 24, 40, d)
err_max_f, err_rms_f = interp_error(192, 144, 48, 80, d)

rate_max = math.log2(err_max_c / err_max_f)
rate_rms = math.log2(err_rms_c / err_rms_f)
print(f"\ncoarse: max {err_max_c:.3e}, rms {err_rms_c:.3e}")
print(f"fine:   max {err_max_f:.3e}, rms {err_rms_f:.3e}")
print(f"observed order: max {rate_max:.2f}, rms {rate_rms:.2f} (expect ~2)")

ok = rate_max > 1.7 and rate_rms > 1.7
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
