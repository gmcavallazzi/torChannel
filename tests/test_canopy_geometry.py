"""
Test canopy geometry generation (canopy.RigidCanopyIBM):
- filament counts and random-in-tile placement (containment, min separation, seed reproducibility)
- ring/marker layout (counts, z-range, surface radius)
- marker volumes and solidity recomputed from the configuration
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import torch
from utils import generate_double_stretched_grid
from canopy import RigidCanopyIBM

torch.set_default_dtype(torch.float64)

# Small Monti-like setup (coarse)
nx, ny = 96, 72
Lx, Ly, Lz = 2 * math.pi, 1.5 * math.pi, 1.0
h = 0.25
nz_canopy, nz_outer = 24, 40
dx, dy = Lx / nx, Ly / ny

z_f, z_c, dz_f, dz_c = generate_double_stretched_grid(
    nz_canopy, nz_outer, h, Lz, 2.0, 'auto')
nz = len(dz_f)

cfg = {
    'h': h, 'n_fil_x': 12, 'n_fil_y': 9,
    'placement': 'random_in_tile', 'seed': 42,
    'diameter': 2.2 * dx, 'markers_per_ring': 4,
}

ibm = RigidCanopyIBM(cfg, nx, ny, nz, dx, dy, Lx, Ly, z_c, z_f, dz_f, dz_c, 'cpu')

failures = []
def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name} {detail}")
    if not cond:
        failures.append(name)

print("\n1. Filament placement")
n_fil = cfg['n_fil_x'] * cfg['n_fil_y']
check("filament count", ibm.n_fil == n_fil, f"({ibm.n_fil})")

tile_x, tile_y = Lx / cfg['n_fil_x'], Ly / cfg['n_fil_y']
ix = torch.div(ibm.fil_x, tile_x, rounding_mode='floor')
iy = torch.div(ibm.fil_y, tile_y, rounding_mode='floor')
one_per_tile = len(set(zip(ix.tolist(), iy.tolist()))) == n_fil
check("exactly one filament per tile", one_per_tile)

# containment with margin d/2
lx_in = (ibm.fil_x - ix * tile_x >= 0.5 * ibm.diameter - 1e-12) & \
        (ibm.fil_x - ix * tile_x <= tile_x - 0.5 * ibm.diameter + 1e-12)
ly_in = (ibm.fil_y - iy * tile_y >= 0.5 * ibm.diameter - 1e-12) & \
        (ibm.fil_y - iy * tile_y <= tile_y - 0.5 * ibm.diameter + 1e-12)
check("cross-sections inside tiles", bool((lx_in & ly_in).all()))

# min pairwise distance (periodic) >= d
dxp = (ibm.fil_x.view(-1, 1) - ibm.fil_x.view(1, -1)).abs()
dxp = torch.minimum(dxp, Lx - dxp)
dyp = (ibm.fil_y.view(-1, 1) - ibm.fil_y.view(1, -1)).abs()
dyp = torch.minimum(dyp, Ly - dyp)
dist = torch.sqrt(dxp ** 2 + dyp ** 2)
dist.fill_diagonal_(float('inf'))
check("min filament separation >= d", dist.min().item() >= ibm.diameter - 1e-12,
      f"(min {dist.min().item():.4f} vs d {ibm.diameter:.4f})")

# placement is genuinely non-regular
off_x = (ibm.fil_x - (ix + 0.5) * tile_x).abs()
check("placement is not regular", off_x.max().item() > 0.05 * tile_x,
      f"(max offset {off_x.max().item():.4f})")

# reproducibility
ibm2 = RigidCanopyIBM(cfg, nx, ny, nz, dx, dy, Lx, Ly, z_c, z_f, dz_f, dz_c, 'cpu')
check("seed reproducibility", bool(torch.equal(ibm.fil_x, ibm2.fil_x) and
                                    torch.equal(ibm.x_lag, ibm2.x_lag)))

print("\n2. Markers")
check("rings = canopy cells", ibm.n_rings == nz_canopy, f"({ibm.n_rings})")
check("marker count", ibm.N_L == n_fil * ibm.n_rings * 4, f"({ibm.N_L})")
check("marker z in (0, h)", bool((ibm.z_lag > 0).all() and (ibm.z_lag < h).all()),
      f"(z in [{ibm.z_lag.min():.4f}, {ibm.z_lag.max():.4f}])")

# markers on the filament surface at radius d/2
fx = ibm.fil_x.view(-1, 1, 1).expand(n_fil, ibm.n_rings, 4).reshape(-1)
fy = ibm.fil_y.view(-1, 1, 1).expand(n_fil, ibm.n_rings, 4).reshape(-1)
r = torch.sqrt((ibm.x_lag - fx) ** 2 + (ibm.y_lag - fy) ** 2)
check("markers at radius d/2", bool(torch.allclose(r, torch.full_like(r, 0.5 * ibm.diameter))),
      f"(r in [{r.min():.5f}, {r.max():.5f}])")

print("\n3. Volumes and solidity")
# total marker volume = canopy solid volume (n_fil * pi d^2/4 * h_covered)
h_covered = ibm.ring_dz.sum().item()  # canopy cells span [0, z_f at tip]
vol_expect = n_fil * 0.25 * math.pi * ibm.diameter ** 2 * h_covered
check("total marker volume = solid volume",
      abs(ibm.dV.sum().item() - vol_expect) / vol_expect < 1e-12,
      f"({ibm.dV.sum().item():.6e} vs {vol_expect:.6e})")

lam = ibm.diameter * h / (tile_x * tile_y)
print(f"  solidity lambda = {lam:.3f} (Monti target for production config: 0.35)")

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("All canopy geometry checks passed.")
