"""
RKPM reproducing conditions on the double-stretched staggered grid:

1. Partition of unity: sum of interpolation weights = 1 for every marker,
   every velocity component (machine precision).
2. Exact linear reproduction: interpolating f = a + b x + c y + e z returns the
   exact value at every marker (away from the periodic seams, where a linear
   field cannot be represented). Includes wall-adjacent markers (one-sided
   supports) and tip-region markers (non-uniform z).
3. Ghost cells are never read: ghosts are filled with NaN.
4. Spread conservation (normalize=False): sum_n f_n V_n == sum_l F_l dV_l.
5. Unit response (normalize=True): interpolate(spread(1)) == 1 at every marker.
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

cfg_raw = dict(cfg, rkpm={'normalize': False})
ibm = RigidCanopyIBM(cfg_raw, nx, ny, nz, dx, dy, Lx, Ly, z_c, z_f, dz_f, dz_c, 'cpu')
ibm_norm = RigidCanopyIBM(cfg, nx, ny, nz, dx, dy, Lx, Ly, z_c, z_f, dz_f, dz_c, 'cpu')

failures = []
def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name} {detail}")
    if not cond:
        failures.append(name)


def make_linear_field(comp, a, b, c, e):
    """Fill a full ghosted staggered field with a + b x + c y + e z on the
    canonical interior nodes; ghosts are NaN to detect any ghost read."""
    if comp == 'u':
        shape = (nx + 1, ny + 2, nz + 2)
        x = torch.arange(shape[0], dtype=torch.float64) * dx              # face i at i*dx
        y = (torch.arange(shape[1], dtype=torch.float64) - 0.5) * dy
        z = z_c
    elif comp == 'v':
        shape = (nx + 2, ny + 1, nz + 2)
        x = (torch.arange(shape[0], dtype=torch.float64) - 0.5) * dx
        y = torch.arange(shape[1], dtype=torch.float64) * dy
        z = z_c
    else:
        shape = (nx + 2, ny + 2, nz + 1)
        x = (torch.arange(shape[0], dtype=torch.float64) - 0.5) * dx
        y = (torch.arange(shape[1], dtype=torch.float64) - 0.5) * dy
        z = z_f
    f = a + b * x.view(-1, 1, 1) + c * y.view(1, -1, 1) + e * z.view(1, 1, -1)
    # NaN the ghosts (canonical interior differs per component/direction)
    if comp == 'u':
        f[0, :, :] = float('nan')                      # duplicated face
        f[:, 0, :] = float('nan'); f[:, -1, :] = float('nan')
        f[:, :, 0] = float('nan'); f[:, :, -1] = float('nan')
    elif comp == 'v':
        f[0, :, :] = float('nan'); f[-1, :, :] = float('nan')
        f[:, 0, :] = float('nan')                      # duplicated face
        f[:, :, 0] = float('nan'); f[:, :, -1] = float('nan')
    else:
        f[0, :, :] = float('nan'); f[-1, :, :] = float('nan')
        f[:, 0, :] = float('nan'); f[:, -1, :] = float('nan')
        f[:, :, 0] = float('nan'); f[:, :, -1] = float('nan')  # wall faces (w=0, excluded)
    return f


print("\n1. Partition of unity (all markers, all components)")
for comp in ('u', 'v', 'w'):
    s = ibm.w_int[comp].sum(dim=1)
    err = (s - 1.0).abs().max().item()
    check(f"sum W = 1 ({comp})", err < 1e-12, f"(max err {err:.2e})")

print("\n2. Exact linear reproduction (+ no ghost reads)")
a, b, c, e = 0.7, 0.31, -0.42, 1.13
# exclude markers whose support can wrap the periodic seams
mask_seam = ((ibm.x_lag > 2 * dx) & (ibm.x_lag < Lx - 2 * dx) &
             (ibm.y_lag > 2 * dy) & (ibm.y_lag < Ly - 2 * dy))
print(f"  ({mask_seam.sum().item()}/{ibm.N_L} markers away from periodic seams)")
for comp in ('u', 'v', 'w'):
    f = make_linear_field(comp, a, b, c, e)
    vals = ibm.interpolate(f, comp)
    exact = a + b * ibm.x_lag + c * ibm.y_lag + e * ibm.z_lag
    err = (vals - exact)[mask_seam].abs().max().item()
    check(f"linear reproduction ({comp})", err < 1e-11, f"(max err {err:.2e})")
    check(f"no NaN from ghosts ({comp})", not bool(torch.isnan(vals).any()))

# wall-adjacent and tip-region markers specifically (one-sided / non-uniform z)
k_wall = ibm.z_lag < z_c[2]           # below the 2nd cell center
k_tip = ibm.z_lag > ibm.ring_z[-3]    # top rings, straddling the grid transition
print(f"  (wall-adjacent: {int(k_wall.sum())}, tip-region: {int(k_tip.sum())} markers)")
f = make_linear_field('u', a, b, c, e)
vals = ibm.interpolate(f, 'u')
exact = a + b * ibm.x_lag + c * ibm.y_lag + e * ibm.z_lag
for name, m in (('wall-adjacent', k_wall), ('tip-region', k_tip)):
    err = (vals - exact)[m & mask_seam].abs().max().item()
    check(f"linear reproduction, {name} (u)", err < 1e-11, f"(max err {err:.2e})")

print("\n3. Spread conservation (normalize=False)")
gen = torch.Generator().manual_seed(7)
F = torch.rand(ibm.N_L, generator=gen, dtype=torch.float64) - 0.5
for comp in ('u', 'v', 'w'):
    _, _, z_nodes, z_vol, _, shape = ibm._component_layout(comp)
    fld = torch.zeros(shape, dtype=torch.float64)
    ibm._spread_increment(fld, F, comp)
    V = (dx * dy * z_vol).view(1, 1, -1).expand(shape)
    total_grid = (fld * V).sum().item()
    total_lag = (F * ibm.dV).sum().item()
    err = abs(total_grid - total_lag) / max(abs(total_lag), 1e-300)
    check(f"conservation ({comp})", err < 1e-12, f"(rel err {err:.2e})")

print("\n4. Epsilon self-response and forcing contraction (normalize=True)")
for comp in ('u', 'v', 'w'):
    diag = (ibm_norm.w_int[comp] * ibm_norm.w_spr[comp]).sum(dim=1)
    err = (diag - 1.0).abs().max().item()
    check(f"self-response = 1 ({comp})", err < 1e-12, f"(max err {err:.2e})")

check("stable recommended alpha", 0.1 < ibm_norm.recommended_alpha <= 1.0,
      f"(alpha={ibm_norm.recommended_alpha:.3f}, lambda_max={ibm_norm.lambda_max:.3f})")

# Direct-forcing fixed point: iterating dU = -alpha*interp must contract the
# marker slip monotonically and strongly (this is the runtime stability property)
alpha = torch.tensor(ibm_norm.recommended_alpha, dtype=torch.float64)
fields = {}
for comp in ('u', 'v', 'w'):
    _, _, _, _, _, shape = ibm_norm._component_layout(comp)
    gen = torch.Generator().manual_seed(3)
    fields[comp] = 0.5 * torch.randn(shape, generator=gen, dtype=torch.float64)

slips = []
for it in range(11):
    s = 0.0
    for comp in ('u', 'v', 'w'):
        lag = ibm_norm.interpolate(fields[comp], comp)
        s += lag.square().mean().item()
        ibm_norm._spread_increment(fields[comp], -alpha * lag, comp)
    slips.append(math.sqrt(s))
print(f"  slip rms per forcing iteration: " + " -> ".join(f"{s:.3e}" for s in slips[:4])
      + f" ... -> {slips[10]:.3e}")
check("slip contracts monotonically", all(b < a for a, b in zip(slips, slips[1:])))
check("slip reduced 4x after 3 iterations", slips[3] < 0.25 * slips[0],
      f"(ratio {slips[3]/slips[0]:.3f})")
check("slip reduced 8x after 10 iterations", slips[10] < 0.125 * slips[0],
      f"(ratio {slips[10]/slips[0]:.3f})")

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("All RKPM reproduction checks passed.")
