"""Verify the parabolic (space-marching) steady scalar solver, scalar.march_scalar_steady.

For a uniform plug flow u=U (v=w=0) the parabolic balance

    U dc/dx = D d2c/dz2

is the 1-D heat equation with the streamwise coordinate playing the role of time, t = x/U.
A sharp inlet interface at z0 therefore develops downstream as the analytic erf solution

    c(x, z) = 1/2 [ 1 + erf( (z - z0) / sqrt(4 D x / U) ) ] .

This is the marcher's analogue of the erf check in test_scalar.py (which validates the
time-stepping diffusion) and confirms zero spurious streamwise/transverse diffusion, mean
conservation and boundedness -- independent of the full solver and of the campaign data.

Run:  PYTORCH_JIT=0 python tests/test_scalar_parabolic.py
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from scalar import march_scalar_steady

# --- uniform duct grid (uniform z so the erf solution is exact) ---------------
nx, ny, nz = 200, 8, 128
Lx, Ly, Lz = 2.0, 1.0, 1.0
dx, dy, dz = Lx / nx, Ly / ny, Lz / nz
U, D, z0 = 1.0, 1.0e-3, 0.5 * Lz

dz_f = torch.full((nz,), dz)               # cell heights
dz_c = torch.full((nz + 1,), dz)           # centre-to-centre spacings (uniform)
z_c = (torch.arange(1, nz + 1) - 0.5) * dz  # interior cell centres

# uniform plug flow; v = w = 0
u = torch.full((nx + 2, ny + 2, nz + 2), U)
v = torch.zeros_like(u)
w = torch.zeros_like(u)

# sharp inlet z-interface (1-cell tanh ~ step), homogeneous in y
zc_g = (torch.arange(nz + 2) - 0.5) * dz   # incl. ghosts
inlet_line = 0.5 * (1.0 + torch.tanh((zc_g - z0) / dz))
c_inlet = inlet_line.view(1, nz + 2).expand(ny + 2, nz + 2).contiguous()

c = march_scalar_steady(c_inlet, u, v, w, nx, ny, nz, dx, dy, dz_c, dz_f, D,
                        bc_y='wall', wall_bc='neumann', cross_adv=False,
                        n_inner=60, tol=1e-13, verbose=False)

ci = c[1:nx + 1, 1:ny + 1, 1:nz + 1]

# [1] erf comparison at a downstream plane (origin = inlet plane ii=1) ----------
ii = 150
prof = ci[ii - 1].mean(dim=0).cpu().numpy()        # average over y (should be uniform)
t = (ii - 1) * dx / U                               # marched distance / speed
width = math.sqrt(4.0 * D * t)
erf = 0.5 * (1.0 + torch.erf((z_c - z0) / width)).cpu().numpy()
err = float(np.max(np.abs(prof - erf)))
print(f"[1] parabolic vs erf @ x={(ii-1)*dx:.3f} (width={width:.4f}): max|num-erf| = {err:.2e}")

# [2] homogeneity in y (no spurious transverse structure) ----------------------
y_spread = float((ci[ii - 1].max(dim=0).values - ci[ii - 1].min(dim=0).values).abs().max())
print(f"[2] y-homogeneity: max spread across y = {y_spread:.2e}")

# [3] mean conservation per plane (no-flux z walls -> column mean = 0.5) --------
plane_mean = ci.mean(dim=(1, 2)).cpu().numpy()
mean_err = float(np.max(np.abs(plane_mean - 0.5)))
print(f"[3] mean conservation: max|mean(x) - 0.5| = {mean_err:.2e}")

# [4] boundedness --------------------------------------------------------------
cmin, cmax = float(ci.min()), float(ci.max())
print(f"[4] boundedness: c in [{cmin:.6f}, {cmax:.6f}]")

ok = (err < 1e-2 and y_spread < 1e-10 and mean_err < 1e-9
      and cmin > -1e-9 and cmax < 1.0 + 1e-9)
print("\n" + ("PARABOLIC MARCH TEST PASSED" if ok else "PARABOLIC MARCH TEST FAILED"))
sys.exit(0 if ok else 1)
