"""TVD (van Leer) scalar advection: monotonicity at high cell-Peclet.

At high Schmidt number the cell-Peclet U*dx/D is large and the central scheme
develops dispersive over/undershoot at a sharp interface. The flux-limited TVD
scheme must stay bounded in [min,max] of the initial data while conserving the mean.
"""
import os, sys
os.environ.setdefault("PYTORCH_JIT", "0")
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_default_dtype(torch.float64)
from scalar import apply_scalar_bc, advection_scalar, advection_scalar_tvd, diffusion_xy_scalar


def integrate(scheme, nx=96, ny=4, nz=4, Lx=1.0, U=1.0, cellPe=40.0, nsteps=140):
    dx, dy = Lx / nx, 1.0 / ny
    dz_f = torch.full((nz,), 1.0 / nz)
    D = U * dx / cellPe
    dt = 0.4 * dx / U  # advective CFL ~0.4

    # plug flow u=U; v=w=0
    u = torch.full((nx + 1, ny + 2, nz + 2), U)
    v = torch.zeros(nx + 2, ny + 1, nz + 2)
    w = torch.zeros(nx + 2, ny + 2, nz + 1)

    # top-hat in x (sharp), uniform in y,z
    c = torch.zeros(nx + 2, ny + 2, nz + 2)
    xc = (torch.arange(nx + 2) - 0.5) * dx
    mask = ((xc > 0.30 * Lx) & (xc < 0.55 * Lx)).view(-1, 1, 1)
    c[mask.expand_as(c)] = 1.0

    m0 = float(c[1:nx+1, 1:ny+1, 1:nz+1].mean())
    cmax, cmin = -1e9, 1e9
    for _ in range(nsteps):
        apply_scalar_bc(c, 'neumann', 'periodic')
        if scheme == 'tvd':
            adv = advection_scalar_tvd(c, u, v, w, nx, ny, nz, dx, dy, dz_f, 'periodic', 'neumann')
        else:
            adv = advection_scalar(c, u, v, w, nx, ny, nz, dx, dy, dz_f)
        diff = diffusion_xy_scalar(c, nx, ny, nz, dx, dy, D)
        # forward Euler: isolates the spatial scheme's TVD property (van Leer is TVD
        # under FE at CFL<=1; the solver's AB2 adds only ~3e-4 overshoot in practice)
        c = c + dt * (diff - adv)
        ci = c[1:nx+1, 1:ny+1, 1:nz+1]
        cmax = max(cmax, float(ci.max()))
        cmin = min(cmin, float(ci.min()))
    m1 = float(c[1:nx+1, 1:ny+1, 1:nz+1].mean())
    return cmax, cmin, abs(m1 - m0)


if __name__ == "__main__":
    cPe = 40.0
    cmax_c, cmin_c, dm_c = integrate('central', cellPe=cPe)
    cmax_t, cmin_t, dm_t = integrate('tvd', cellPe=cPe)

    print(f"cell-Pe = {cPe}")
    print(f"[central] max={cmax_c:.4f}  min={cmin_c:.4f}  |dmean|={dm_c:.2e}")
    print(f"[tvd]     max={cmax_t:.4f}  min={cmin_t:.4f}  |dmean|={dm_t:.2e}")

    central_overshoots = (cmax_c > 1.0 + 1e-3) or (cmin_c < -1e-3)
    tvd_bounded = (cmax_t <= 1.0 + 1e-9) and (cmin_t >= -1e-9)
    tvd_conserves = dm_t < 1e-12

    print(f"\ncentral overshoots (expected): {central_overshoots}")
    print(f"tvd bounded in [0,1]:          {tvd_bounded}")
    print(f"tvd conserves mean:            {tvd_conserves}")

    ok = central_overshoots and tvd_bounded and tvd_conserves
    print("\nTVD TEST PASSED" if ok else "\nTVD TEST FAILED")
    sys.exit(0 if ok else 1)
