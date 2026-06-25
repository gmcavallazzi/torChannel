"""Verification tests for passive-scalar transport (scalar.py).

Run:  python tests/test_scalar.py   (torChannel style, exits nonzero on failure)

Covers:
  1. constant-field preservation (advection + diffusion of a uniform scalar)
  2. mean conservation under no-flux walls with a non-trivial velocity
  3. pure z-diffusion vs the analytic error-function solution
     -> certifies the physical diffusivity D = nu/Sc with NO numerical diffusion
  4. pure advection: low numerical diffusion + exact periodic return
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
torch.set_default_dtype(torch.float64)

from scalar import (apply_scalar_bc, advection_scalar, diffusion_xy_scalar,
                    solve_implicit_diffusion_scalar, scalar_stats)
from math import erf as _erf


def uniform_grid(nz, Lz):
    """Uniform z grid in the (z_f, z_c, dz_f, dz_c) layout used by the operators."""
    z_f = torch.linspace(0.0, Lz, nz + 1)
    z_c_inn = 0.5 * (z_f[:-1] + z_f[1:])
    z_c = torch.cat([torch.tensor([-z_c_inn[0]]), z_c_inn,
                     torch.tensor([2 * z_f[-1] - z_c_inn[-1]])])
    dz_f = z_f[1:] - z_f[:-1]
    dz_c = z_c[1:] - z_c[:-1]
    return z_f, z_c, dz_f, dz_c


def imex_scalar_step(c, u, v, w, nx, ny, nz, dx, dy, dz_c, dz_f, D, dt,
                     rhs_prev, wall_bc='neumann', theta=0.5):
    """One IMEX scalar step, identical to ChannelFlow.advance_scalar."""
    apply_scalar_bc(c, wall_bc)
    adv = advection_scalar(c, u, v, w, nx, ny, nz, dx, dy, dz_f)
    diff_xy = diffusion_xy_scalar(c, nx, ny, nz, dx, dy, D)
    rhs = diff_xy - adv
    c = c + dt * rhs if rhs_prev is None else c + dt * (1.5 * rhs - 0.5 * rhs_prev)
    apply_scalar_bc(c, wall_bc)
    c = solve_implicit_diffusion_scalar(c, float(dt), nx, ny, nz, dz_c, dz_f,
                                        D, theta=theta, wall_bc=wall_bc)
    apply_scalar_bc(c, wall_bc)
    return c, rhs


def _zeros_vel(nx, ny, nz):
    return (torch.zeros(nx + 1, ny + 2, nz + 2),
            torch.zeros(nx + 2, ny + 1, nz + 2),
            torch.zeros(nx + 2, ny + 2, nz + 1))


def test_constant_field():
    nx = ny = nz = 12
    Lx = Ly = Lz = 1.0
    _, _, dz_f, dz_c = uniform_grid(nz, Lz)
    c = torch.full((nx + 2, ny + 2, nz + 2), 0.42)
    u, v, w = _zeros_vel(nx, ny, nz)
    u[:] = 0.8; v[:] = -0.5  # arbitrary divergence-free-ish in-plane flow
    rhs_prev = None
    for _ in range(20):
        c, rhs_prev = imex_scalar_step(c, u, v, w, nx, ny, nz, Lx/nx, Ly/ny,
                                       dz_c, dz_f, D=0.01, dt=0.01, rhs_prev=rhs_prev)
    err = float((c[1:-1, 1:-1, 1:-1] - 0.42).abs().max())
    print(f"[1] constant field: max deviation = {err:.2e}")
    assert err < 1e-12, err
    return err


def test_mean_conservation():
    nx = ny = nz = 16
    Lx = Ly = Lz = 1.0
    _, _, dz_f, dz_c = uniform_grid(nz, Lz)
    torch.manual_seed(0)
    c = torch.rand(nx + 2, ny + 2, nz + 2)
    u, v, w = _zeros_vel(nx, ny, nz)
    u[:] = 1.3; v[:] = 0.7   # uniform => divergence-free; w=0 at walls (no-flux)
    m0 = scalar_stats(c, nx, ny, nz, dz_f)["mean"]
    rhs_prev = None
    for _ in range(50):
        c, rhs_prev = imex_scalar_step(c, u, v, w, nx, ny, nz, Lx/nx, Ly/ny,
                                       dz_c, dz_f, D=0.005, dt=0.01, rhs_prev=rhs_prev)
    m1 = scalar_stats(c, nx, ny, nz, dz_f)["mean"]
    err = abs(m1 - m0)
    print(f"[2] mean conservation (no-flux walls): |dmean| = {err:.2e}")
    assert err < 1e-12, err
    return err


def test_diffusion_vs_erf():
    """Pure z-diffusion of an erf interface must stay erf with the physical D."""
    nx = ny = 4
    nz = 256
    Lz = 2.0
    z_f, z_c, dz_f, dz_c = uniform_grid(nz, Lz)
    zc_i = z_c[1:nz+1]                      # interior cell centres
    z0 = 1.0
    nu, Sc = 0.02, 1.0
    D = nu / Sc

    t0, t1 = 1.0, 3.0                       # evolve erf(t0) -> should match erf(t1)
    def erf_profile(t):
        s = 2.0 * np.sqrt(D * t)
        vals = 0.5 * (1.0 + torch.tensor([_erf(float((z - z0) / s)) for z in zc_i]))
        return vals

    c = torch.zeros(nx + 2, ny + 2, nz + 2)
    c[:, :, 1:nz+1] = erf_profile(t0).view(1, 1, -1)
    u, v, w = _zeros_vel(nx, ny, nz)        # no advection

    dt = 0.01
    nsteps = int(round((t1 - t0) / dt))
    rhs_prev = None
    for _ in range(nsteps):
        c, rhs_prev = imex_scalar_step(c, u, v, w, nx, ny, nz, 1.0, 1.0,
                                       dz_c, dz_f, D=D, dt=dt, rhs_prev=rhs_prev,
                                       wall_bc='neumann')
    num = c[1:nx+1, 1:ny+1, 1:nz+1].mean(dim=(0, 1))   # (nz,)
    exact = erf_profile(t1)
    err = float((num - exact).abs().max())
    # infer effective diffusivity from the centre slope: dc/dz|z0 = 1/(2 sqrt(pi D t1))
    dz = float(dz_f[0])
    k0 = nz // 2
    slope = float((num[k0] - num[k0 - 1]) / dz)
    D_eff = 1.0 / (4 * np.pi * t1 * slope**2)
    print(f"[3] diffusion vs erf: max|num-erf| = {err:.2e}, "
          f"D_eff/D - 1 = {D_eff/D - 1:+.3%}")
    assert err < 5e-3, err
    assert abs(D_eff / D - 1) < 0.03, D_eff / D
    return err


def test_pure_advection():
    """Uniform u, D=0: a cosine must translate with ~no numerical diffusion and
    return to itself after one period."""
    nx = 64
    ny = nz = 4
    Lx = Ly = Lz = 1.0
    _, _, dz_f, dz_c = uniform_grid(nz, Lz)
    xc = (torch.arange(nx + 2) - 0.5) * (Lx / nx)
    c = torch.zeros(nx + 2, ny + 2, nz + 2)
    c[:, :, :] = (0.5 + 0.5 * torch.cos(2 * np.pi * xc / Lx)).view(-1, 1, 1)
    U = 1.0
    u, v, w = _zeros_vel(nx, ny, nz)
    u[:] = U
    var0 = scalar_stats(c, nx, ny, nz, dz_f)["var"]

    T = Lx / U
    dt = 0.2 * (Lx / nx) / U      # CFL ~ 0.2
    nsteps = int(round(T / dt))
    dt = T / nsteps               # land exactly on one period
    rhs_prev = None
    for _ in range(nsteps):
        c, rhs_prev = imex_scalar_step(c, u, v, w, nx, ny, nz, Lx/nx, Ly/ny,
                                       dz_c, dz_f, D=0.0, dt=dt, rhs_prev=rhs_prev)
    var1 = scalar_stats(c, nx, ny, nz, dz_f)["var"]
    cnum = c[1:nx+1, 1, 1]
    cexact = (0.5 + 0.5 * torch.cos(2 * np.pi * xc[1:nx+1] / Lx))
    err = float((cnum - cexact).abs().max())
    print(f"[4] pure advection one period: var ratio = {var1/var0:.4f}, "
          f"max|c-c0| = {err:.2e}")
    # central differencing has no numerical diffusion; AB2 adds only tiny dispersion
    assert abs(var1 / var0 - 1.0) < 0.02, var1 / var0
    assert err < 0.05, err
    return err


if __name__ == "__main__":
    fails = 0
    for fn in [test_constant_field, test_mean_conservation,
               test_diffusion_vs_erf, test_pure_advection]:
        try:
            fn()
        except AssertionError as e:
            print(f"    FAILED: {fn.__name__}: {e}")
            fails += 1
    print("\n" + ("ALL SCALAR TESTS PASSED" if fails == 0 else f"{fails} TEST(S) FAILED"))
    sys.exit(1 if fails else 0)
