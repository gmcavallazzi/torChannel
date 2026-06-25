"""Passive-scalar transport for the staggered-grid channel solver.

A passive scalar c (concentration in [0,1]) is carried at CELL CENTRES
(collocated with pressure), shape (nx+2, ny+2, nz+2) with ghost cells at index
0 and -1 in every direction. It obeys

    dc/dt + u.grad c = D laplacian(c),     D = nu / Sc,

advected by the (divergence-free) MAC velocity field. Spatial operators mirror
the momentum discretisation: conservative flux-form advection, 2nd-order central
diffusion, periodic in x,y and a configurable wall BC in z. The same IMEX split
as the momentum solver is used (advection + xy-diffusion explicit via AB2,
z-diffusion implicit via the theta-method), reusing the batched tridiagonal
solver.

Wall BC (z), set by `wall_bc`:
    'neumann'   -> zero scalar flux dc/dz = 0 (impermeable wall). Conserves the
                   mean <c>; the segregation variance decays to zero. This is the
                   right choice for a MIXING / DECAY study and is the default.
    'dirichlet' -> c = 0 at both walls (homogeneous; absorbing). Provided for a
                   future sustained-gradient setup.
"""

from __future__ import annotations

import numpy as np
import torch

from operators import solve_tridiagonal_batch


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------
@torch.jit.script
def apply_scalar_bc(c: torch.Tensor, wall_bc: str = 'neumann') -> None:
    """Fill ghost cells: periodic in x,y; Neumann or Dirichlet(0) walls in z."""
    # Periodic in x (cell-centred: ghosts at 0 and -1)
    c[0, :, :] = c[-2, :, :]
    c[-1, :, :] = c[1, :, :]
    # Periodic in y
    c[:, 0, :] = c[:, -2, :]
    c[:, -1, :] = c[:, 1, :]
    # Walls in z
    if wall_bc == 'dirichlet':
        c[:, :, 0] = -c[:, :, 1]     # c = 0 at wall
        c[:, :, -1] = -c[:, :, -2]
    else:  # neumann (no-flux)
        c[:, :, 0] = c[:, :, 1]      # dc/dz = 0 at wall
        c[:, :, -1] = c[:, :, -2]


# ---------------------------------------------------------------------------
# Spatial operators
# ---------------------------------------------------------------------------
@torch.jit.script
def advection_scalar(c: torch.Tensor, u: torch.Tensor, v: torch.Tensor,
                     w: torch.Tensor, nx: int, ny: int, nz: int,
                     dx: float, dy: float, dz_f: torch.Tensor) -> torch.Tensor:
    """Conservative flux-form advection div(u c) of a cell-centred scalar.

    Equals u.grad c when the velocity is divergence-free. Faces use the MAC
    velocities (u at x-faces, v at y-faces, w at z-faces); the scalar is linearly
    interpolated to each face. Requires ghost cells to be up to date.
    """
    adv = torch.zeros_like(c)
    ci = c[1:nx+1, 1:ny+1, 1:nz+1]

    # d(u c)/dx : u[i] is the right face of cell i, u[i-1] the left face
    Fxr = u[1:nx+1, 1:ny+1, 1:nz+1] * 0.5 * (ci + c[2:nx+2, 1:ny+1, 1:nz+1])
    Fxl = u[0:nx,   1:ny+1, 1:nz+1] * 0.5 * (c[0:nx, 1:ny+1, 1:nz+1] + ci)
    dudx = (Fxr - Fxl) / dx

    # d(v c)/dy
    Gyr = v[1:nx+1, 1:ny+1, 1:nz+1] * 0.5 * (ci + c[1:nx+1, 2:ny+2, 1:nz+1])
    Gyl = v[1:nx+1, 0:ny,   1:nz+1] * 0.5 * (c[1:nx+1, 0:ny, 1:nz+1] + ci)
    dvdy = (Gyr - Gyl) / dy

    # d(w c)/dz  (w=0 at walls => no advective flux through the wall)
    Hzr = w[1:nx+1, 1:ny+1, 1:nz+1] * 0.5 * (ci + c[1:nx+1, 1:ny+1, 2:nz+2])
    Hzl = w[1:nx+1, 1:ny+1, 0:nz]   * 0.5 * (c[1:nx+1, 1:ny+1, 0:nz] + ci)
    dwdz = (Hzr - Hzl) / dz_f.view(1, 1, -1)

    adv[1:nx+1, 1:ny+1, 1:nz+1] = dudx + dvdy + dwdz
    return adv


@torch.jit.script
def diffusion_xy_scalar(c: torch.Tensor, nx: int, ny: int, nz: int,
                        dx: float, dy: float, D: float) -> torch.Tensor:
    """Explicit in-plane diffusion D*(d2c/dx2 + d2c/dy2). Periodic via ghosts."""
    out = torch.zeros_like(c)
    d2x = (c[2:nx+2, 1:ny+1, 1:nz+1] - 2 * c[1:nx+1, 1:ny+1, 1:nz+1]
           + c[0:nx, 1:ny+1, 1:nz+1]) / dx**2
    d2y = (c[1:nx+1, 2:ny+2, 1:nz+1] - 2 * c[1:nx+1, 1:ny+1, 1:nz+1]
           + c[1:nx+1, 0:ny, 1:nz+1]) / dy**2
    out[1:nx+1, 1:ny+1, 1:nz+1] = D * (d2x + d2y)
    return out


@torch.jit.script
def solve_implicit_diffusion_scalar(c: torch.Tensor, dt: float,
                                    nx: int, ny: int, nz: int,
                                    dz_c: torch.Tensor, dz_f: torch.Tensor,
                                    D: float, theta: float = 0.5,
                                    wall_bc: str = 'neumann') -> torch.Tensor:
    """Implicit z-diffusion (theta-method) for the scalar, both walls `wall_bc`.

    Solves (I - theta*dt*D d2/dz2) c^{n+1} = c^* + (1-theta)*dt*D d2c^*/dz2 with
    a batched tridiagonal solve over all (i,j) columns. theta=0.5 is Crank-Nicolson.
    """
    c_new = c.clone()
    alpha = theta * dt * D

    dz_left = dz_c[0:nz]
    dz_right = dz_c[1:nz+1]
    dz_cell = dz_f[0:nz]

    coeff_lower = -alpha / (dz_left * dz_cell)
    coeff_center = 1.0 + alpha * (1.0 / dz_left + 1.0 / dz_right) / dz_cell
    coeff_upper = -alpha / (dz_right * dz_cell)

    a = coeff_lower.clone()
    b = coeff_center.clone()
    cc = coeff_upper.clone()

    # Bottom wall (k=0): eliminate ghost c[0]
    a[0] = 0.0
    if wall_bc == 'dirichlet':
        b[0] = coeff_center[0] - coeff_lower[0]   # ghost = -c[1]
    else:
        b[0] = coeff_center[0] + coeff_lower[0]   # ghost =  c[1] (no-flux)

    # Top wall (k=nz-1): eliminate ghost c[nz+1]
    if wall_bc == 'dirichlet':
        b[nz-1] = coeff_center[nz-1] - coeff_upper[nz-1]
    else:
        b[nz-1] = coeff_center[nz-1] + coeff_upper[nz-1]
    cc[nz-1] = 0.0

    d = c[1:nx+1, 1:ny+1, 1:nz+1].clone()

    if theta < 1.0:
        beta = (1.0 - theta) * dt * D
        C = c[1:nx+1, 1:ny+1, :]  # (nx, ny, nz+2) with ghosts
        d2 = ((C[:, :, 2:nz+2] - C[:, :, 1:nz+1]) / dz_right
              - (C[:, :, 1:nz+1] - C[:, :, 0:nz]) / dz_left) / dz_cell
        # exact wall closures (independent of ghost values)
        if wall_bc == 'dirichlet':
            d2[:, :, 0] = ((C[:, :, 2] - C[:, :, 1]) / dz_right[0]
                           - 2.0 * C[:, :, 1] / dz_left[0]) / dz_cell[0]
            d2[:, :, nz-1] = (-2.0 * C[:, :, nz] / dz_right[nz-1]
                              - (C[:, :, nz] - C[:, :, nz-1]) / dz_left[nz-1]) / dz_cell[nz-1]
        else:
            d2[:, :, 0] = ((C[:, :, 2] - C[:, :, 1]) / dz_right[0]) / dz_cell[0]
            d2[:, :, nz-1] = (-(C[:, :, nz] - C[:, :, nz-1]) / dz_left[nz-1]) / dz_cell[nz-1]
        d = d + beta * d2

    d_batch = d.reshape(nx * ny, nz)
    x_batch = solve_tridiagonal_batch(a, b, cc, d_batch)
    c_new[1:nx+1, 1:ny+1, 1:nz+1] = x_batch.reshape(nx, ny, nz)
    return c_new


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------
def initialize_scalar(nx: int, ny: int, nz: int, z_c: torch.Tensor,
                      Lx: float, Ly: float, Lz: float, init_type: str = 'interface_z',
                      interface_pos: float = 0.5, eps_cells: float = 1.0,
                      device: str = 'cpu') -> torch.Tensor:
    """Create the initial scalar field (nx+2, ny+2, nz+2).

    init_types:
        'interface_z' : two fluids split across the channel at z = interface_pos*Lz,
                        tanh-smoothed. Reduces to pure z-diffusion (erf) when u=0.
        'interface_y' : split across the span at y = interface_pos*Ly (two
                        co-flowing streams meeting on the centreline).
        'uniform'     : c = interface_pos everywhere (constant-field test).
    """
    c = torch.zeros(nx + 2, ny + 2, nz + 2, device=device, dtype=torch.float64)

    if init_type == 'uniform':
        c[:] = interface_pos
        return c

    if init_type == 'interface_z':
        zc = z_c.to(device)                      # (nz+2,) cell centres incl. ghosts
        eps = eps_cells * (Lz / nz)
        prof = 0.5 * (1.0 + torch.tanh((zc - interface_pos * Lz) / eps))  # (nz+2,)
        c[:, :, :] = prof.view(1, 1, -1)
    elif init_type == 'interface_y':
        dy = Ly / ny
        yc = (torch.arange(ny + 2, device=device, dtype=torch.float64) - 0.5) * dy
        eps = eps_cells * dy
        prof = 0.5 * (1.0 + torch.tanh((yc - interface_pos * Ly) / eps))  # (ny+2,)
        c[:, :, :] = prof.view(1, -1, 1)
    else:
        raise ValueError(f"unknown scalar init_type {init_type!r}")
    return c


# ---------------------------------------------------------------------------
# Diagnostics & I/O
# ---------------------------------------------------------------------------
def scalar_stats(c: torch.Tensor, nx: int, ny: int, nz: int,
                 dz_f: torch.Tensor) -> dict:
    """Volume-weighted mean and variance (intensity of segregation) of c.

    Weights cells by their (stretched) z-height so the measure is the true volume
    average. Returns mean, var and the normalised mixedness M = std/std_max with
    std_max = sqrt(mean*(1-mean)).
    """
    ci = c[1:nx+1, 1:ny+1, 1:nz+1]
    wz = dz_f[0:nz].view(1, 1, -1)
    vol = wz.expand_as(ci)
    total = vol.sum()
    mean = (ci * vol).sum() / total
    var = (((ci - mean) ** 2) * vol).sum() / total
    std = torch.sqrt(torch.clamp(var, min=0.0))
    std_max = torch.sqrt(torch.clamp(mean * (1.0 - mean), min=1e-30))
    return {"mean": float(mean), "var": float(var), "std": float(std),
            "M": float(std / std_max)}


def save_scalar_field(c: torch.Tensor, results_folder: str, filename: str,
                      step: int, time: float, Sc: float) -> None:
    import os
    path = os.path.join(results_folder, filename)
    np.savez(path, c=c.detach().cpu().numpy(), step=step, time=time, Sc=Sc)


def load_scalar_field(path: str, device: str = 'cpu') -> torch.Tensor:
    data = np.load(path)
    return torch.tensor(data['c'], device=device, dtype=torch.float64)
