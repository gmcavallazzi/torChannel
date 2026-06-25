"""Volume-penalization immersed boundary for the staggered-grid channel solver.

Brinkman (volume) penalization adds a body force  f = -(chi/eta) * (u - u_solid)
to the momentum equation, where chi is the indicator of the solid region (1 in
solid, 0 in fluid), eta is a small penalization parameter (eta -> 0 => rigid),
and u_solid = 0 for a stationary wall. This drives the velocity to zero inside
the solid while leaving the fluid equations untouched, so the FFT-based pressure
projection (which requires periodicity in x,y) is PRESERVED: the penalization is
just an extra force added before the projection step.

Treatment is IMPLICIT and POINTWISE. The penalty operator is diagonal, so the
backward-Euler update of du/dt = ... - (chi/eta) u is a pure rescale

    u** = u* / (1 + dt*chi/eta),

with no linear solve and unconditional stability for any eta > 0. The projection
that follows reintroduces an O(sqrt(eta)) slip velocity in the solid; this is the
standard, accepted penalization-IB error (Angot, Bruneau & Fabrie 1999).

Masks live on the STAGGERED locations of each velocity component (u at x-faces,
v at y-faces, w at z-faces) plus the cell centres (for the scalar diagnostic).
A solid region is described by a height field h(x, y): a cell/face is solid where
its z-coordinate lies below h at that (x, y). h is built by `height_field`:
    'slab'    : h = z1 everywhere (flat solid slab; Phase-0 validation geometry)
    'grooves' : h = h0 + A*sin(kx*x + ky*y)  (oblique grooves; Phase-1 mechanism)
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Staggered coordinate vectors (including ghost cells)
# ---------------------------------------------------------------------------
def _coords(nx, ny, nz, Lx, Ly, z_c, z_f, device):
    """Return coordinate vectors for cell centres and faces (with ghosts).

    x,y are uniform; z is the (possibly stretched) grid passed in.
      xc (nx+2): cell-centre x, xc[i] = (i-0.5)*dx          i = 0..nx+1
      xf (nx+1): u-face x,      xf[i] = i*dx                i = 0..nx
      yc (ny+2): cell-centre y, yc[j] = (j-0.5)*dy
      yf (ny+1): v-face y,      yf[j] = j*dy
      zc (nz+2): cell-centre z  (= z_c, ghosts included)
      zf (nz+1): w-face z       (= z_f)
    """
    dx, dy = Lx / nx, Ly / ny
    xc = (torch.arange(nx + 2, device=device, dtype=torch.float64) - 0.5) * dx
    xf = torch.arange(nx + 1, device=device, dtype=torch.float64) * dx
    yc = (torch.arange(ny + 2, device=device, dtype=torch.float64) - 0.5) * dy
    yf = torch.arange(ny + 1, device=device, dtype=torch.float64) * dy
    zc = z_c.to(device)
    zf = z_f.to(device)
    return xc, xf, yc, yf, zc, zf


def height_field(x, y, kind='slab', z1=0.2, h0=0.2, A=0.1, kx=1.0, ky=1.0):
    """Solid-region height h(x, y); a point is solid where its z < h(x, y).

    x, y are broadcast together (e.g. meshgrid). kx, ky are wavenumbers in
    rad / length already (caller converts n_waves -> 2*pi*n/L).
    """
    if kind == 'slab':
        return torch.full_like(x + y, float(z1))
    if kind == 'grooves':
        return h0 + A * torch.sin(kx * x + ky * y)
    raise ValueError(f"unknown immersed height kind {kind!r}")


def build_masks(nx, ny, nz, Lx, Ly, Lz, z_c, z_f, device='cpu', **hf):
    """Build solid-indicator masks on the four grid locations.

    Returns a dict with chi_u (nx+1,ny+2,nz+2), chi_v (nx+2,ny+1,nz+2),
    chi_w (nx+2,ny+2,nz+1) and chi_c (nx+2,ny+2,nz+2), each 1.0 in solid and
    0.0 in fluid (float64). `hf` is forwarded to `height_field` (kind, z1, ...).
    """
    xc, xf, yc, yf, zc, zf = _coords(nx, ny, nz, Lx, Ly, z_c, z_f, device)

    def mask(xv, yv, zv):
        # h on the (x, y) plane of this location, then compare to z (broadcast).
        X, Y = torch.meshgrid(xv, yv, indexing='ij')          # (Nx, Ny)
        H = height_field(X, Y, **hf)                           # (Nx, Ny)
        return (zv.view(1, 1, -1) < H.unsqueeze(-1)).to(torch.float64)

    return {
        'chi_u': mask(xf, yc, zc),
        'chi_v': mask(xc, yf, zc),
        'chi_w': mask(xc, yc, zf),
        'chi_c': mask(xc, yc, zc),
    }


# ---------------------------------------------------------------------------
# Implicit penalization (pointwise backward-Euler rescale)
# ---------------------------------------------------------------------------
def penalize(field: torch.Tensor, chi: torch.Tensor, dt: float, eta: float) -> torch.Tensor:
    """Apply one implicit penalization step:  field <- field / (1 + dt*chi/eta).

    Diagonal/pointwise, so this is the exact backward-Euler solve of the penalty
    term. No-op where chi == 0 (fluid). Returns a new tensor.
    """
    return field / (1.0 + (dt / eta) * chi)


# ---------------------------------------------------------------------------
# Fluid-weighted volumes (so bulk forcing / mixing stats ignore the solid)
# ---------------------------------------------------------------------------
def fluid_cell_volume(cell_vol: torch.Tensor, chi_u: torch.Tensor,
                      nx: int, ny: int, nz: int):
    """Fluid-only cell-volume weights and total, on the u-control volumes.

    The bulk forcing acts on u[1:nx+1,...]; weighting those cells by their fluid
    fraction (1 - chi_u) makes the PI controller target the interstitial (fluid)
    mean velocity instead of the superficial (whole-box) one. Returns
    (fluid_vol (nx,ny,nz), total_fluid_volume float).
    """
    fluid = (1.0 - chi_u[1:nx + 1, 1:ny + 1, 1:nz + 1])
    fluid_vol = cell_vol * fluid
    return fluid_vol, float(fluid_vol.sum())


def solid_fraction(chi_c: torch.Tensor, nx: int, ny: int, nz: int,
                   dz_f: torch.Tensor) -> float:
    """Volume fraction of the box occupied by solid (cell-centre mask)."""
    ci = chi_c[1:nx + 1, 1:ny + 1, 1:nz + 1]
    wz = dz_f[0:nz].view(1, 1, -1).expand_as(ci)
    return float((ci * wz).sum() / wz.sum())
