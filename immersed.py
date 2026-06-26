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


def _koch_zigzag_disp(N, r, y_norm):
    """Generation-N area-balanced zigzag displacement (same generator as the scalar
    Koch IC, D_f = log4/log r) sampled at base coordinate y_norm in [0,1]. N=0 -> 0.
    Used to give the herringbone ridge a self-similar, multi-scale (fractal) spanwise
    corrugation: higher N adds finer wall scales at ~constant envelope amplitude."""
    if N <= 0:
        return np.zeros_like(y_norm)
    a = np.sqrt((1.0 / r) ** 2 - 0.25 ** 2)
    motif = np.array([[0., 0.], [0.25, a], [0.5, 0.], [0.75, -a], [1., 0.]])
    pts = np.array([[0., 0.], [1., 0.]])
    mx, my = motif[:, 0], motif[:, 1]
    for _ in range(int(N)):
        p0, p1 = pts[:-1], pts[1:]
        d = p1 - p0
        perp = np.stack([-d[:, 1], d[:, 0]], axis=1)
        new = (p0[:, None, :] + mx[None, :, None] * d[:, None, :]
               + my[None, :, None] * perp[:, None, :])
        pts = np.concatenate([new[:, :-1, :].reshape(-1, 2), pts[-1:]], 0)
    s, disp = pts[:, 0], pts[:, 1]
    return np.interp(y_norm, s, disp)


def height_field(x, y, kind='slab', z1=0.2, h0=0.2, A=0.1, kx=1.0, ky=1.0,
                 Lx=1.0, Ly=1.0, n_waves_x=2, n_waves_y=1,
                 apex_frac=0.5, stagger=True, N=0, r=3.0, koch_amp=1.0, **kwargs):
    """Solid-region height h(x, y); a point is solid where its z < h(x, y).

    x, y are broadcast together (e.g. meshgrid). For 'grooves', kx, ky are
    angular wavenumbers (caller converts n_waves -> 2*pi*n/L).

    'herringbone' builds a STAGGERED chevron-groove floor (the SHM geometry that
    drives chaotic advection): within each streamwise cycle the groove crests bend
    into a V at y = apex_frac*Ly; the apex side flips every cycle (stagger), so the
    two counter-rotating helical rolls alternate along x — the exponential
    stretch/fold a single oblique groove cannot give.
    """
    if kind == 'slab':
        return torch.full_like(x + y, float(z1))
    if kind == 'grooves':
        return h0 + A * torch.sin(kx * x + ky * y)
    if kind == 'herringbone':
        period_x = Lx / n_waves_x
        cyc = torch.floor(x / period_x)
        xloc = x / period_x - cyc                       # in [0,1) within a cycle
        if stagger:
            apex = torch.where((cyc.long() % 2) == 0,
                               torch.full_like(x, apex_frac),
                               torch.full_like(x, 1.0 - apex_frac))
        else:
            apex = torch.full_like(x, apex_frac)
        ya = apex * Ly
        # triangle wave in y peaking (=1) at the apex, 0 at y=0 and y=Ly (so the
        # field stays periodic in y); this bends the streamwise ridges into chevrons.
        tri = torch.where(y < ya, y / ya.clamp(min=1e-9),
                          (Ly - y) / (Ly - ya).clamp(min=1e-9))
        phase = 2.0 * np.pi * xloc - np.pi * n_waves_y * tri
        return h0 + A * torch.cos(phase)
    if kind == 'koch_herringbone':
        # staggered herringbone whose spanwise ridge profile is a generation-N
        # area-balanced Koch zigzag -> a "fractal inlet surface" proxy: the
        # corrugated wall carries multi-scale structure (more wall scales with N),
        # driving a multi-scale near-wall secondary flow. N=0 == smooth herringbone.
        period_x = Lx / n_waves_x
        cyc = torch.floor(x / period_x)
        xloc = x / period_x - cyc
        if stagger:
            apex = torch.where((cyc.long() % 2) == 0,
                               torch.full_like(x, apex_frac),
                               torch.full_like(x, 1.0 - apex_frac))
        else:
            apex = torch.full_like(x, apex_frac)
        ya = apex * Ly
        tri = torch.where(y < ya, y / ya.clamp(min=1e-9),
                          (Ly - y) / (Ly - ya).clamp(min=1e-9))
        # generation-N Koch corrugation of the ridge (computed once on the y grid)
        yn = (y / Ly).detach().cpu().numpy()
        disp = _koch_zigzag_disp(N, r, yn.ravel()).reshape(yn.shape)
        delta = torch.as_tensor(disp, device=x.device, dtype=x.dtype) * float(koch_amp)
        phase = 2.0 * np.pi * xloc - np.pi * n_waves_y * (tri + delta)
        return h0 + A * torch.cos(phase)
    raise ValueError(f"unknown immersed height kind {kind!r}")


def build_masks(nx, ny, nz, Lx, Ly, Lz, z_c, z_f, device='cpu', **hf):
    """Build solid-indicator masks on the four grid locations.

    Returns a dict with chi_u (nx+1,ny+2,nz+2), chi_v (nx+2,ny+1,nz+2),
    chi_w (nx+2,ny+2,nz+1) and chi_c (nx+2,ny+2,nz+2), each 1.0 in solid and
    0.0 in fluid (float64). `hf` is forwarded to `height_field` (kind, z1, ...).
    """
    xc, xf, yc, yf, zc, zf = _coords(nx, ny, nz, Lx, Ly, z_c, z_f, device)

    if hf.get('kind', 'slab') == 'pipe':
        # Circular cross-section (pipe): fill the corners of the rectangular box with
        # solid so the fluid is the inscribed disc. Solid where the (y,z) point lies
        # OUTSIDE a circle of radius R centred at (cy, cz). Uniform in x.
        R = float(hf.get('pipe_R', 0.5 * min(Ly, Lz)))
        cy = float(hf.get('pipe_yc', 0.5 * Ly))
        cz = float(hf.get('pipe_zc', 0.5 * Lz))

        def mask(xv, yv, zv):
            Y, Z = torch.meshgrid(yv, zv, indexing='ij')        # (Ny, Nz)
            solid = ((Y - cy) ** 2 + (Z - cz) ** 2) > R ** 2    # (Ny, Nz)
            return solid.to(torch.float64).unsqueeze(0).expand(len(xv), -1, -1).clone()
    elif hf.get('kind', 'slab') == 'pipe_koch':
        # Circular cross-section whose WALL carries a generation-N, area-balanced Koch
        # corrugation in the AZIMUTHAL direction -- the faithful "fractal inlet surface"
        # on a round orifice (no obstacles; the wall itself is folded). The corrugation
        # is localised at the INLET via a streamwise envelope env(x): full at x=0,
        # smoothly to zero by x = inlet_len; downstream the pipe is the smooth disc.
        #   R_wall(theta, x) = R + amp * env(x) * d_hat(theta)
        # d_hat is the Koch zigzag wrapped n_lobes times around the circle, normalised
        # to unit peak so the radial amplitude (= koch_amp*R) is the SAME for every N
        # (constant envelope, more scales with N) and zero-mean in theta (area-balanced
        # to first order). N=0 -> d_hat=0 -> smooth inscribed disc (the baseline).
        R = float(hf.get('pipe_R', 0.5 * min(Ly, Lz)))
        cy = float(hf.get('pipe_yc', 0.5 * Ly))
        cz = float(hf.get('pipe_zc', 0.5 * Lz))
        N = int(hf.get('N', 0)); r = float(hf.get('r', 3.0))
        amp = float(hf.get('koch_amp', 0.1)) * R
        n_lobes = int(hf.get('n_lobes', 1))
        inlet_len = float(hf.get('inlet_len', 0.1 * Lx))

        def mask(xv, yv, zv):
            Y, Z = torch.meshgrid(yv, zv, indexing='ij')        # (Ny, Nz)
            dY, dZ = Y - cy, Z - cz
            rho = torch.sqrt(dY ** 2 + dZ ** 2)                 # (Ny, Nz)
            theta = torch.atan2(dZ, dY)                         # (-pi, pi]
            tn = ((theta + np.pi) / (2.0 * np.pi) * n_lobes) % 1.0
            d = _koch_zigzag_disp(N, r, tn.detach().cpu().numpy().ravel()).reshape(tn.shape)
            m = float(np.max(np.abs(d)))
            d_hat = torch.as_tensor(d / m if m > 0 else d, device=xv.device, dtype=torch.float64)
            # streamwise inlet envelope: half-cosine bump, 1 at x=0 -> 0 at inlet_len
            env = torch.where(xv < inlet_len,
                              0.5 * (1.0 + torch.cos(np.pi * xv.clamp(max=inlet_len) / max(inlet_len, 1e-12))),
                              torch.zeros_like(xv))              # (Nx,)
            Rwall = R + amp * env.view(-1, 1, 1) * d_hat.unsqueeze(0)   # (Nx, Ny, Nz)
            solid = rho.unsqueeze(0) > Rwall                    # (Nx, Ny, Nz)
            return solid.to(torch.float64)
    else:
        def mask(xv, yv, zv):
            # h on the (x, y) plane of this location, then compare to z (broadcast).
            X, Y = torch.meshgrid(xv, yv, indexing='ij')          # (Nx, Ny)
            H = height_field(X, Y, Lx=Lx, Ly=Ly, **hf)            # (Nx, Ny)
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
