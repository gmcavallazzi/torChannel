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
def apply_scalar_bc(c: torch.Tensor, wall_bc: str = 'neumann', bc_y: str = 'periodic',
                    bc_x: str = 'periodic') -> None:
    """Fill ghost cells: periodic in x, or inflow/outflow ('inout', where the inlet
    ghost c[0] is the prescribed inlet profile set separately and the outlet is
    zero-gradient); periodic or no-flux walls in y (duct); Neumann/Dirichlet walls in z."""
    # x: periodic, or inflow/outflow (inlet ghost c[0] set separately; outlet zero-grad)
    if bc_x == 'periodic':
        c[0, :, :] = c[-2, :, :]
        c[-1, :, :] = c[1, :, :]
    else:
        c[-1, :, :] = c[-2, :, :]    # outflow: dc/dx = 0
    # y: periodic, or no-flux walls (duct)
    if bc_y == 'wall':
        c[:, 0, :] = c[:, 1, :]      # dc/dy = 0 at wall (no scalar flux through wall)
        c[:, -1, :] = c[:, -2, :]
    else:
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
def _vanleer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """van Leer limited slope: 2ab/(a+b) when a,b have the same sign, else 0
    (div-by-zero safe). Returns the limited undivided difference at a cell."""
    ab = a * b
    return torch.where(ab > 0.0, 2.0 * ab / (a + b + 1e-30), torch.zeros_like(ab))


@torch.jit.script
def advection_scalar_tvd(c: torch.Tensor, u: torch.Tensor, v: torch.Tensor,
                         w: torch.Tensor, nx: int, ny: int, nz: int,
                         dx: float, dy: float, dz_f: torch.Tensor,
                         bc_y: str = 'periodic', wall_bc: str = 'neumann',
                         bc_x: str = 'periodic') -> torch.Tensor:
    """Conservative flux-form advection div(u c) with a van Leer flux limiter.

    Same conservative discretisation as `advection_scalar`, but each face value is a
    MUSCL upwind reconstruction with the van Leer limiter instead of the central
    average, so the scheme is TVD/monotone — no dispersive over/undershoot at the
    high cell-Peclet of high-Schmidt runs. Requires up-to-date 1-cell ghosts (call
    after apply_scalar_bc); the 2nd ghost layer each reconstruction needs is built
    here per direction from the BCs (periodic wrap, no-flux reflection, or
    Dirichlet negation). Face velocity is exactly 0 at every wall, so the limiter
    never affects a wall flux.
    """
    adv = torch.zeros_like(c)

    # ---------- x: periodic, or inflow/outflow 2nd ghosts ----------
    if bc_x == 'periodic':
        cx = torch.cat([c[nx-1:nx, :, :], c, c[2:3, :, :]], dim=0)   # logical cells -1..nx+2
    else:
        # inlet: constant upstream state (= inlet ghost c[0]); outlet: zero-gradient
        cx = torch.cat([c[0:1, :, :], c, c[nx+1:nx+2, :, :]], dim=0)
    sf   = _vanleer(cx[1:nx+2] - cx[0:nx+1], cx[2:nx+3] - cx[1:nx+2])   # slope at cell f
    sfp1 = _vanleer(cx[2:nx+3] - cx[1:nx+2], cx[3:nx+4] - cx[2:nx+3])   # slope at cell f+1
    Uf = u[0:nx+1, :, :]
    cfx = torch.where(Uf >= 0.0, cx[1:nx+2] + 0.5 * sf, cx[2:nx+3] - 0.5 * sfp1)
    Fx = Uf * cfx
    dudx = (Fx[1:nx+1, 1:ny+1, 1:nz+1] - Fx[0:nx, 1:ny+1, 1:nz+1]) / dx

    # ---------- y: periodic or no-slip walls (duct) ----------
    if bc_y == 'wall':
        cyL = c[:, 2:3, :]            # no-flux reflection 2nd ghost
        cyR = c[:, ny-1:ny, :]
    else:
        cyL = c[:, ny-1:ny, :]       # periodic wrap
        cyR = c[:, 2:3, :]
    cy = torch.cat([cyL, c, cyR], dim=1)
    sg   = _vanleer(cy[:, 1:ny+2, :] - cy[:, 0:ny+1, :], cy[:, 2:ny+3, :] - cy[:, 1:ny+2, :])
    sgp1 = _vanleer(cy[:, 2:ny+3, :] - cy[:, 1:ny+2, :], cy[:, 3:ny+4, :] - cy[:, 2:ny+3, :])
    Vf = v[:, 0:ny+1, :]
    cfy = torch.where(Vf >= 0.0, cy[:, 1:ny+2, :] + 0.5 * sg, cy[:, 2:ny+3, :] - 0.5 * sgp1)
    Fy = Vf * cfy
    dvdy = (Fy[1:nx+1, 1:ny+1, 1:nz+1] - Fy[1:nx+1, 0:ny, 1:nz+1]) / dy

    # ---------- z: walls (Neumann no-flux, or Dirichlet c=0) ----------
    if wall_bc == 'dirichlet':
        czL = -c[:, :, 2:3]
        czR = -c[:, :, nz-1:nz]
    else:
        czL = c[:, :, 2:3]
        czR = c[:, :, nz-1:nz]
    cz = torch.cat([czL, c, czR], dim=2)
    sh   = _vanleer(cz[:, :, 1:nz+2] - cz[:, :, 0:nz+1], cz[:, :, 2:nz+3] - cz[:, :, 1:nz+2])
    shp1 = _vanleer(cz[:, :, 2:nz+3] - cz[:, :, 1:nz+2], cz[:, :, 3:nz+4] - cz[:, :, 2:nz+3])
    Wf = w[:, :, 0:nz+1]
    cfz = torch.where(Wf >= 0.0, cz[:, :, 1:nz+2] + 0.5 * sh, cz[:, :, 2:nz+3] - 0.5 * shp1)
    Fz = Wf * cfz
    dwdz = (Fz[1:nx+1, 1:ny+1, 1:nz+1] - Fz[1:nx+1, 1:ny+1, 0:nz]) / dz_f.view(1, 1, -1)

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
def diffusion_z_scalar(c: torch.Tensor, nx: int, ny: int, nz: int,
                       dz_c: torch.Tensor, dz_f: torch.Tensor, D: float) -> torch.Tensor:
    """Explicit wall-normal diffusion D*d2c/dz2 on the stretched z-grid, using the z
    ghost cells (filled by apply_scalar_bc, which encodes the no-flux/Dirichlet wall BC).
    Companion to diffusion_xy_scalar for the fully-explicit SSP-RK3 scalar march
    (frozen velocity, high Sc) where z-diffusion is cheap and need not be implicit."""
    out = torch.zeros_like(c)
    C = c[1:nx+1, 1:ny+1, :]                       # (nx, ny, nz+2), ghosts at 0 and nz+1
    dz_left = dz_c[0:nz].view(1, 1, nz)
    dz_right = dz_c[1:nz+1].view(1, 1, nz)
    dz_cell = dz_f[0:nz].view(1, 1, nz)
    d2z = ((C[:, :, 2:nz+2] - C[:, :, 1:nz+1]) / dz_right
           - (C[:, :, 1:nz+1] - C[:, :, 0:nz]) / dz_left) / dz_cell
    out[1:nx+1, 1:ny+1, 1:nz+1] = D * d2z
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
# Koch fractal interface (cross-section)
# ---------------------------------------------------------------------------
def _zigzag_motif(r: float) -> np.ndarray:
    """Area-balanced 4-segment Koch generator, each segment of length 1/r.

    D_f = log(4)/log(r). Valid for 1 < r < 4. Every outward triangle is matched
    by an equal inward indentation, so the interface keeps the 50/50 split.
    """
    if not (1.0 < r < 4.0):
        raise ValueError(f"zigzag motif needs 1 < r < 4, got r={r}")
    a = np.sqrt((1.0 / r) ** 2 - 0.25 ** 2)
    return np.array([[0.0, 0.0], [0.25, a], [0.5, 0.0], [0.75, -a], [1.0, 0.0]])


def _koch_polyline(motif: np.ndarray, N: int,
                   base=((0.0, 0.0), (1.0, 0.0))) -> np.ndarray:
    """Generation-N Koch polyline by affine edge replacement (vectorised)."""
    pts = np.array(base, dtype=float)
    if N <= 0:
        return pts
    mx, my = motif[:, 0], motif[:, 1]
    for _ in range(N):
        p0, p1 = pts[:-1], pts[1:]
        d = p1 - p0
        perp = np.stack([-d[:, 1], d[:, 0]], axis=1)
        new = (p0[:, None, :] + mx[None, :, None] * d[:, None, :]
               + my[None, :, None] * perp[:, None, :])  # (S, K, 2)
        pts = np.concatenate([new[:, :-1, :].reshape(-1, 2), pts[-1:][None, 0]], 0)
    return pts


def _signed_distance(points: np.ndarray, polyline: np.ndarray,
                     chunk: int = 8192) -> np.ndarray:
    """Signed distance from each point to an open polyline (sign = nearest side)."""
    A, B = polyline[:-1], polyline[1:]
    AB = B - A
    L2 = np.einsum("sk,sk->s", AB, AB)
    L2 = np.where(L2 == 0.0, 1.0, L2)
    out = np.empty(points.shape[0])
    for lo in range(0, points.shape[0], chunk):
        G = points[lo:lo + chunk]
        AP = G[:, None, :] - A[None, :, :]
        t = np.clip(np.einsum("csk,sk->cs", AP, AB) / L2, 0.0, 1.0)
        closest = A[None, :, :] + t[:, :, None] * AB[None, :, :]
        diff = G[:, None, :] - closest
        dist = np.sqrt(np.einsum("csk,csk->cs", diff, diff))
        cross = AB[None, :, 0] * AP[:, :, 1] - AB[None, :, 1] * AP[:, :, 0]
        j = np.argmin(dist, axis=1)
        rows = np.arange(G.shape[0])
        sign = np.sign(cross[rows, j])
        sign = np.where(sign == 0.0, 1.0, sign)
        out[lo:lo + chunk] = dist[rows, j] * sign
    return out


def _point_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorised even-odd ray-casting test; True for points inside the closed polygon.

    Robust regardless of fold density (winding is well-defined for a CLOSED loop),
    unlike a nearest-segment side test on an OPEN polyline, whose sign is ambiguous
    near the curve's endpoints and flips at high generation N."""
    px, py = points[:, 0], points[:, 1]
    vx0, vy0 = polygon[:-1, 0], polygon[:-1, 1]
    vx1, vy1 = polygon[1:, 0], polygon[1:, 1]
    straddle = (vy0[None, :] > py[:, None]) != (vy1[None, :] > py[:, None])
    denom = np.where((vy1 - vy0) == 0.0, 1.0, vy1 - vy0)
    xint = (vx1 - vx0)[None, :] * (py[:, None] - vy0[None, :]) / denom[None, :] + vx0[None, :]
    crossings = (straddle & (px[:, None] < xint)).sum(axis=1)
    return (crossings % 2) == 1


def koch_interface_yz(y_c: np.ndarray, z_c: np.ndarray, Ly: float, Lz: float,
                      N: int, r: float, eps: float) -> np.ndarray:
    """Rasterise a Koch interface in the (z, y) cross-section.

    The base interface runs wall-to-wall in z at y = Ly/2 and is Koch-folded in y
    (generation N). Returns c(y, z) of shape (len(y_c), len(z_c)) via a smoothed
    signed distance, c = 1/2 (1 + tanh(phi/eps)).

    The interface DISTANCE comes from the open Koch polyline, but its SIGN (which
    stream a cell belongs to) is taken from a point-in-polygon test on the interface
    closed along the top wall. This is robust at high N; the older nearest-segment
    side test produced wrong-sign corner patches for N>=3 (open-curve sign ambiguity).

    NOTE: this single-interface field is NOT periodic in y — c jumps 0->1 across the
    y = 0/Ly seam, which adds a spurious flat interface in a periodic box. Use
    `koch_strip_yz` for a seam-free initial condition.
    """
    motif = _zigzag_motif(r)
    # base spans z in [0, Lz] at y = Ly/2; coordinates ordered (z, y)
    poly = _koch_polyline(motif, N, base=((0.0, Ly / 2.0), (Lz, Ly / 2.0)))
    ZZ, YY = np.meshgrid(z_c, y_c, indexing="xy")   # YY,ZZ shape (len(y), len(z))
    pts = np.stack([ZZ.ravel(), YY.ravel()], axis=1)
    dist = np.abs(_signed_distance(pts, poly))              # distance magnitude only
    # close the interface into a polygon enclosing the UPPER (y > interface) stream,
    # so c = 1 there matches the legacy N=0 convention
    closed = np.vstack([poly, [Lz, Ly], [0.0, Ly]])
    inside = _point_in_polygon(pts, closed)
    phi = np.where(inside, dist, -dist).reshape(len(y_c), len(z_c))
    return 0.5 * (1.0 + np.tanh(phi / eps))


def koch_strip_yz(y_c: np.ndarray, z_c: np.ndarray, Ly: float, Lz: float,
                  N: int, r: float, eps: float, width_frac: float = 0.5) -> np.ndarray:
    """Seam-free Koch initial condition: a c=1 strip with TWO Koch-folded edges.

    A band of width `width_frac*Ly` centred at y = Ly/2 is filled with c=1; both of
    its edges (at y = Ly/2 -/+ width) run wall-to-wall in z and are Koch-folded in y
    (generation N). c -> 0 at y = 0 and y = Ly, so the field is PERIODIC in y with no
    spurious seam interface, while still carrying the fractal interfacial area (now on
    two interfaces). Mean(c) ~ width_frac. Returns c(y, z), shape (len(y_c), len(z_c)).
    """
    motif = _zigzag_motif(r)
    half = 0.5 * width_frac * Ly
    poly_lo = _koch_polyline(motif, N, base=((0.0, Ly / 2.0 - half), (Lz, Ly / 2.0 - half)))
    poly_hi = _koch_polyline(motif, N, base=((0.0, Ly / 2.0 + half), (Lz, Ly / 2.0 + half)))
    ZZ, YY = np.meshgrid(z_c, y_c, indexing="xy")
    pts = np.stack([ZZ.ravel(), YY.ravel()], axis=1)
    phi_lo = _signed_distance(pts, poly_lo).reshape(len(y_c), len(z_c))  # >0 above lower edge
    phi_hi = _signed_distance(pts, poly_hi).reshape(len(y_c), len(z_c))  # >0 above upper edge
    # inside the strip: above the lower edge AND below the upper edge
    return 0.5 * (1.0 + np.tanh(phi_lo / eps)) * 0.5 * (1.0 + np.tanh(-phi_hi / eps))


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------
def initialize_scalar(nx: int, ny: int, nz: int, z_c: torch.Tensor,
                      Lx: float, Ly: float, Lz: float, init_type: str = 'interface_z',
                      interface_pos: float = 0.5, eps_cells: float = 1.0,
                      N: int = 0, r: float = 3.0, device: str = 'cpu') -> torch.Tensor:
    """Create the initial scalar field (nx+2, ny+2, nz+2).

    init_types:
        'interface_z' : two fluids split across the channel at z = interface_pos*Lz,
                        tanh-smoothed. Reduces to pure z-diffusion (erf) when u=0.
        'interface_y' : split across the span at y = interface_pos*Ly (two
                        co-flowing streams meeting on the centreline).
        'koch'        : Koch-fractal interface (generation N, ratio r) in the
                        (z, y) cross-section, homogeneous in streamwise x. N=0
                        reduces to the flat 'interface_y' baseline.
        'uniform'     : c = interface_pos everywhere (constant-field test).
    """
    c = torch.zeros(nx + 2, ny + 2, nz + 2, device=device, dtype=torch.float64)

    if init_type == 'uniform':
        c[:] = interface_pos
        return c

    if init_type in ('koch', 'koch_strip'):
        dy = Ly / ny
        y_c = (np.arange(ny + 2) - 0.5) * dy
        z_c_np = z_c.detach().cpu().numpy()
        eps = eps_cells * min(dy, Lz / nz)
        if init_type == 'koch_strip':
            prof = koch_strip_yz(y_c, z_c_np, Ly, Lz, N=N, r=r, eps=eps)
        else:
            prof = koch_interface_yz(y_c, z_c_np, Ly, Lz, N=N, r=r, eps=eps)  # (ny+2, nz+2)
        c[:, :, :] = torch.tensor(prof, device=device, dtype=torch.float64)[None, :, :]
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
                 dz_f: torch.Tensor, chi_c: torch.Tensor = None) -> dict:
    """Volume-weighted mean and variance (intensity of segregation) of c.

    Weights cells by their (stretched) z-height so the measure is the true volume
    average. Returns mean, var and the normalised mixedness M = std/std_max with
    std_max = sqrt(mean*(1-mean)). If `chi_c` (cell-centre solid mask, 1 in solid)
    is given, only FLUID cells contribute — the right measure with an immersed
    boundary, where scalar that diffuses into the solid is not real mixing.
    """
    ci = c[1:nx+1, 1:ny+1, 1:nz+1]
    wz = dz_f[0:nz].view(1, 1, -1)
    vol = wz.expand_as(ci).clone()
    if chi_c is not None:
        vol = vol * (1.0 - chi_c[1:nx+1, 1:ny+1, 1:nz+1])
    total = vol.sum()
    mean = (ci * vol).sum() / total
    var = (((ci - mean) ** 2) * vol).sum() / total
    std = torch.sqrt(torch.clamp(var, min=0.0))
    std_max = torch.sqrt(torch.clamp(mean * (1.0 - mean), min=1e-30))
    return {"mean": float(mean), "var": float(var), "std": float(std),
            "M": float(std / std_max)}


def scalar_dissipation(c: torch.Tensor, nx: int, ny: int, nz: int,
                       dx: float, dy: float, dz_f: torch.Tensor,
                       chi_c: torch.Tensor = None) -> float:
    """Volume-weighted mean square scalar gradient <|grad c|^2>.

    The (un-scaled by D) scalar dissipation rate. Unlike the variance-based M,
    this is sensitive to INTERFACIAL AREA: a higher Koch generation has more
    interface, hence larger <|grad c|^2>, so it captures the fractal's effect that
    a gravest-mode-dominated variance metric cannot. Central differences using the
    ghost cells (call after apply_scalar_bc). Fluid-masked if chi_c is given.
    """
    gx = (c[2:nx+2, 1:ny+1, 1:nz+1] - c[0:nx, 1:ny+1, 1:nz+1]) / (2.0 * dx)
    gy = (c[1:nx+1, 2:ny+2, 1:nz+1] - c[1:nx+1, 0:ny, 1:nz+1]) / (2.0 * dy)
    dz2 = (dz_f[0:nz] + 0.0).view(1, 1, -1)
    gz = (c[1:nx+1, 1:ny+1, 2:nz+2] - c[1:nx+1, 1:ny+1, 0:nz]) / (2.0 * dz2)
    g2 = gx**2 + gy**2 + gz**2
    wz = dz_f[0:nz].view(1, 1, -1).expand_as(g2).clone()
    if chi_c is not None:
        wz = wz * (1.0 - chi_c[1:nx+1, 1:ny+1, 1:nz+1])
    return float((g2 * wz).sum() / wz.sum())


def save_scalar_field(c: torch.Tensor, results_folder: str, filename: str,
                      step: int, time: float, Sc: float) -> None:
    import os
    path = os.path.join(results_folder, filename)
    np.savez(path, c=c.detach().cpu().numpy(), step=step, time=time, Sc=Sc)


def load_scalar_field(path: str, device: str = 'cpu') -> torch.Tensor:
    data = np.load(path)
    return torch.tensor(data['c'], device=device, dtype=torch.float64)
