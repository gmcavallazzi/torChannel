"""
RKPM-based immersed boundary method for rigid filamentous canopies.

Direct-forcing IBM following Pinelli et al. (JCP 2010): Lagrangian markers on
the filament surfaces, Roma (1999) kernel windows corrected by reproducing
(moment) conditions so interpolation is exact for linear fields on the
non-uniform staggered grid. Spreading is the transpose scaled by marker/cell
volume ratio — force conservation follows from the partition of unity
(sum of weights = 1), with NO global solve.

Target configuration: Monti et al. (2022) "On the solidity parameter in canopy
flows" — rigid wall-normal cylindrical filaments, each represented by rings of
surface markers (4 per cross-section in the paper), one ring per wall-normal
grid cell, filaments placed RANDOMLY within their lattice tile to avoid
artificially locking the canopy-layer/outer-layer exchange.

All precomputation is batched torch on the target device (seconds, no MPI).
The runtime path is static-shape gather / index_add_ only: no host syncs, no
data-dependent control flow, safe alongside torch.compile and the CUDA-graphed
Poisson solve.

Note: index_add_ uses atomics on CUDA, so spreading is nondeterministic in the
last bits between runs where marker supports overlap. Irrelevant for turbulence
statistics.
"""

import math
import torch


def roma_kernel_1d(q):
    """Roma et al. (1999) 3-point kernel. q = (x_node - x_marker)/h, support |q| <= 1.5."""
    a = torch.abs(q)
    inner = (1.0 + torch.sqrt(torch.clamp(1.0 - 3.0 * a * a, min=0.0))) / 3.0
    b = 1.0 - a
    outer = (5.0 - 3.0 * a - torch.sqrt(torch.clamp(1.0 - 3.0 * b * b, min=0.0))) / 6.0
    zero = torch.zeros_like(a)
    return torch.where(a <= 0.5, inner, torch.where(a <= 1.5, outer, zero))


class RigidCanopyIBM:
    """
    Rigid canopy immersed boundary with RKPM transfer.

    Public API:
        apply_forcing(u, v, w, dt_t, gain_t) -> (3,) drag tensor (in-place update of u, v, w)
        interpolate(field, comp)             -> (N_L,) marker values
        slip_rms(u, v, w)                    -> 0-D tensor diagnostic
        save_geometry(folder)
    """

    CAGE = 4          # nodes per direction in the support (Roma support = 1.5 spacings)
    CHUNK = 65536     # markers per precompute batch (bounds transient memory)

    def __init__(self, cfg, nx, ny, nz, dx, dy, Lx, Ly, z_c, z_f, dz_f, dz_c, device):
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx, self.dy = dx, dy
        self.Lx, self.Ly = Lx, Ly
        self.z_c = z_c.to(self.device)
        self.z_f = z_f.to(self.device)
        self.dz_f = dz_f.to(self.device)
        self.dz_c = dz_c.to(self.device)

        self.h = float(cfg['h'])
        self.n_fil_x = int(cfg['n_fil_x'])
        self.n_fil_y = int(cfg['n_fil_y'])
        self.placement = cfg.get('placement', 'random_in_tile')
        self.seed = int(cfg.get('seed', 0))
        self.diameter = float(cfg.get('diameter', 2.2 * dx))
        self.markers_per_ring = int(cfg.get('markers_per_ring', 4))
        self.n_iter = int(cfg.get('forcing', {}).get('n_iter', 1))
        self.normalize = bool(cfg.get('rkpm', {}).get('normalize', True))

        self._generate_filaments()
        self._generate_markers()

        # Per-component RKPM transfer tensors (flat indices + weights)
        self.idx = {}
        self.w_int = {}
        self.w_spr = {}
        import time
        t0 = time.time()
        for comp in ('u', 'v', 'w'):
            self.idx[comp], self.w_int[comp], self.w_spr[comp] = self._build_weights(comp)
        if self.normalize:
            self._normalize_response()
        # Effective marker volumes AFTER all weight scalings: dV_eff[l] =
        # sum_n w_spr[l,n] V_n, so that sum_l dU_l dV_eff[l] equals the grid
        # momentum actually deposited (exact drag bookkeeping under epsilon)
        self.dV_eff = {}
        for comp in ('u', 'v', 'w'):
            _, _, _, z_vol, _, shape = self._component_layout(comp)
            k = self.idx[comp] % shape[2]
            V_n = (self.dx * self.dy) * z_vol[k]
            self.dV_eff[comp] = (self.w_spr[comp] * V_n).sum(dim=1).contiguous()
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        print(f"  RKPM weights built for {self.N_L} markers in {time.time()-t0:.2f} s", flush=True)

        lam = self.diameter * self.h / (self.tile_x * self.tile_y)
        print(f"Canopy: {self.n_fil} filaments ({self.n_fil_x}x{self.n_fil_y}), "
              f"h={self.h}, d={self.diameter:.4f} ({self.diameter/dx:.2f} dx), "
              f"solidity lambda={lam:.3f}", flush=True)
        print(f"  {self.n_rings} rings/filament x {self.markers_per_ring} markers/ring "
              f"-> N_L={self.N_L}, placement={self.placement} (seed={self.seed})", flush=True)

    # ------------------------------------------------------------------ geometry

    def _generate_filaments(self):
        """Filament centers: one per lattice tile, placed randomly within it (Monti 2022)."""
        self.tile_x = self.Lx / self.n_fil_x
        self.tile_y = self.Ly / self.n_fil_y
        self.n_fil = self.n_fil_x * self.n_fil_y

        ix = torch.arange(self.n_fil_x, dtype=torch.float64).repeat_interleave(self.n_fil_y)
        iy = torch.arange(self.n_fil_y, dtype=torch.float64).repeat(self.n_fil_x)

        if self.placement == 'regular':
            fx, fy = 0.5 * torch.ones(self.n_fil, dtype=torch.float64), \
                     0.5 * torch.ones(self.n_fil, dtype=torch.float64)
        elif self.placement == 'random_in_tile':
            # Keep the whole cross-section inside the tile (margin d/2) so
            # filaments in adjacent tiles stay at least d apart.
            gen = torch.Generator().manual_seed(self.seed)
            r = torch.rand(self.n_fil, 2, generator=gen, dtype=torch.float64)
            mx = 0.5 * self.diameter / self.tile_x
            my = 0.5 * self.diameter / self.tile_y
            if mx >= 0.5 or my >= 0.5:
                raise ValueError("Filament diameter exceeds lattice tile size")
            fx = mx + r[:, 0] * (1.0 - 2.0 * mx)
            fy = my + r[:, 1] * (1.0 - 2.0 * my)
        else:
            raise ValueError(f"Unknown canopy placement '{self.placement}'")

        self.fil_x = ((ix + fx) * self.tile_x).to(self.device)
        self.fil_y = ((iy + fy) * self.tile_y).to(self.device)

    def _generate_markers(self):
        """Rings of surface markers: one ring per wall-normal cell inside the canopy."""
        z_centers = self.z_c[1:self.nz + 1]           # interior cell centers
        ring_mask = z_centers < self.h
        self.ring_z = z_centers[ring_mask]            # (n_rings,)
        self.ring_dz = self.dz_f[:self.nz][ring_mask]  # cell thickness per ring
        self.n_rings = int(self.ring_z.shape[0])
        if self.n_rings == 0:
            raise ValueError("No grid cells inside the canopy: check h vs the grid")

        m = self.markers_per_ring
        R = 0.5 * self.diameter if m > 1 else 0.0

        # Random azimuthal rotation per (filament, ring) so marker columns
        # don't align across rings/filaments (seeded, reproducible)
        gen = torch.Generator().manual_seed(self.seed + 1)
        rot = torch.rand(self.n_fil, self.n_rings, 1, generator=gen, dtype=torch.float64).to(self.device)
        j = torch.arange(m, dtype=torch.float64, device=self.device).view(1, 1, m)
        theta = 2.0 * math.pi * (j + rot) / m         # (n_fil, n_rings, m)

        x = self.fil_x.view(-1, 1, 1) + R * torch.cos(theta)
        y = self.fil_y.view(-1, 1, 1) + R * torch.sin(theta)
        z = self.ring_z.view(1, -1, 1).expand(self.n_fil, -1, m)

        self.x_lag = x.reshape(-1).contiguous()
        self.y_lag = y.reshape(-1).contiguous()
        self.z_lag = z.reshape(-1).contiguous()
        self.N_L = self.x_lag.shape[0]

        # Marker volume: share of the filament slice it represents.
        # For surface rings: (pi d^2/4) * dz_ring / markers_per_ring (~ one grid
        # cell for d = 2.2 dx). For a single centerline marker: the local cell.
        if m > 1:
            slice_vol = 0.25 * math.pi * self.diameter ** 2 * self.ring_dz / m
        else:
            slice_vol = self.dx * self.dy * self.ring_dz
        self.dV = slice_vol.view(1, -1, 1).expand(self.n_fil, -1, m).reshape(-1).contiguous()

    # ------------------------------------------------------------------ RKPM weights

    def _component_layout(self, comp):
        """Node coordinate definitions per staggered component (full ghosted arrays).

        Returns (x_type, y_type, z_nodes, z_vol, z_valid, shape) where *_type is
        'face' or 'center', z_nodes are the physical node coordinates indexed by
        the RAW array index, z_vol[k] the wall-normal volume weight of node k,
        z_valid = (k_min, k_max) the canonical interior range, and shape the
        full array shape used for flat indexing.
        """
        nx, ny, nz = self.nx, self.ny, self.nz
        if comp == 'u':
            # u[i, j, k]: x-face i at i*dx (canonical 1..nx), y-center, z-center
            z_vol = torch.cat([torch.zeros(1, device=self.device), self.dz_f,
                               torch.zeros(1, device=self.device)])
            return 'face', 'center', self.z_c, z_vol, (1, nz), (nx + 1, ny + 2, nz + 2)
        if comp == 'v':
            z_vol = torch.cat([torch.zeros(1, device=self.device), self.dz_f,
                               torch.zeros(1, device=self.device)])
            return 'center', 'face', self.z_c, z_vol, (1, nz), (nx + 2, ny + 1, nz + 2)
        if comp == 'w':
            # w[i, j, k]: z-face k at z_f[k]; k=0 and k=nz are the walls (w=0 there,
            # excluded — spreading into them would be wiped by the BCs)
            return 'center', 'center', self.z_f, self.dz_c, (1, nz - 1), (nx + 2, ny + 2, nz + 1)
        raise ValueError(comp)

    def _uniform_dir(self, s, spacing, n, kind):
        """Cage indices, coordinates and kernel values for a uniform periodic direction.

        Returns (idx, phi, xi): canonical wrapped array indices (B, 4), kernel
        values (B, 4), scaled node-marker offsets (B, 4).
        """
        if kind == 'face':
            base = torch.floor(s / spacing).to(torch.int64)
            coord = lambda i: i.to(torch.float64) * spacing
        else:
            base = torch.floor(s / spacing + 0.5).to(torch.int64)
            coord = lambda i: (i.to(torch.float64) - 0.5) * spacing
        offs = torch.arange(-1, 3, device=self.device, dtype=torch.int64)
        raw = base.unsqueeze(1) + offs                    # (B, 4) unwrapped
        xi = (coord(raw) - s.unsqueeze(1)) / spacing
        phi = roma_kernel_1d(xi)
        idx = (raw - 1).remainder(n) + 1                  # canonical [1..n]
        return idx, phi, xi

    def _wall_dir(self, s, z_nodes, k_valid):
        """Cage indices/kernel for the non-uniform wall-normal direction.

        Out-of-range nodes are masked (zero weight); the RKPM moment correction
        restores exact linear reproduction on the truncated one-sided support.
        """
        k_min, k_max = k_valid
        k0 = torch.searchsorted(z_nodes, s).to(torch.int64) - 1
        offs = torch.arange(-1, 3, device=self.device, dtype=torch.int64)
        raw = k0.unsqueeze(1) + offs                      # (B, 4)
        valid = (raw >= k_min) & (raw <= k_max)
        idx = raw.clamp(k_min, k_max)
        zn = z_nodes[idx]
        # Local kernel width: largest node gap inside the cage (non-uniform grid)
        gaps = (zn[:, 1:] - zn[:, :-1]).abs()
        h_z = gaps.max(dim=1, keepdim=True).values.clamp(min=1e-300)
        xi = (zn - s.unsqueeze(1)) / h_z
        phi = roma_kernel_1d(xi) * valid.to(torch.float64)
        return idx, phi, xi

    def _build_weights(self, comp):
        """RKPM weights for one component: (flat_idx, w_int, w_spr), each (N_L, 64)."""
        x_kind, y_kind, z_nodes, z_vol, z_valid, shape = self._component_layout(comp)
        sx, sy, sz = shape
        n_sup = self.CAGE ** 3

        flat_idx = torch.empty(self.N_L, n_sup, dtype=torch.int64, device=self.device)
        w_int = torch.empty(self.N_L, n_sup, dtype=torch.float64, device=self.device)
        w_spr = torch.empty(self.N_L, n_sup, dtype=torch.float64, device=self.device)

        e0 = torch.zeros(4, dtype=torch.float64, device=self.device)
        e0[0] = 1.0

        for lo in range(0, self.N_L, self.CHUNK):
            hi = min(lo + self.CHUNK, self.N_L)
            xs = self.x_lag[lo:hi]
            ys = self.y_lag[lo:hi]
            zs = self.z_lag[lo:hi]
            B = hi - lo

            ix, phx, xix = self._uniform_dir(xs, self.dx, self.nx, x_kind)
            iy, phy, xiy = self._uniform_dir(ys, self.dy, self.ny, y_kind)
            iz, phz, xiz = self._wall_dir(zs, z_nodes, z_valid)

            # Tensor-product kernel and node volumes over the 4x4x4 cage
            phi = (phx.view(B, 4, 1, 1) * phy.view(B, 1, 4, 1) * phz.view(B, 1, 1, 4)).reshape(B, n_sup)
            vol = (self.dx * self.dy) * z_vol[iz]                      # (B, 4)
            vol = vol.view(B, 1, 1, 4).expand(B, 4, 4, 4).reshape(B, n_sup)

            # Reproducing-condition correction (linear basis, scaled coordinates)
            p = torch.stack([
                torch.ones(B, n_sup, dtype=torch.float64, device=self.device),
                xix.view(B, 4, 1, 1).expand(B, 4, 4, 4).reshape(B, n_sup),
                xiy.view(B, 1, 4, 1).expand(B, 4, 4, 4).reshape(B, n_sup),
                xiz.view(B, 1, 1, 4).expand(B, 4, 4, 4).reshape(B, n_sup),
            ], dim=2)                                                   # (B, 64, 4)
            m_w = phi * vol                                             # (B, 64)
            M = torch.einsum('bn,bni,bnj->bij', m_w, p, p)              # (B, 4, 4)
            a = torch.linalg.solve(M, e0.expand(B, 4))
            W = torch.einsum('bni,bi->bn', p, a) * m_w                  # sum_n W = 1 exactly

            # Spreading: transpose scaled by volume ratio (conservative)
            vol_safe = torch.where(vol > 0, vol, torch.ones_like(vol))
            Ws = W * self.dV[lo:hi].unsqueeze(1) / vol_safe

            fi = (ix.view(B, 4, 1, 1) * sy + iy.view(B, 1, 4, 1)) * sz + iz.view(B, 1, 1, 4)
            flat_idx[lo:hi] = fi.reshape(B, n_sup)
            w_int[lo:hi] = W
            w_spr[lo:hi] = Ws

        return flat_idx, w_int.contiguous(), w_spr.contiguous()

    def _apply_coupling(self, d, comp):
        """A d where A = interp(spread(.)) is the marker-to-marker coupling."""
        _, _, _, _, _, shape = self._component_layout(comp)
        n_flat = shape[0] * shape[1] * shape[2]
        work = torch.zeros(n_flat, dtype=torch.float64, device=self.device)
        work.index_add_(0, self.idx[comp].reshape(-1),
                        (d.unsqueeze(1) * self.w_spr[comp]).reshape(-1))
        return (work[self.idx[comp]] * self.w_int[comp]).sum(dim=1)

    def _normalize_response(self):
        """Pinelli et al. (2010) epsilon scaling + stability-safe gain.

        epsilon is the SELF-response normalization: scale each marker's spread
        weights so that interpolating back its own spread footprint returns
        exactly 1 (A_ll = 1). No coupled solve: with overlapping markers (4 per
        cross-section ring) the full system A d = 1 is ill-conditioned — that
        dense solve is what destabilized the removed implementation.

        Cross-talk between overlapping markers makes the largest eigenvalue of
        A exceed 1; single-shot forcing with gain alpha is stable iff
        alpha * lambda_max < 2. We estimate lambda_max matrix-free by power
        iteration and expose recommended_alpha = min(1, 1.8/lambda_max) for the
        solver's 'auto' gain setting.
        """
        self.lambda_max = 0.0
        for comp in ('u', 'v', 'w'):
            # epsilon: diagonal of A = same-marker interp*spread products
            diag = (self.w_int[comp] * self.w_spr[comp]).sum(dim=1)
            if (diag <= 0).any():
                raise RuntimeError(f"Non-positive self-response in component {comp}")
            self.w_spr[comp] = (self.w_spr[comp] / diag.unsqueeze(1)).contiguous()
            print(f"  epsilon ({comp}): self-response 1/eps in "
                  f"[{diag.min().item():.4f}, {diag.max().item():.4f}]", flush=True)

            # power iteration for lambda_max of the normalized coupling
            gen = torch.Generator().manual_seed(99)
            q = torch.rand(self.N_L, generator=gen, dtype=torch.float64).to(self.device)
            q = q / q.norm()
            lam = 0.0
            for _ in range(50):
                Aq = self._apply_coupling(q, comp)
                lam = Aq.norm().item()
                q = Aq / max(lam, 1e-300)
            self.lambda_max = max(self.lambda_max, lam)
            print(f"  coupling lambda_max ({comp}) ~ {lam:.3f}", flush=True)

        self.recommended_alpha = min(1.0, 1.8 / self.lambda_max)
        print(f"  recommended forcing gain alpha = {self.recommended_alpha:.3f} "
              f"(stability bound 2/lambda_max = {2.0/self.lambda_max:.3f})", flush=True)

    # ------------------------------------------------------------------ runtime

    def interpolate(self, field, comp):
        """Interpolate a full ghosted field to the markers. Returns (N_L,)."""
        return (field.reshape(-1)[self.idx[comp]] * self.w_int[comp]).sum(dim=1)

    def _spread_increment(self, field, dU, comp):
        """Add the spread of marker velocity increments dU (N_L,) to the field, in place."""
        field.reshape(-1).index_add_(0, self.idx[comp].reshape(-1),
                                     (dU.unsqueeze(1) * self.w_spr[comp]).reshape(-1))

    def apply_forcing(self, u, v, w, dt_t, gain_t):
        """Direct forcing towards zero velocity at the markers (rigid canopy).

        Updates u, v, w in place (interior nodes only; ghosts must be refreshed
        by apply_bc afterwards, as the solver already does). Returns the total
        canopy force on the fluid, (3,) device tensor (no host sync).
        """
        acc_u = torch.zeros((), dtype=torch.float64, device=self.device)
        acc_v = torch.zeros((), dtype=torch.float64, device=self.device)
        acc_w = torch.zeros((), dtype=torch.float64, device=self.device)
        for _ in range(self.n_iter):
            dU = -gain_t * self.interpolate(u, 'u')
            dV = -gain_t * self.interpolate(v, 'v')
            dW = -gain_t * self.interpolate(w, 'w')
            self._spread_increment(u, dU, 'u')
            self._spread_increment(v, dV, 'v')
            self._spread_increment(w, dW, 'w')
            acc_u = acc_u + (dU * self.dV_eff['u']).sum()
            acc_v = acc_v + (dV * self.dV_eff['v']).sum()
            acc_w = acc_w + (dW * self.dV_eff['w']).sum()
        # Force ON the fluid = momentum added per unit time (negative x-force = drag)
        return torch.stack([acc_u, acc_v, acc_w]) / dt_t

    def slip_rms(self, u, v, w):
        """RMS residual marker velocity (diagnostic; call sparingly, e.g. every n_out)."""
        eu = self.interpolate(u, 'u')
        ev = self.interpolate(v, 'v')
        ew = self.interpolate(w, 'w')
        return torch.sqrt((eu.square() + ev.square() + ew.square()).mean())

    # ------------------------------------------------------------------ I/O

    def save_geometry(self, folder):
        import os
        import numpy as np
        path = os.path.join(folder, 'canopy_geometry.npz')
        np.savez(path,
                 fil_x=self.fil_x.cpu().numpy(), fil_y=self.fil_y.cpu().numpy(),
                 ring_z=self.ring_z.cpu().numpy(),
                 x_lag=self.x_lag.cpu().numpy(), y_lag=self.y_lag.cpu().numpy(),
                 z_lag=self.z_lag.cpu().numpy(), dV=self.dV.cpu().numpy(),
                 h=self.h, diameter=self.diameter, seed=self.seed,
                 n_fil_x=self.n_fil_x, n_fil_y=self.n_fil_y,
                 markers_per_ring=self.markers_per_ring, placement=self.placement)
        print(f"Canopy geometry saved to {path}", flush=True)
