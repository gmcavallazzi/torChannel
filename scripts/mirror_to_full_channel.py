#!/usr/bin/env python
"""Build a full (closed) channel field by mirroring an open-channel one.

An open channel of depth delta and a closed channel of half-height delta share
the same lower half, and -- for the tanh grids used here -- share it exactly:
the symmetric grid on [0, 2delta] with nz cells has a lower half bit-identical
to the 'bottom' grid on [0, delta] with nz/2 cells. So the mirror needs no
interpolation.

    u, v   mirror EVEN   about the centreline
    w      mirrors ODD                          (w(delta) = 0)

WHY A DISTURBANCE IS MANDATORY
------------------------------
Reflection about the centreline is an EXACT symmetry of the Navier-Stokes
equations. A perfectly mirrored field lies on an invariant manifold: in exact
arithmetic it would stay symmetric forever, and in floating point it decays only
as round-off amplifies at the Lyapunov rate -- far too slow to wait out. The two
halves must be decorrelated deliberately.

The default here is a SPANWISE SHIFT of the mirrored half by Ly/2. That is
preferable to adding noise:

  * it is exact under periodicity, so the upper half remains a genuine, fully
    developed turbulent field rather than a perturbed one;
  * Ly/2 = pi*delta is many streak spacings (~100 wall units ~ 0.55 delta), so
    the halves are decorrelated immediately rather than after a growth phase;
  * random noise is not turbulence -- it has to break down first, which costs
    more transient than it saves.

The cost is a velocity jump of order u'_rms at the centreline plane. It is a
single plane, it is smoothed by viscosity within a few time units, and the
solver re-projects the field on load, so the divergence is removed. A small
random perturbation can be added on top (--noise) to seed three-dimensional
breakdown of any residual coherence.

EVEN SO, BUDGET A REAL TRANSIENT. Centreline-spanning structures have to grow
from scratch; nothing here creates them.

Usage:
    python scripts/mirror_to_full_channel.py \
        results_re180_open/fields.npz full180_seed.npz --shift-y
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def mirror(src_path, out_path, shift_y=True, shift_x=False, noise=0.0, seed=0):
    d = np.load(src_path)
    u, v, w, p = d['u'], d['v'], d['w'], d['p']
    z_c, z_f = d['z_c'], d['z_f']
    Lx, Ly = float(d['Lx']), float(d['Ly'])
    Lz = float(z_f[-1])

    nx = u.shape[0] - 1
    ny = v.shape[1] - 1
    nz = len(z_f) - 1

    # ---- mirrored grid: [0, Lz] + reflection -> [0, 2 Lz] --------------------
    z_f_full = np.concatenate([z_f, 2.0 * Lz - z_f[-2::-1]])
    zc_i = 0.5 * (z_f_full[:-1] + z_f_full[1:])
    z_c_full = np.concatenate([[-zc_i[0]], zc_i, [2.0 * z_f_full[-1] - zc_i[-1]]])
    nz2 = 2 * nz

    rng = np.random.default_rng(seed)
    sy = ny // 2 if shift_y else 0
    sx = nx // 2 if shift_x else 0

    def _roll_periodic(b, shift, axis, n_interior, staggered):
        """Periodic shift of the INTERIOR along a ghosted/staggered axis.

        Rolling the raw array is wrong: these axes carry two ghost layers (or,
        when staggered, a duplicated seam face), so np.roll over the full length
        does not correspond to a periodic shift of the n_interior physical
        cells -- it scrambles the seam and leaves a divergence the periodic
        Poisson solver cannot represent. Roll the interior only; the solver
        refreshes the ghosts on load.
        """
        if not shift:
            return b
        b = b.copy()
        sl = [slice(None)] * b.ndim
        if staggered:
            # faces 0..n_interior, with face n_interior == face 0 (periodic seam)
            sl[axis] = slice(0, n_interior)
            core = np.roll(b[tuple(sl)], shift, axis=axis)
            b[tuple(sl)] = core
            sl_last = [slice(None)] * b.ndim; sl_last[axis] = slice(n_interior, n_interior + 1)
            sl_first = [slice(None)] * b.ndim; sl_first[axis] = slice(0, 1)
            b[tuple(sl_last)] = b[tuple(sl_first)]
        else:
            # one ghost each side: interior is 1..n_interior
            sl[axis] = slice(1, n_interior + 1)
            b[tuple(sl)] = np.roll(b[tuple(sl)], shift, axis=axis)
        return b

    def refl(a, axis_z, odd=False, stag_x=False, stag_y=False):
        """Reflect the interior in z, then shift in x/y to break the symmetry."""
        b = np.flip(a, axis=axis_z)
        if odd:
            b = -b
        b = _roll_periodic(b, sy, 1, ny, stag_y)
        b = _roll_periodic(b, sx, 0, nx, stag_x)
        return b

    # u: (nx+1, ny+2, nz+2)   cell-centred in z, interior 1..nz
    U = np.zeros((nx + 1, ny + 2, nz2 + 2))
    U[:, :, 1:nz + 1] = u[:, :, 1:nz + 1]
    U[:, :, nz + 1:nz2 + 1] = refl(u[:, :, 1:nz + 1], 2, stag_x=True)

    # v: (nx+2, ny+1, nz+2)
    V = np.zeros((nx + 2, ny + 1, nz2 + 2))
    V[:, :, 1:nz + 1] = v[:, :, 1:nz + 1]
    V[:, :, nz + 1:nz2 + 1] = refl(v[:, :, 1:nz + 1], 2, stag_y=True)

    # w: (nx+2, ny+2, nz+1)  at z-faces, interior 0..nz; ODD, and w(centre) = 0
    W = np.zeros((nx + 2, ny + 2, nz2 + 1))
    W[:, :, 0:nz + 1] = w[:, :, 0:nz + 1]
    W[:, :, nz + 1:nz2 + 1] = refl(w[:, :, 0:nz], 2, odd=True)
    W[:, :, nz] = 0.0          # centreline face
    W[:, :, 0] = 0.0
    W[:, :, -1] = 0.0

    # p: cell-centred, even. Only a starting guess -- the projection rebuilds it.
    P = np.zeros((nx + 2, ny + 2, nz2 + 2))
    P[:, :, 1:nz + 1] = p[:, :, 1:nz + 1]
    P[:, :, nz + 1:nz2 + 1] = refl(p[:, :, 1:nz + 1], 2)

    if noise > 0.0:
        for F in (U, V, W):
            F[1:-1, 1:-1, 2:-2] += (rng.random(F[1:-1, 1:-1, 2:-2].shape) - 0.5) * 2 * noise

    np.savez(out_path, u=U, v=V, w=W, p=P,
             z_c=z_c_full, z_f=z_f_full, Lx=Lx, Ly=Ly,
             step=0, time=0.0,
             u_tau=float(d['u_tau']) if 'u_tau' in d.files else 0.0,
             forcing=float(d['forcing']) if 'forcing' in d.files else 0.0)

    # ---- report -------------------------------------------------------------
    Ui = U[1:nx + 1, 1:ny + 1, 1:nz2 + 1]
    lo = Ui[:, :, nz - 1].mean(axis=(0, 1))
    hi = Ui[:, :, nz].mean(axis=(0, 1))
    corr = np.corrcoef(Ui[:, :, nz - 1].ravel(), Ui[:, :, nz].ravel())[0, 1]
    print(f"source : {nx}x{ny}x{nz}   Lz = {Lz:.4f}")
    print(f"output : {nx}x{ny}x{nz2}  Lz = {2*Lz:.4f}   -> {out_path}")
    print(f"symmetry breaking: shift_y={shift_y} shift_x={shift_x} noise={noise}")
    print(f"  correlation of u across the centreline planes = {corr:+.4f}"
          f"   ({'DECORRELATED' if abs(corr) < 0.3 else 'STILL CORRELATED -- symmetry not broken'})")
    print(f"  mean u just below / above centreline = {lo:.5f} / {hi:.5f}")
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('source', help='open-channel fields.npz')
    p.add_argument('output', help='destination full-channel fields.npz')
    p.add_argument('--shift-y', action='store_true', default=True,
                   help='shift the mirrored half by Ly/2 (default, recommended)')
    p.add_argument('--no-shift-y', dest='shift_y', action='store_false')
    p.add_argument('--shift-x', action='store_true',
                   help='also shift by Lx/2')
    p.add_argument('--noise', type=float, default=0.0,
                   help='additional uniform random perturbation amplitude')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args(argv)
    mirror(a.source, a.output, a.shift_y, a.shift_x, a.noise, a.seed)
    return 0


if __name__ == '__main__':
    sys.exit(main())
