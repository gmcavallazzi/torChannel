#!/usr/bin/env python
"""Build a full (closed) channel field by mirroring an open-channel one.

An open channel of depth delta and a closed channel of half-height delta share
the same lower half, and -- for the tanh grids used here -- share it exactly:
the symmetric grid on [0, 2delta] with nz cells has a lower half bit-identical
to the 'bottom' grid on [0, delta] with nz/2 cells. So the mirror needs no
interpolation.

    u, v   mirror EVEN   about the centreline
    w      mirrors ODD                          (w(delta) = 0)

WHY A PERTURBATION IS MANDATORY, AND WHAT IT MUST CONTAIN
--------------------------------------------------------
Reflection about the centreline is an EXACT symmetry of Navier-Stokes. A
perfectly mirrored field lies on an invariant manifold: it would stay symmetric
indefinitely, decaying only as round-off amplifies. The symmetry must be broken
deliberately.

But breaking it is not enough -- WHAT is injected matters. Under the mirror,

    omega_x = dw/dy - dv/dz     and     omega_y = du/dz - dw/dx

are both ODD about z = delta, so both vanish identically on the centreline
plane, and w vanishes there too. The centreline is then a symmetry plane: no
vortex lines cross it and no fluid is exchanged between the halves. A
perturbation that does not restore omega_x, omega_y and w at the centre does not
promote exchange, however thoroughly it decorrelates the halves.

This is why a SPANWISE SHIFT of the mirrored half is the wrong tool, despite
decorrelating the halves perfectly. Measured on such a seed:

    z        rms omega_x   rms omega_y   rms w
    0.504       0.856         0.805      0.0483
    0.993       2.476         3.161      0.0046     <- centreline
    1.007       2.478         3.158      0.0045

w at the centre is 9% of its ambient value -- nothing crosses the plane -- while
omega_x and omega_y are 3-4x TOO LARGE and confined to the two cells straddling
the centre. That is not turbulence: it is a vortex sheet produced by the
discontinuity the shift creates, and it radiates spurious structure as it rolls
up.

The default is therefore a SMOOTH SOLENOIDAL PERTURBATION (--perturb), applied
with a Gaussian weight centred on the centreline. It is built by upsampling a
coarse random field, so it carries a physical length scale rather than grid-scale
white noise (which viscosity erases before it can do anything), and it puts
energy directly into w and into omega_x, omega_y where they are needed.

BUDGET A REAL TRANSIENT REGARDLESS. Centreline-spanning structures still have to
grow; the perturbation seeds them, it does not create them.

Usage:
    python scripts/mirror_to_full_channel.py \
        results_re180_open/fields.npz full180_seed.npz --shift-y
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _smooth_noise(shape, rng, coarsen=8):
    """Random field with a length scale ~coarsen cells (trilinear upsample).

    Grid-scale white noise is useless here: viscosity removes it long before it
    can generate any cross-centreline motion. Coarsening first gives the
    perturbation a length scale comparable to the local eddies.
    """
    cs = tuple(max(2, n // coarsen) for n in shape)
    a = rng.standard_normal(cs)
    for ax, (n, c) in enumerate(zip(shape, cs)):
        # PERIODIC upsample in x and y (axes 0,1): wrap the upper neighbour, so
        # the perturbation has no seam. A seam in x or y would give the field a
        # net flux, hence a non-zero-mean divergence, which the all-Neumann
        # Poisson problem cannot represent -- the projection would silently fail.
        periodic = ax in (0, 1)
        idx = np.arange(n) * (c / n) if periodic else np.linspace(0, c - 1, n)
        lo = np.floor(idx).astype(int) % c
        hi = (lo + 1) % c if periodic else np.minimum(lo + 1, c - 1)
        f = (idx - np.floor(idx)).reshape([-1 if k == ax else 1 for k in range(3)])
        a = np.take(a, lo, axis=ax) * (1 - f) + np.take(a, hi, axis=ax) * f
    return a / (a.std() + 1e-30)


def mirror(src_path, out_path, shift_y=False, shift_x=False, noise=0.0, seed=0,
           perturb=0.0, perturb_width=0.6):
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

    if perturb > 0.0:
        # Smooth solenoidal-ish perturbation, weighted onto the centreline. The
        # solver projects on load, so the divergence it introduces is removed;
        # what survives is the rotational part, which is what we want.
        zc_i = z_c_full[1:-1]
        gz = np.exp(-((zc_i - Lz) / (perturb_width * Lz)) ** 2)      # centred on z = Lz (= delta)
        gzf = np.exp(-((z_f_full - Lz) / (perturb_width * Lz)) ** 2)
        # u is staggered in x: faces 0 and nx are the SAME location, so they must
        # receive the same perturbation or the x-flux stops telescoping to zero.
        Up = perturb * gz[None, None, :] * _smooth_noise((nx, ny, nz2), rng)
        U[0:nx, 1:ny + 1, 1:nz2 + 1] += Up
        U[nx, 1:ny + 1, 1:nz2 + 1] = U[0, 1:ny + 1, 1:nz2 + 1]
        # v is staggered in y: same argument on faces 0 and ny.
        Vp = perturb * gz[None, None, :] * _smooth_noise((nx, ny, nz2), rng)
        V[1:nx + 1, 0:ny, 1:nz2 + 1] += Vp
        V[1:nx + 1, ny, 1:nz2 + 1] = V[1:nx + 1, 0, 1:nz2 + 1]
        # w is cell-centred in x and y; only the walls are constrained.
        Wp = perturb * gzf[None, None, :] * _smooth_noise((nx, ny, nz2 + 1), rng)
        Wp[:, :, 0] = 0.0; Wp[:, :, -1] = 0.0        # impermeable walls
        W[1:nx + 1, 1:ny + 1, :] += Wp

    np.savez(out_path, u=U, v=V, w=W, p=P,
             z_c=z_c_full, z_f=z_f_full, Lx=Lx, Ly=Ly,
             step=0, time=0.0,
             u_tau=float(d['u_tau']) if 'u_tau' in d.files else 0.0,
             forcing=float(d['forcing']) if 'forcing' in d.files else 0.0)

    # ---- report -------------------------------------------------------------
    Ui = U[1:nx + 1, 1:ny + 1, 1:nz2 + 1]
    lo = Ui[:, :, nz - 1].mean(axis=(0, 1))
    hi = Ui[:, :, nz].mean(axis=(0, 1))
    # Symmetry metric: correlate u at delta-D against u at delta+D. Correlating
    # the two planes ADJACENT to the centre is meaningless -- they are 0.01 delta
    # apart and would correlate strongly in any turbulent field, symmetric or not.
    zc_i = z_c_full[1:-1]
    print("  mirror symmetry  corr(u(delta-D), u(delta+D)):")
    for dk in (4, 12, 30, 60, 100):
        if nz - dk < 0 or nz - 1 + dk >= 2 * nz:
            continue
        a = Ui[:, :, nz - dk]; b = Ui[:, :, nz - 1 + dk]
        c = np.corrcoef(a.ravel(), b.ravel())[0, 1]
        print(f"      D = {abs(zc_i[nz - dk] - Lz):.3f} delta : {c:+.4f}"
              f"{'   <- still symmetric' if c > 0.9 else ''}")
    corr = np.corrcoef(Ui[:, :, nz - 12].ravel(), Ui[:, :, nz - 1 + 12].ravel())[0, 1]
    print(f"source : {nx}x{ny}x{nz}   Lz = {Lz:.4f}")
    print(f"output : {nx}x{ny}x{nz2}  Lz = {2*Lz:.4f}   -> {out_path}")
    print(f"symmetry breaking: perturb={perturb} (width {perturb_width} delta), "
          f"shift_y={shift_y} shift_x={shift_x} noise={noise}")
    print(f"  symmetry at D = 0.15 delta: {corr:+.4f}"
          f"   ({'BROKEN' if abs(corr) < 0.8 else 'NOT SUFFICIENTLY BROKEN -- raise --perturb'})")
    print(f"  mean u just below / above centreline = {lo:.5f} / {hi:.5f}")
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('source', help='open-channel fields.npz')
    p.add_argument('output', help='destination full-channel fields.npz')
    p.add_argument('--perturb', type=float, default=0.05,
                   help='amplitude of the smooth centreline perturbation, in U_b '
                        '(default 0.05 ~ w_rms at a closed-channel centreline)')
    p.add_argument('--perturb-width', type=float, default=0.6,
                   help='Gaussian half-width of the perturbation, in delta. Wide '
                        'enough to desymmetrise most of the channel, while still '
                        'peaking at the centreline and leaving the near-wall '
                        'region (where the turbulence is already correct) largely '
                        'undisturbed')
    p.add_argument('--shift-y', action='store_true', default=False,
                   help='ALSO shift the mirrored half by Ly/2. Decorrelates the '
                        'halves but injects a vortex sheet at the centreline -- '
                        'not recommended, see the module docstring')
    p.add_argument('--no-shift-y', dest='shift_y', action='store_false')
    p.add_argument('--shift-x', action='store_true',
                   help='also shift by Lx/2')
    p.add_argument('--noise', type=float, default=0.0,
                   help='additional uniform random perturbation amplitude')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args(argv)
    mirror(a.source, a.output, a.shift_y, a.shift_x, a.noise, a.seed,
           a.perturb, a.perturb_width)
    return 0


if __name__ == '__main__':
    sys.exit(main())
