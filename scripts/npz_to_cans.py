"""Convert a torChannel fields.npz into a CaNS restart file (fld.bin).

The inverse of scripts/cans_to_npz.py, so that CaNS can be started from a
converged torChannel field instead of spinning up from scratch. Both codes use
the same MAC staggering, so no interpolation is involved -- this is a pure
layout change (drop ghosts, Fortran order, append the two trailing scalars).

CaNS restart layout, from src/load.f90:
    good = (product(ng)*4 + 2) * sizeof(real64)
i.e. u, v, w, p each ng(1)*ng(2)*ng(3) reals in Fortran order with NO halo,
back to back, then time and istep as two more reals.

INDEXING. Both codes put u at (x_face, y_centre, z_centre), v at
(x_centre, y_face, z_centre), w at (x_centre, y_centre, z_face), and in both
the stored index of a face is the face on the HIGH side of that cell. So all
four fields map from the same torChannel interior slice
[1:nx+1, 1:ny+1, 1:nz+1]:

    CaNS u(i,j,k) = tc_u[i, j, k]      i = 1..nx  -> face at x = i*dx
    CaNS v(i,j,k) = tc_v[i, j, k]      j = 1..ny  -> face at y = j*dy
    CaNS w(i,j,k) = tc_w[i, j, k]      k = 1..nz  -> face at z = z_f[k],
                                       so k = nz is the top wall (w = 0)
    CaNS p(i,j,k) = tc_p[i, j, k]

THE GRID IS NOT STORED IN fld.bin -- CaNS rebuilds it from input.nml. The
caller is responsible for matching it. For torChannel's 'symmetric' tanh grid
this is exact rather than approximate:

    torChannel  z_f = 0.5*Lz*(1 + tanh(gamma*xi)/tanh(gamma)),  xi = 2k/nz - 1
    CaNS gtype=1 z_f = 0.5*Lz*(1 + tanh((k/nz - 0.5)*gr)/tanh(gr/2))

which coincide for gr = 2*gamma (verified: max|dz| = 2.2e-16 at nz=256,
gamma=1.6). This script checks that identity and refuses to write if it fails.

Usage:
    python scripts/npz_to_cans.py results_re180_closed/fields_final.npz \\
        <cans_run_dir>/data/fld.bin --gamma 1.6
"""

import argparse
import os
import sys

import numpy as np


def cans_grid(nz, Lz, gr):
    """CaNS gtype=1 (CLUSTER_TWO_END) face coordinates, from initgrid.f90."""
    xi = np.arange(nz + 1, dtype=np.float64) / nz
    return 0.5 * (1.0 + np.tanh((xi - 0.5) * gr) / np.tanh(gr / 2.0)) * Lz


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("npz")
    ap.add_argument("out", help="destination fld.bin")
    ap.add_argument("--gamma", type=float, default=None,
                    help="torChannel stretching parameter; if given, verifies "
                         "that CaNS gr = 2*gamma reproduces the same grid")
    ap.add_argument("--time", type=float, default=0.0,
                    help="time to stamp into the restart (default 0)")
    ap.add_argument("--istep", type=int, default=0)
    a = ap.parse_args(argv)

    d = np.load(a.npz)
    u, v, w, p = d["u"], d["v"], d["w"], d["p"]
    nx, ny, nz = u.shape[0] - 1, v.shape[1] - 1, w.shape[2] - 1
    z_f = d["z_f"]
    Lz = float(z_f[-1] - z_f[0])
    print(f"  {os.path.basename(a.npz)}: {nx}x{ny}x{nz}, Lz = {Lz:.6f}, "
          f"t = {float(d['time']):.3f}")

    if a.gamma is not None:
        gr = 2.0 * a.gamma
        err = np.abs(cans_grid(nz, Lz, gr) - z_f).max()
        print(f"  grid check: CaNS gtype=1 gr={gr} vs torChannel gamma={a.gamma}"
              f" -> max|dz_f| = {err:.3e}")
        if err > 1e-12:
            sys.exit("grid mismatch: CaNS would integrate on a different mesh")

    # Interior only, identical slice for all four (see module docstring).
    sl = (slice(1, nx + 1), slice(1, ny + 1), slice(1, nz + 1))
    fields = [np.ascontiguousarray(f[sl], dtype=np.float64) for f in (u, v, w, p)]
    for name, f in zip("uvwp", fields):
        assert f.shape == (nx, ny, nz), f"{name}: {f.shape}"

    # w's last plane is the top wall and must be exactly zero, which doubles as
    # a check that the z-indexing convention was not shifted by one.
    top = np.abs(fields[2][:, :, -1]).max()
    if top > 1e-12:
        sys.exit(f"w at the top wall is {top:.3e}, not 0 -- z indexing is off")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "wb") as fh:
        for f in fields:
            f.flatten(order="F").tofile(fh)
        np.array([a.time, float(a.istep)], dtype=np.float64).tofile(fh)

    expect = (nx * ny * nz * 4 + 2) * 8
    got = os.path.getsize(a.out)
    print(f"  wrote {a.out}: {got} bytes (CaNS expects {expect}) "
          f"{'OK' if got == expect else 'SIZE MISMATCH'}")
    return 0 if got == expect else 1


if __name__ == "__main__":
    sys.exit(main())
