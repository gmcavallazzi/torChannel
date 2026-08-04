"""Time-average CaNS velstats_fld_*.out into a torChannel turbulence_stats.npz.

Lets plot_statistics.py draw a CaNS run with exactly the same code path, axes and
reference overlays as a torChannel run, so the two are compared like for like.
Everything except the 2D spectra, which CaNS's out1d_chan does not write.

CaNS writes ONE FILE PER OUTPUT (output.f90:364-372), each holding the
INSTANTANEOUS plane averages at that step:

    z, um, vm, wm, u2, v2, w2, uw

where `u2 = <u^2>_xy - um^2` etc. are formed per snapshot (output.f90:360-363).

THE CORRECTION. Because each u2 is a variance about that snapshot's OWN plane
mean, naively averaging the u2 column over files gives <(u - U(t))^2>, not the
Reynolds stress <(u - <U>)^2>. The two differ by the time variance of the plane
mean, and the identity is exact:

    <(u - <U>)^2>  =  mean_t(u2)  +  var_t(um)

so this script adds var_t back. It biases u'u' and v'v' LOW if omitted, and is
invisible in w'w' and <u'w'> because continuity plus periodicity pin the plane
mean of w to zero. torChannel accumulates the correction internally; CaNS leaves
it to the user, which is precisely the trap this script exists to avoid.

CaNS also already does the two things it should on the staggered grid: `um` and
`u2` are taken at the u NODES with no interpolation, and `w2` averages
0.5*(w(k)^2 + w(k-1)^2) -- the statistic, not the field. So no de-filtering is
needed here.

Usage:
    python scripts/cans_stats_to_npz.py <cans_run>/data out.npz \\
        --nu 3.5807e-4 --Lx 12.566370614359172 --Ly 6.283185307179586 \\
        [--skip 0] [--gamma 1.6]
"""

import argparse
import glob
import os
import sys

import numpy as np


def read_velstats(path):
    rows = []
    for line in open(path):
        s = line.split()
        if not s:
            continue
        try:
            rows.append([float(x) for x in s])
        except ValueError:
            continue
    return np.asarray(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data_dir", help="CaNS data/ directory")
    ap.add_argument("out", help="output .npz")
    ap.add_argument("--nu", type=float, required=True)
    ap.add_argument("--Lx", type=float, required=True)
    ap.add_argument("--Ly", type=float, required=True)
    ap.add_argument("--skip", type=int, default=0,
                    help="drop this many leading files (restart transient)")
    ap.add_argument("--gamma", type=float, default=None,
                    help="if given, rebuild z_f from torChannel's symmetric tanh "
                         "grid and check it against CaNS's own z (they coincide "
                         "for CaNS gr = 2*gamma)")
    a = ap.parse_args(argv)

    files = sorted(glob.glob(os.path.join(a.data_dir, "velstats_fld_*.out")))
    files = files[a.skip:]
    if not files:
        sys.exit(f"no velstats_fld_*.out in {a.data_dir} after skipping {a.skip}")

    acc = None
    for i, f in enumerate(files):
        r = read_velstats(f)
        if acc is None:
            acc = np.zeros((len(files), *r.shape))
        elif r.shape != acc.shape[1:]:
            sys.exit(f"{f}: shape {r.shape} != {acc.shape[1:]}")
        acc[i] = r
    n = len(files)

    z = acc[0, :, 0]
    um, vm, wm = acc[:, :, 1], acc[:, :, 2], acc[:, :, 3]
    u2, v2, w2, uw = acc[:, :, 4], acc[:, :, 5], acc[:, :, 6], acc[:, :, 7]

    U_mean = um.mean(axis=0)
    V_mean = vm.mean(axis=0)
    W_mean = wm.mean(axis=0)

    # The correction (see module docstring). var_t of the plane means, added to
    # the time-averaged per-snapshot variances.
    uu_mean = u2.mean(axis=0) + um.var(axis=0)
    vv_mean = v2.mean(axis=0) + vm.var(axis=0)
    ww_mean = w2.mean(axis=0) + wm.var(axis=0)
    uw_mean = uw.mean(axis=0) + ((um - U_mean) * (wm - W_mean)).mean(axis=0)

    nz = len(z)
    # z are CELL CENTRES. Faces follow from z_c[k] = (z_f[k] + z_f[k+1])/2 with
    # z_f[0] = 0 -- exact for any grid, no assumption about the stretching.
    z_f = np.empty(nz + 1)
    z_f[0] = 0.0
    for k in range(nz):
        z_f[k + 1] = 2.0 * z[k] - z_f[k]
    Lz = z_f[-1]
    dz_f = np.diff(z_f)

    if a.gamma is not None:
        xi = 2.0 * np.arange(nz + 1) / nz - 1.0
        ref = 0.5 * Lz * (1.0 + np.tanh(a.gamma * xi) / np.tanh(a.gamma))
        err = np.abs(ref - z_f).max()
        print(f"  grid check vs torChannel gamma={a.gamma}: max|dz_f| = {err:.3e}")

    u_tau = np.sqrt(a.nu * 0.5 * (U_mean[0] / z[0] + U_mean[-1] / (Lz - z[-1])))
    print(f"  {n} snapshots, nz = {nz}, Lz = {Lz:.6f}")
    print(f"  u_tau = {u_tau:.6f}, Re_tau = {u_tau * (Lz / 2) / a.nu:.2f}")
    corr = 100.0 * (np.sqrt(uu_mean.max() / u2.mean(axis=0).max()) - 1.0)
    print(f"  plane-mean-variance correction: {corr:+.3f}% on peak u'_rms")

    np.savez(a.out,
             z_c=z, dz_f=dz_f, z_f=z_f, nx=192, ny=192, nz=nz,
             U_mean=U_mean, V_mean=V_mean, W_mean=W_mean,
             uu_mean=uu_mean, vv_mean=vv_mean, ww_mean=ww_mean, uw_mean=uw_mean,
             n_samples=n, nu=a.nu, Lx=a.Lx, Ly=a.Ly, Lz=Lz, delta=Lz / 2,
             top_wall_bc_type="dirichlet", u_tau=u_tau)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
