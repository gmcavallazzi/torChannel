"""Diagnostic: sample the inlet scalar c(y,z) along the pipe WALL circumference
(radius R=0.42) vs azimuth theta, one row per generation N. A clean two-stream
interface crosses the circle at exactly two points -> a square wave with TWO jumps
between flat c=0 and c=1 arcs. Extra jumps / intermediate plateaus reveal the
signed-distance corner artifact intruding onto the wall. usetex on."""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

usetex = os.environ.get("TORCHANNEL_USETEX", "1") == "1"
plt.rcParams.update({"text.usetex": usetex, "font.family": "serif", "font.size": 10})

R, CY, CZ = 0.42, 0.5, 0.5
LY = LZ = 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='baffle')
    ap.add_argument('--Sc', type=float, default=10.0)
    ap.add_argument('--Ns', type=int, nargs='+', default=[0, 1, 2, 3])
    ap.add_argument('--indir', default='results/campaign')
    ap.add_argument('--rfrac', type=float, default=1.0,
                    help="sample radius as fraction of R (1.0 = on the wall)")
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    th = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    rs = a.rfrac * R
    ys = CY + rs * np.cos(th)
    zs = CZ + rs * np.sin(th)
    pts = np.column_stack([ys, zs])

    rows = []
    for N in a.Ns:
        fp = os.path.join(a.indir, f"{a.mode}_Sc{int(a.Sc)}_N{N}_final.npz")
        if not os.path.exists(fp):
            continue
        cc = np.load(fp, allow_pickle=True)['scalar'][0, :, :]   # inlet (ny,nz)
        ny, nz = cc.shape
        yc = (np.arange(ny) + 0.5) * LY / ny
        zc = (np.arange(nz) + 0.5) * LZ / nz
        interp = RegularGridInterpolator((yc, zc), cc, bounds_error=False, fill_value=None)
        rows.append((N, interp(pts)))

    if not rows:
        print("no final.npz checkpoints found"); return
    nr = len(rows)
    fig, axs = plt.subplots(nr, 1, figsize=(6.5, 1.7 * nr), sharex=True, squeeze=False)
    thd = np.degrees(th)
    for ri, (N, vals) in enumerate(rows):
        ax = axs[ri][0]
        ax.plot(thd, vals, lw=1.3, color='C0')
        ax.axhline(0.5, color='0.6', ls=':', lw=0.7)
        # total variation around the loop: ~2 for a clean two-stream (down once, up once);
        # larger when extra non-monotonic wiggles (corner artifact) appear on the wall.
        tv = float(np.abs(np.diff(np.r_[vals, vals[0]])).sum())
        ax.set_ylabel(r"$N=%d$" % N)
        ax.set_ylim(-0.05, 1.05)
        ax.text(0.99, 0.92, r"TV$=%.2f$" % tv, transform=ax.transAxes,
                ha='right', va='top', fontsize=9)
        ax.grid(True, alpha=0.3)
    axs[-1][0].set_xlabel(r"$\theta$ (deg)")
    axs[-1][0].set_xlim(0, 360)
    fig.suptitle(r"$c$ on the wall, Sc$=%d$" % int(a.Sc), y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = a.out or f"results/figures/campaign_{a.mode}_Sc{int(a.Sc)}_circumference.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print("wrote", out)


if __name__ == "__main__":
    main()
