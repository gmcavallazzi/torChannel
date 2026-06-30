"""Plot the parabolic-marcher baffle Sc-sweep from the LIGHTWEIGHT march npz
(`{mode}_Sc{Sc}_N{N}_march.npz`, which stores Mx + c_yz cross-sections, NOT the full
256^3 field). Two figures, NO L_mix-vs-Eq.4 ratio panel:
  1. M(x) decay, N overlaid (left-plot-only convention).
  2. (y,z) scalar cross-sections, rows = generation N, columns = streamwise station x,
     clipped to the analytic pipe circle (baffle = plain disc R=0.42).
usetex on (run after `module load texlive`)."""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

usetex = os.environ.get("TORCHANNEL_USETEX", "1") == "1"
plt.rcParams.update({"text.usetex": usetex, "font.family": "serif", "font.size": 10})

R, CY, CZ, LY, LZ = 0.42, 0.5, 0.5, 1.0, 1.0   # baffle: inscribed disc in unit square


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='baffle')
    ap.add_argument('--Sc', type=float, default=100.0)
    ap.add_argument('--Ns', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--indir', default='results/campaign')
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--thrs', type=float, nargs='+', default=[0.97, 0.95, 0.93, 0.90],
                    help="thresholds for the L_mix(N)/L_mix(0) readout")
    a = ap.parse_args()
    outdir = a.outdir or f"results/figures/march_Sc{int(a.Sc)}"

    rows = []
    for N in a.Ns:
        fp = os.path.join(a.indir, f"{a.mode}_Sc{int(a.Sc)}_N{N}_march.npz")
        if not os.path.exists(fp):
            print("skip (missing):", fp); continue
        rows.append((N, np.load(fp)))
    if not rows:
        print("no march npz found"); return
    os.makedirs(outdir, exist_ok=True)

    # ---- L_mix(N)/L_mix(0) at several thresholds (find where the ratio saturates) ----
    def lmix(x, M, thr):
        b = np.where(M < thr)[0]
        if len(b) == 0 or b[0] == 0:
            return np.nan
        i = b[0]
        return float(x[i-1] + (thr - M[i-1]) * (x[i] - x[i-1]) / (M[i] - M[i-1]))
    print(f"L_mix(N)/L_mix(0), Sc={int(a.Sc)}:")
    print("  thr  " + "".join(f"  N={N}" for N, _ in rows))
    for thr in a.thrs:
        L = {N: lmix(d['x'], d['Mx'], thr) for N, d in rows}
        L0 = L[rows[0][0]]
        cells = "".join(f"  {L[N]/L0:.3f}" if np.isfinite(L[N]) and np.isfinite(L0) else "   nan"
                        for N, _ in rows)
        print(f"  {thr:.2f}{cells}")

    # ---- (1) M(x) decay, N overlaid ----
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    cmap = plt.get_cmap('viridis')
    for N, d in rows:
        ax.plot(d['x'], d['Mx'], lw=1.7, color=cmap(N / max(1, len(rows) - 1)),
                label=r"$N=%d$" % N)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$M(x)$")
    ax.set_title(r"Sc$=%d$ baffle" % int(a.Sc))
    ax.grid(True, ls=':', lw=0.6, alpha=0.7)
    ax.legend(ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    o1 = os.path.join(outdir, f"{a.mode}_Sc{int(a.Sc)}_Mx_march.png")
    fig.savefig(o1, dpi=150, bbox_inches='tight'); print("wrote", o1)

    # ---- (2) (y,z) cross-sections, rows = N, cols = stations ----
    xs = rows[0][1]['xs']
    nc = len(xs)
    fig, axs = plt.subplots(len(rows), nc, figsize=(1.9 * nc, 1.9 * len(rows) + 0.4),
                            squeeze=False)
    th = np.linspace(0, 2 * np.pi, 2001)
    yb, zb = CY + R * np.cos(th), CZ + R * np.sin(th)
    for ri, (N, d) in enumerate(rows):
        c_yz = d['c_yz']                       # (nstations, ny, nz)
        ny, nz = c_yz.shape[1], c_yz.shape[2]
        yc = (np.arange(ny) + 0.5) * LY / ny
        zc = (np.arange(nz) + 0.5) * LZ / nz
        Yc, Zc = np.meshgrid(yc, zc, indexing='ij')
        for ci in range(nc):
            ax = axs[ri][ci]
            pcm = ax.pcolormesh(Yc, Zc, c_yz[ci], shading='gouraud', cmap='RdBu_r',
                                vmin=0, vmax=1)
            clip = PathPatch(Path(np.column_stack([yb, zb])), transform=ax.transData,
                             facecolor='none', edgecolor='none')
            ax.add_patch(clip); pcm.set_clip_path(clip)
            ax.plot(yb, zb, color='k', lw=0.7)
            ax.set_aspect('equal'); ax.set_xlim(0, LY); ax.set_ylim(0, LZ)
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.99, 0.99, r"$M{=}%.2f$" % float(d['Mx'][np.argmin(np.abs(d['x'] - xs[ci]))]),
                    transform=ax.transAxes, ha='right', va='top', fontsize=9)
            if ri == 0:
                ax.set_title(r"$x=%.1f$" % xs[ci], fontsize=15)
            if ci == 0:
                ax.set_ylabel(r"$N=%d$" % N, fontsize=16)
    fig.suptitle(r"Sc$=%d$ baffle: $c(y,z)$" % int(a.Sc), y=1.0, fontsize=30)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    o2 = os.path.join(outdir, f"{a.mode}_Sc{int(a.Sc)}_xsections_march.png")
    fig.savefig(o2, dpi=150, bbox_inches='tight'); print("wrote", o2)


if __name__ == "__main__":
    main()
