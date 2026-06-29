"""Double-panel M(x) segregation decay for two Schmidt numbers side by side
(reads the lightweight parabolic-march npz `{mode}_Sc{Sc}_N{N}_march.npz`).

Left panel Sc=100, right panel Sc=1000 by default; generation N overlaid in each
(viridis). usetex on by default (run after `module load texlive`); set
TORCHANNEL_USETEX=0 to disable. Larger base font than the single-Sc plots.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

usetex = os.environ.get("TORCHANNEL_USETEX", "1") == "1"
plt.rcParams.update({"text.usetex": usetex, "font.family": "serif", "font.size": 14})


def panel(ax, indir, mode, Sc, Ns):
    rows = []
    for N in Ns:
        fp = os.path.join(indir, f"{mode}_Sc{int(Sc)}_N{N}_march.npz")
        if os.path.exists(fp):
            rows.append((N, np.load(fp)))
    cmap = plt.get_cmap('viridis')
    for N, d in rows:
        ax.plot(d['x'], d['Mx'], lw=1.9, color=cmap(N / max(1, len(rows) - 1)),
                label=r"$N=%d$" % N)
    ax.set_xlabel(r"$x$")
    ax.set_title(r"$Sc=%d$" % int(Sc))
    ax.grid(True, ls=':', lw=0.6, alpha=0.7)
    ax.legend(ncol=2, frameon=False, fontsize=12)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='baffle')
    ap.add_argument('--indir', default='results/campaign')
    ap.add_argument('--outdir', default='results/figures/march_double')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.4))
    panel(axs[0], a.indir, a.mode, 100, [0, 1, 2, 3, 4])
    panel(axs[1], a.indir, a.mode, 1000, [0, 1, 2, 3, 4, 5])
    axs[0].set_ylabel(r"$M(x)$")
    fig.tight_layout()
    o = os.path.join(a.outdir, f"{a.mode}_Mx_double_Sc100_Sc1000.png")
    fig.savefig(o, dpi=150, bbox_inches='tight'); print("wrote", o)


if __name__ == "__main__":
    main()
