"""Threshold-sensitivity of the fractal mixing ratio L_mix(N)/L_mix(0).

For each Schmidt number, sweep the mixedness threshold M and plot the ratio
L_mix(N)/L_mix(0) vs M, one curve per generation N. Demonstrates that the
ratio varies smoothly and monotonically with the threshold (it does not
flicker) and that the N-ordering is preserved at every M, so the qualitative
conclusions are threshold-independent (only the magnitude shifts).

usetex on by default (run after `module load texlive`); TORCHANNEL_USETEX=0 to disable.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

usetex = os.environ.get("TORCHANNEL_USETEX", "1") == "1"
plt.rcParams.update({"text.usetex": usetex, "font.family": "serif", "font.size": 14})

def lmix(Mx, Lx, thr):
    x = np.linspace(0, Lx, len(Mx)); b = np.where(Mx < thr)[0]
    if len(b) == 0 or b[0] == 0:
        return np.nan
    i = b[0]
    return x[i-1] + (thr - Mx[i-1]) * (x[i] - x[i-1]) / (Mx[i] - Mx[i-1])


def panel(ax, indir, Sc, kind, Ns, Mlo, Mhi):
    data = {N: np.load(os.path.join(indir, f"baffle_Sc{Sc}_N{N}_{kind}.npz")) for N in [0] + Ns}
    Lx = {N: float(data[N]['Lx']) for N in data}
    M = np.arange(Mlo, Mhi, 0.004)
    cmap = plt.get_cmap('viridis')               # same scheme as the M(x) figure
    for N in Ns:
        r = np.array([lmix(data[N]['Mx'], Lx[N], t) / lmix(data[0]['Mx'], Lx[0], t) for t in M])
        ok = np.isfinite(r)
        ax.plot(M[ok], r[ok], '-', color=cmap(N / max(Ns)), lw=1.9, label=r"$N=%d$" % N)
    ax.set_xlabel(r"threshold $M$")
    ax.set_title(r"$Sc=%d$" % Sc)
    ax.set_ylim(0, 1.02)
    ax.grid(True, ls=':', lw=0.6, alpha=0.7)
    ax.legend(frameon=False, fontsize=12, loc='upper right')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--indir', default='results/campaign')
    ap.add_argument('--outdir', default='results/figures/march_double')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.4))
    panel(axs[0], a.indir, 10,  'final', [1, 2, 3, 4], 0.655, 0.925)
    panel(axs[1], a.indir, 100, 'march', [1, 2, 3, 4], 0.655, 0.925)
    axs[0].set_ylabel(r"$L_{\mathrm{mix}}(N)/L_{\mathrm{mix}}(0)$")
    fig.tight_layout()
    o = os.path.join(a.outdir, "baffle_ratio_vs_threshold.png")
    fig.savefig(o, dpi=150, bbox_inches='tight'); print("wrote", o)


if __name__ == "__main__":
    main()
