"""Render Sc-campaign results from the incremental checkpoints written by
mixing_campaign.py. Works on partial data (read whatever *_history/_final exist).

  baffle         : M(t) decay per N (t == downstream distance x=U*t) + L_mix(N) ratio.
  surface_baffle : steady M(x) per N + the near-inlet N-dependence + L_mix(N) ratio.

Usage: python scripts/plot_campaign.py --mode surface_baffle --Sc 10 [--Ns 0 1 2 3 4]
L_mix(N) is the t (or x) at which M crosses a threshold; the ratio L_mix(N)/L_mix(0) is
the proposal's Eq.4 observable (predicted r^{-Df N}; r=3,Df=1.262 -> 0.25,0.06,...)."""
import os, sys, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

usetex = os.environ.get("TORCHANNEL_USETEX", "1") == "1"
plt.rcParams.update({"text.usetex": usetex, "font.family": "serif", "font.size": 10})


def crossing(xvec, Mvec, thr):
    """First x where M drops below thr (linear interp); nan if never."""
    M = np.asarray(Mvec); x = np.asarray(xvec)
    below = np.where(M <= thr)[0]
    if len(below) == 0 or below[0] == 0:
        return np.nan
    i = below[0]
    x0, x1, m0, m1 = x[i-1], x[i], M[i-1], M[i]
    return x0 + (thr - m0) * (x1 - x0) / (m1 - m0) if m1 != m0 else x1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['baffle', 'surface_baffle'])
    ap.add_argument('--Sc', type=float, default=10.0)
    ap.add_argument('--Ns', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--indir', default='results/campaign')
    ap.add_argument('--thr', type=float, default=0.5, help="M threshold defining L_mix")
    a = ap.parse_args()
    Df, r = np.log(4)/np.log(3), 3.0

    fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.0))
    Lmix = {}
    for N in a.Ns:
        tag = f"{a.mode}_Sc{int(a.Sc)}_N{N}"
        hp = os.path.join(a.indir, f"{tag}_history.npz")
        fp = os.path.join(a.indir, f"{tag}_final.npz")
        if os.path.exists(fp):
            d = np.load(fp); xv, Mv = d['x'], d['Mx']; xlab = r"$x$"      # steady M(x)
        elif os.path.exists(hp):
            d = np.load(hp); xv, Mv = d['t'], d['M']; xlab = r"$t$"        # partial: M(t) so far
        else:
            continue
        axs[0].plot(xv, Mv, lw=1.6, label=r"$N=%d$" % N)
        Lmix[N] = crossing(xv, Mv, a.thr)

    axs[0].axhline(a.thr, color='0.6', ls=':', lw=0.8)
    axs[0].set_xlabel(xlab); axs[0].set_ylabel(r"segregation $M$")
    axs[0].set_title(r"$M$ decay (Sc$=%d$)" % int(a.Sc)); axs[0].grid(True, alpha=0.3); axs[0].legend()

    Ns = sorted(k for k in Lmix if np.isfinite(Lmix[k]))
    if Ns and 0 in Ns and np.isfinite(Lmix[0]) and Lmix[0] > 0:
        ratio = [Lmix[N]/Lmix[0] for N in Ns]
        axs[1].plot(Ns, ratio, 'o-', lw=1.6, label=r"DNS")
        axs[1].plot(Ns, [r**(-Df*N) for N in Ns], 's--', color='C3', label=r"Eq.4 $r^{-D_f N}$")
        axs[1].set_xlabel(r"$N$"); axs[1].set_ylabel(r"$L_{\rm mix}(N)/L_{\rm mix}(0)$")
        axs[1].set_title(r"At $M=%.2f$" % a.thr); axs[1].grid(True, alpha=0.3); axs[1].legend()
    fig.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out = f"results/figures/campaign_{a.mode}_Sc{int(a.Sc)}.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print("L_mix(N) (M=%.2f):" % a.thr, {N: round(float(Lmix.get(N, np.nan)), 3) for N in a.Ns})
    print("wrote", out)


if __name__ == "__main__":
    main()
