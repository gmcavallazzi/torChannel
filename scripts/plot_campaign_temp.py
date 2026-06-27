"""TEMPORARY preview of the Sc=10 baffle campaign from whatever checkpoints exist.

Uses the developing-duct steady observable M(x) for every N: converged cases from
*_final.npz (x, Mx); a still-running case from the latest snapshot in *_snaps.npz
(Mx), drawn dashed and tagged 'partial'. Right panel: L_mix(N)/L_mix(0) at a chosen
M threshold vs the proposal's Eq.4 r^{-Df N}. Throwaway -> usetex off for speed.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

usetex = os.environ.get("TORCHANNEL_USETEX", "1") == "1"
plt.rcParams.update({"text.usetex": usetex, "font.family": "serif", "font.size": 10})


def crossing(xvec, Mvec, thr):
    M = np.asarray(Mvec); x = np.asarray(xvec)
    below = np.where(M <= thr)[0]
    if len(below) == 0 or below[0] == 0:
        return np.nan
    i = below[0]
    x0, x1, m0, m1 = x[i-1], x[i], M[i-1], M[i]
    return x0 + (thr - m0) * (x1 - x0) / (m1 - m0) if m1 != m0 else x1


def load_Mx(indir, mode, Sc, N, force_snap=False):
    """Return (x, Mx, converged?) using final.npz if present else latest snapshot.
    force_snap=True takes the latest snapshot even if a final.npz exists (use when the
    final on disk is stale, e.g. a case being rerun)."""
    tag = f"{mode}_Sc{int(Sc)}_N{N}"
    fp = os.path.join(indir, f"{tag}_final.npz")
    sp = os.path.join(indir, f"{tag}_snaps.npz")
    if os.path.exists(fp) and not force_snap:
        d = np.load(fp, allow_pickle=True)
        return d['x'], d['Mx'], True
    if os.path.exists(sp):
        d = np.load(sp, allow_pickle=True)
        snaps = d['snaps']; Lx = float(d['Lx'])
        x = np.linspace(0, Lx, len(snaps[-1]['Mx']))
        return x, snaps[-1]['Mx'], False
    return None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='baffle')
    ap.add_argument('--Sc', type=float, default=10.0)
    ap.add_argument('--Ns', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--indir', default='results/campaign')
    ap.add_argument('--thr', type=float, default=0.5)
    ap.add_argument('--out', default=None)
    ap.add_argument('--left-only', action='store_true',
                    help="plot only the M(x) panel (no L_mix ratio panel)")
    ap.add_argument('--snap-Ns', type=int, nargs='+', default=[],
                    help="take these N from the latest snapshot (stale/rerunning final)")
    a = ap.parse_args()
    Df, r = np.log(4)/np.log(3), 3.0

    if a.left_only:
        fig, ax0 = plt.subplots(1, 1, figsize=(5.0, 4.0))
        axs = [ax0, None]
    else:
        fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.0))
    Lmix = {}
    for N in a.Ns:
        x, Mx, conv = load_Mx(a.indir, a.mode, a.Sc, N, force_snap=(N in a.snap_Ns))
        if x is None:
            continue
        lab = (r"$N=%d$" % N) + ("" if conv else " (partial)")
        axs[0].plot(x, Mx, lw=1.6, ls='-' if conv else '--', label=lab)
        Lmix[N] = crossing(x, Mx, a.thr)

    axs[0].axhline(a.thr, color='0.6', ls=':', lw=0.8)
    axs[0].set_xlabel(r"$x$"); axs[0].set_ylabel(r"$M$")
    axs[0].set_title(r"$M(x)$, Sc$=%d$" % int(a.Sc))
    axs[0].grid(True, alpha=0.3); axs[0].legend()

    Ns = sorted(k for k in Lmix if np.isfinite(Lmix[k]))
    if not a.left_only and Ns and 0 in Ns and np.isfinite(Lmix[0]) and Lmix[0] > 0:
        ratio = [Lmix[N]/Lmix[0] for N in Ns]
        axs[1].plot(Ns, ratio, 'o-', lw=1.6, label=r"DNS")
        axs[1].plot(Ns, [r**(-Df*N) for N in Ns], 's--', color='C3',
                    label=r"Eq.4 $r^{-D_f N}$")
        axs[1].set_xlabel(r"$N$")
        axs[1].set_ylabel(r"$L_{\rm mix}(N)/L_{\rm mix}(0)$")
        axs[1].set_title(r"Ratio at $M=%.2f$" % a.thr)
        axs[1].grid(True, alpha=0.3); axs[1].legend()
    fig.tight_layout()
    out = a.out or f"results/figures/TEMP_campaign_{a.mode}_Sc{int(a.Sc)}.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print("L_mix(N) (M=%.2f):" % a.thr,
          {N: round(float(Lmix.get(N, np.nan)), 3) for N in a.Ns})
    print("ratio:", {N: round(float(Lmix[N]/Lmix[0]), 4)
                     for N in Ns} if Ns and 0 in Ns else "n/a")
    print("wrote", out)


if __name__ == "__main__":
    main()
