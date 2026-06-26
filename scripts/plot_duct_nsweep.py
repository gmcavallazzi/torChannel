"""Summary figure for the duct Koch N-sweep: measured L_mix ratio (Sc-independent,
flat) vs the proposal's r^{-Df N}, plus the M(t) collapse across N."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

usetex = os.environ.get("TORCHANNEL_USETEX", "0") == "1"
plt.rcParams.update({"text.usetex": usetex, "font.family": "serif",
                     "axes.labelsize": 12, "axes.titlesize": 12, "legend.fontsize": 9})

def load(path, sc):
    d = np.load(path, allow_pickle=True)
    r = float(d['r']); Df = float(d['Df'])
    out = {}
    for N in (0, 1, 2):
        t = d[f"Sc{sc:g}_N{N}_t"]; M = d[f"Sc{sc:g}_N{N}_M"]
        # t_mix = first M<0.05 crossing
        tm = np.nan
        for i in range(1, len(M)):
            if M[i-1] > 0.05 >= M[i]:
                tm = t[i-1] + (M[i-1]-0.05)/(M[i-1]-M[i])*(t[i]-t[i-1]); break
        out[N] = dict(t=t, M=M, tmix=tm)
    return out, r, Df

s1, r, Df = load("results/duct_diff_Sc1/duct_diffusion.npz", 1)
s16, _, _ = load("results/duct_diff_Sc16/duct_diffusion.npz", 16)
Ns = np.array([0, 1, 2])
ratio1 = np.array([s1[N]['tmix']/s1[0]['tmix'] for N in Ns])
ratio16 = np.array([s16[N]['tmix']/s16[0]['tmix'] for N in Ns])
pred = r ** (-Df * Ns)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.3))
ax.plot(Ns, pred, 'k--s', lw=1.8, ms=6, label=r"proposal $r^{-D_f N}$")
ax.plot(Ns, ratio1, 'o-', color="#1f77b4", lw=1.8, ms=7, label=r"measured Sc=1")
ax.plot(Ns, ratio16, '^-', color="#d62728", lw=1.8, ms=7, label=r"measured Sc=16")
ax.set_yscale('log'); ax.set_xticks(Ns)
ax.set_xlabel(r"Koch generation $N$")
ax.set_ylabel(r"$L_{\mathrm{mix}}(N)/L_{\mathrm{mix}}(0)$")
ax.set_title("mixing length vs prediction")
ax.legend(); ax.grid(True, which='both', alpha=0.3)

for N in Ns:
    ax2.semilogy(s16[N]['t'], s16[N]['M'], 'o-', ms=2.5, lw=1.5, label=rf"$N={N}$")
ax2.axhline(0.05, ls=':', color='k', lw=1)
ax2.set_xlabel(r"time $t$"); ax2.set_ylabel(r"segregation $M(t)$")
ax2.set_title("Sc=16: M(t) collapses across N")
ax2.legend(); ax2.grid(True, which='both', alpha=0.3)

os.makedirs("results/figures", exist_ok=True)
out = "results/figures/duct_koch_Nsweep.png"
fig.savefig(out, dpi=140, bbox_inches='tight')
print("wrote", out)
print(f"measured ratios  Sc=1 : {ratio1}")
print(f"measured ratios  Sc=16: {ratio16}")
print(f"prediction r^-DfN     : {pred}")
