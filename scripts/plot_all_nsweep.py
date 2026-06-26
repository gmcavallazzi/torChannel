"""Combined summary: L_mix(N)/L_mix(0) across all tested regimes vs the proposal's
r^{-Df N}. Plain duct (Sc=1, Sc=16) and herringbone duct (Sc=16) all stay flat."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

usetex = os.environ.get("TORCHANNEL_USETEX", "0") == "1"
plt.rcParams.update({"text.usetex": usetex, "font.family": "serif",
                     "axes.labelsize": 12, "axes.titlesize": 12, "legend.fontsize": 9})

def tmix(t, M, thr=0.2):
    for i in range(1, len(M)):
        if M[i-1] > thr >= M[i]:
            return t[i-1] + (M[i-1]-thr)/(M[i-1]-M[i])*(t[i]-t[i-1])
    return np.nan

def ratios_diff(path, sc):
    d = np.load(path, allow_pickle=True)
    tm = [tmix(d[f"Sc{sc:g}_N{N}_t"], d[f"Sc{sc:g}_N{N}_M"]) for N in (0,1,2)]
    return np.array(tm)/tm[0]

def ratios_mix(path):
    d = np.load(path, allow_pickle=True)
    tm = [tmix(d[f"N{N}_t"], d[f"N{N}_M"]) for N in (0,1,2)]
    return np.array(tm)/tm[0]

Ns = np.array([0,1,2]); r=3.0; Df=np.log(4)/np.log(r)
series = [
    ("plain duct, Sc=1",      ratios_diff("results/duct_diff_Sc1/duct_diffusion.npz", 1),  'o-', "#1f77b4"),
    ("plain duct, Sc=16",     ratios_diff("results/duct_diff_Sc16/duct_diffusion.npz",16), 's-', "#2ca02c"),
    ("herringbone duct, Sc=16", ratios_mix("results/herringbone_duct/mixing_results.npz"), '^-', "#d62728"),
]

fig, ax = plt.subplots(figsize=(6.6, 4.8))
ax.plot(Ns, r**(-Df*Ns), 'k--D', lw=2, ms=7, label=r"proposal $r^{-D_f N}$")
for lab, rr, mk, col in series:
    ax.plot(Ns, rr, mk, color=col, lw=1.8, ms=7, label=lab)
ax.set_yscale('log'); ax.set_xticks(Ns)
ax.set_xlabel(r"Koch generation $N$")
ax.set_ylabel(r"$L_{\mathrm{mix}}(N)/L_{\mathrm{mix}}(0)$")
ax.set_title(r"Mixing length: measured (flat) vs proposed scaling")
ax.legend(loc='lower left'); ax.grid(True, which='both', alpha=0.3)
ax.set_ylim(0.04, 1.5)
os.makedirs("results/figures", exist_ok=True)
out = "results/figures/all_regimes_Nsweep.png"
fig.savefig(out, dpi=140, bbox_inches='tight'); print("wrote", out)
for lab, rr, _, _ in series: print(f"{lab:28s}: {rr}")
print(f"prediction r^-DfN          : {r**(-Df*Ns)}")
