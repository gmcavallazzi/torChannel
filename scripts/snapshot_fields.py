"""Cross-section snapshots c(y,z) at mid-x for the rebuttal: initial Koch ICs and
their evolution in the strong chaotic mixer. Solid (immersed groove) masked white."""
import os, sys, copy, tempfile, yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_JIT", "0")
import torch
from solver import ChannelFlow

usetex = os.environ.get("TORCHANNEL_USETEX", "1") == "1"
plt.rcParams.update({"text.usetex": usetex, "font.family": "serif", "font.size": 10})

CFG = "configs/herringbone_duct_strong.yaml"
base = yaml.safe_load(open(CFG))
base['compute']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
times = [0.0, 1.0, 4.0, 12.0, 29.0]

def run(N):
    c = copy.deepcopy(base); c['scalar']['N'] = N
    f = tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False); yaml.safe_dump(c, f); f.close()
    sim = ChannelFlow(f.name)
    nx, ny, nz = sim.nx, sim.ny, sim.nz
    ix = nx // 2
    solid = sim.chi_c[1+ix, 1:ny+1, 1:nz+1].cpu().numpy() > 0.5
    frames = {}
    step_targets = [int(t/sim.dt) for t in times]
    nmax = max(step_targets)
    def grab():
        cc = sim.scalar[1+ix, 1:ny+1, 1:nz+1].detach().cpu().numpy().copy()
        cc[solid] = np.nan
        return cc
    frames[0] = grab()
    for n in range(1, nmax+1):
        sim.step_imex(sim.dt)
        if n in step_targets:
            frames[n] = grab()
    return [frames[0]] + [frames[s] for s in step_targets[1:]], sim

# y,z extents
Ly = base['domain']['Ly']; Lz = base['domain']['Lz']
frames0, sim = run(0)
frames2, _ = run(2)
ny = sim.ny; nz = sim.nz

cmap = plt.cm.RdBu_r; cmap.set_bad('0.6')
fig, axes = plt.subplots(2, len(times), figsize=(2.5*len(times), 4.8))
for row, (frames, lab) in enumerate([(frames0, r"$N=0$"), (frames2, r"$N=2$")]):
    for col, (fr, t) in enumerate(zip(frames, times)):
        ax = axes[row, col]
        ax.imshow(fr.T, origin='lower', extent=[0, Ly, 0, Lz], aspect='auto',
                  cmap=cmap, vmin=0, vmax=1)
        if row == 0: ax.set_title(rf"$t={t:g}$")
        if col == 0: ax.set_ylabel(lab + r"\\ $z$" if usetex else f"{lab}  z")
        ax.set_xticks([]); ax.set_yticks([])
        if row == 1: ax.set_xlabel(r"$y$")
fig.suptitle(r"Scalar $c(y,z)$ at mid-$x$ (strong herringbone duct, Sc$=16$)")
fig.tight_layout(rect=[0, 0, 1, 0.96])
os.makedirs("results/figures", exist_ok=True)
fig.savefig("results/figures/snapshots_herringbone.png", dpi=140, bbox_inches='tight')
print("wrote results/figures/snapshots_herringbone.png")

# initial-IC-only figure for N=0,1,2 (cheap second pass not needed: reuse N0,N2 t=0; build N1)
frames1, _ = run(1)
fig2, axes2 = plt.subplots(1, 3, figsize=(7.5, 2.9))
for ax, fr, lab in zip(axes2, [frames0[0], frames1[0], frames2[0]], [r"$N=0$", r"$N=1$", r"$N=2$"]):
    ax.imshow(fr.T, origin='lower', extent=[0, Ly, 0, Lz], aspect='auto', cmap=cmap, vmin=0, vmax=1)
    ax.set_title(lab); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(r"$y$")
axes2[0].set_ylabel(r"$z$")
fig2.suptitle(r"Initial Koch interface $c(y,z)$")
fig2.tight_layout(rect=[0, 0, 1, 0.95])
fig2.savefig("results/figures/koch_ic.png", dpi=140, bbox_inches='tight')
print("wrote results/figures/koch_ic.png")
