"""Surface-experiment snapshots: scalar c(y,z) at mid-x with a FLAT interface over a
smooth wall (N=0) vs a fractal-corrugated wall (N=2). Shows the fractal floor and that
the fractal wall mixes a flat interface SLOWER (fragmented rolls)."""
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
cmap = plt.cm.RdBu_r; cmap.set_bad('0.6')

base = yaml.safe_load(open("configs/herringbone_duct_strong.yaml"))
base['compute']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
base['immersed']['kind'] = 'koch_herringbone'; base['immersed']['r'] = 3.0; base['immersed']['koch_amp'] = 1.0
base['scalar']['init_type'] = 'interface_y'; base['scalar']['interface_pos'] = 0.5   # FLAT interface
times = [0.0, 1.0, 4.0, 12.0, 29.0]
Ly, Lz = base['domain']['Ly'], base['domain']['Lz']

def run(wallN):
    c = copy.deepcopy(base); c['immersed']['N'] = wallN
    f = tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False); yaml.safe_dump(c, f); f.close()
    sim = ChannelFlow(f.name); nx, ny, nz = sim.nx, sim.ny, sim.nz; ix = nx//2
    solid = sim.chi_c[1+ix, 1:ny+1, 1:nz+1].cpu().numpy() > 0.5
    targets = [int(t/sim.dt) for t in times]
    def grab():
        cc = sim.scalar[1+ix, 1:ny+1, 1:nz+1].detach().cpu().numpy().copy(); cc[solid] = np.nan; return cc
    frames = [grab()]
    for n in range(1, max(targets)+1):
        sim.step_imex(sim.dt)
        if n in targets: frames.append(grab())
    return frames

f0 = run(0); f2 = run(2)
fig, ax = plt.subplots(2, len(times), figsize=(2.5*len(times), 4.8))
for row, (fr, lab) in enumerate([(f0, r"smooth ($N=0$)"), (f2, r"fractal ($N=2$)")]):
    for col, (img, t) in enumerate(zip(fr, times)):
        a = ax[row, col]
        a.imshow(img.T, origin='lower', extent=[0,Ly,0,Lz], aspect='auto', cmap=cmap, vmin=0, vmax=1)
        if row == 0: a.set_title(rf"$t={t:g}$")
        if col == 0: a.set_ylabel(lab+r"\\ $z$" if usetex else lab)
        if row == 1: a.set_xlabel(r"$y$")
        a.set_xticks([]); a.set_yticks([])
fig.suptitle(r"Fractal inlet \emph{surface}: flat interface, smooth vs fractal wall (Sc$=16$)")
fig.tight_layout(rect=[0,0,1,0.95])
os.makedirs("results/figures", exist_ok=True)
fig.savefig("results/figures/snapshots_fractal_wall.png", dpi=140, bbox_inches='tight')
print("wrote results/figures/snapshots_fractal_wall.png")
