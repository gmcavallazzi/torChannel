"""Extra cross-section snapshots: (A) plain-duct pure-diffusion N=0 vs N=2 (no flow),
(B) the herringbone secondary-flow rolls (v,w) that drive the chaotic mixing."""
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
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

def mkcfg(d):
    f = tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False); yaml.safe_dump(d, f); f.close()
    return f.name

# ---------- (A) plain-duct pure diffusion (velocity frozen at 0) ----------
duct = dict(grid={'nx':8,'ny':64,'nz':64}, domain={'Lx':3.1416,'Ly':1.0,'Lz':1.0,'bc_y':'wall'},
    flow={'Re':50.0,'Re_tau':10.0,'U_bulk':1.0,'gamma':1.2},
    boundary_conditions={'top_wall':{'type':'dirichlet'}},
    initialization={'type':'parabolic','perturbation_intensity':0.0}, solver={'type':'fft'},
    time={'dt':0.02,'n_steps':1,'t_max':1e9,'CFL_target':0.25,'dt_update_interval':0,
          'dt_max':0.02,'dt_min':1e-4,'scheme':'IMEX'},
    compute={'device':DEV}, output={'results_folder':'/tmp/snapA','n_out':10**9,'n_save':10**9},
    statistics={'enabled':False,'n_stats':0},
    scalar={'enabled':True,'Sc':16.0,'wall_bc':'neumann','scheme':'tvd','init_type':'koch',
            'N':0,'r':3.0,'eps_cells':1.0,'theta':0.5})
times_d = [0.0, 40.0, 234.0]
def run_diff(N):
    c = copy.deepcopy(duct); c['scalar']['N'] = N
    sim = ChannelFlow(mkcfg(c)); sim.u.zero_(); sim.v.zero_(); sim.w.zero_()
    nx, ny, nz = sim.nx, sim.ny, sim.nz; ix = nx//2
    targets = [int(t/sim.dt) for t in times_d]
    grab = lambda: sim.scalar[1+ix, 1:ny+1, 1:nz+1].detach().cpu().numpy().copy()
    frames = [grab()]
    for n in range(1, max(targets)+1):
        sim.advance_scalar(sim.dt)
        if n in targets: frames.append(grab())
    return frames, sim

Ly, Lz = 1.0, 1.0
f0, sim = run_diff(0); f2, _ = run_diff(2)
fig, ax = plt.subplots(2, len(times_d), figsize=(2.5*len(times_d), 4.8))
for row, (fr, lab) in enumerate([(f0, r"$N=0$"), (f2, r"$N=2$")]):
    for col, (img, t) in enumerate(zip(fr, times_d)):
        a = ax[row, col]
        a.imshow(img.T, origin='lower', extent=[0,Ly,0,Lz], aspect='auto', cmap=cmap, vmin=0, vmax=1)
        if row == 0: a.set_title(rf"$t={t:g}$")
        if col == 0: a.set_ylabel(lab+r"\quad $z$" if usetex else f"{lab} z")
        if row == 1: a.set_xlabel(r"$y$")
        a.set_xticks([]); a.set_yticks([])
fig.suptitle(r"Pure diffusion, no flow (plain duct, Sc$=16$): the fractal just smooths out")
fig.tight_layout(rect=[0,0,1,0.95])
os.makedirs("results/figures", exist_ok=True)
fig.savefig("results/figures/snapshots_diffusion.png", dpi=140, bbox_inches='tight')
print("wrote results/figures/snapshots_diffusion.png")

# ---------- (B) herringbone secondary-flow rolls (v,w) over c ----------
sim = ChannelFlow("configs/herringbone_duct_strong.yaml")
nx, ny, nz = sim.nx, sim.ny, sim.nz; ix = nx//2
for n in range(int(8.0/sim.dt)):   # develop flow + partial mixing to t=8
    sim.step_imex(sim.dt)
solid = sim.chi_c[1+ix,1:ny+1,1:nz+1].cpu().numpy() > 0.5
cc = sim.scalar[1+ix,1:ny+1,1:nz+1].detach().cpu().numpy().copy(); cc[solid] = np.nan
# interpolate v (y-faces), w (z-faces) to cell centres at mid-x
v = sim.v[1+ix].detach().cpu().numpy(); w = sim.w[1+ix].detach().cpu().numpy()
vc = 0.5*(v[1:ny+1,1:nz+1] + v[0:ny,1:nz+1])
wc = 0.5*(w[1:ny+1,1:nz+1] + w[1:ny+1,0:nz])
yc = (np.arange(ny)+0.5)*(Ly/ny); zc = sim.z_c[1:nz+1].cpu().numpy()
YY, ZZ = np.meshgrid(yc, zc, indexing='ij')
vc[solid] = np.nan; wc[solid] = np.nan
s = 4  # subsample arrows
fig2, a = plt.subplots(figsize=(5.2, 4.4))
a.imshow(cc.T, origin='lower', extent=[0,Ly,0,sim.Lz], aspect='auto', cmap=cmap, vmin=0, vmax=1, alpha=0.85)
a.quiver(YY[::s,::s], ZZ[::s,::s], vc[::s,::s], wc[::s,::s], color='k', scale=8, width=0.004)
a.set_xlabel(r"$y$"); a.set_ylabel(r"$z$")
a.set_title(r"Secondary-flow rolls $(v,w)$ over $c$ (strong herringbone, $t=8$)")
a.set_xticks([]); a.set_yticks([])
fig2.tight_layout()
fig2.savefig("results/figures/secondary_flow.png", dpi=140, bbox_inches='tight')
print("wrote results/figures/secondary_flow.png")
