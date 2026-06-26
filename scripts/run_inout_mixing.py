"""Developing-duct (inflow/outflow) Koch-interface mixing: the faithful SPATIAL baffle.
Interface prescribed at the inlet, mixing measured DOWNSTREAM (M as a function of x).
Visualises the (x,y) midplane for N=0 vs N=2 and the streamwise segregation M(x)."""
import os, sys, copy, tempfile, yaml
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_JIT", "0")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from solver import ChannelFlow
from utils import compute_divergence

usetex = os.environ.get("TORCHANNEL_USETEX", "1") == "1"
plt.rcParams.update({"text.usetex": usetex, "font.family": "serif", "font.size": 10})

BASE = dict(
    grid={'nx': 64, 'ny': 32, 'nz': 32},
    domain={'Lx': 10.0, 'Ly': 1.0, 'Lz': 1.0, 'bc_y': 'wall', 'bc_x': 'inout'},
    flow={'Re': 40.0, 'Re_tau': 10.0, 'U_bulk': 1.0, 'gamma': 1.0},
    boundary_conditions={'top_wall': {'type': 'dirichlet'}},
    initialization={'type': 'parabolic', 'perturbation_intensity': 0.0},
    solver={'type': 'fft'},
    time={'dt': 0.003, 'n_steps': 1, 't_max': 1e9, 'CFL_target': 0.25, 'dt_update_interval': 0,
          'dt_max': 0.01, 'dt_min': 0.0001, 'scheme': 'IMEX'},
    compute={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
    output={'results_folder': '/tmp/torchannel_inout_mix', 'n_out': 10**9, 'n_save': 10**9},
    statistics={'enabled': False, 'n_stats': 0},
    scalar={'enabled': True, 'Sc': 0.5, 'wall_bc': 'neumann', 'scheme': 'central',
            'init_type': 'koch', 'N': 0, 'r': 3.0, 'eps_cells': 1.0, 'theta': 0.5})
NSTEPS = 6000

def Mx(sim):
    """Segregation M as a function of streamwise x: std(c) per cross-section / std_max."""
    nx, ny, nz = sim.nx, sim.ny, sim.nz
    c = sim.scalar[1:nx+1, 1:ny+1, 1:nz+1]
    wz = sim.dz_f[0:nz].view(1, 1, -1)
    vol = wz.expand_as(c)
    mean = (c * vol).sum(dim=(1, 2)) / vol.sum(dim=(1, 2))
    var = (((c - mean.view(-1, 1, 1))**2) * vol).sum(dim=(1, 2)) / vol.sum(dim=(1, 2))
    std = torch.sqrt(var.clamp(min=0))
    std_max = torch.sqrt((mean * (1 - mean)).clamp(min=1e-30))
    return (std / std_max).detach().cpu().numpy()

def run(N):
    c = copy.deepcopy(BASE); c['scalar']['N'] = N
    f = tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False); yaml.safe_dump(c, f); f.close()
    sim = ChannelFlow(f.name)
    for _ in range(NSTEPS):
        sim.step_imex(sim.dt)
    nx, ny, nz = sim.nx, sim.ny, sim.nz
    div = float(torch.max(torch.abs(compute_divergence(sim.u, sim.v, sim.w, nx, ny, nz, sim.dx, sim.dy, sim.dz_f))))
    cmin = float(sim.scalar[1:nx+1, 1:ny+1, 1:nz+1].min())
    cmax = float(sim.scalar[1:nx+1, 1:ny+1, 1:nz+1].max())
    mid = sim.scalar[1:nx+1, 1:ny+1, nz//2].detach().cpu().numpy()   # (x,y) midplane
    print(f"  N={N}: div={div:.2e}  c in [{cmin:.3e},{cmax:.4f}]  M(in)={Mx(sim)[0]:.3f} M(out)={Mx(sim)[-1]:.3f}", flush=True)
    return mid, Mx(sim)

print("=== developing-duct Koch mixing (Re=40, Sc=0.5) ===", flush=True)
mid0, M0 = run(0)
mid2, M2 = run(2)
Lx = BASE['domain']['Lx']; Ly = BASE['domain']['Ly']
xg = np.linspace(0, Lx, len(M0))

fig, axs = plt.subplots(3, 1, figsize=(8.5, 6.2), gridspec_kw={'height_ratios': [1, 1, 1.1]})
for ax, mid, lab in [(axs[0], mid0, r"$N=0$"), (axs[1], mid2, r"$N=2$")]:
    ax.imshow(mid.T, origin='lower', extent=[0, Lx, 0, Ly], aspect='auto', cmap='RdBu_r', vmin=0, vmax=1)
    ax.set_ylabel(lab + r"\\ $y$" if usetex else f"{lab} y"); ax.set_xticks([])
axs[1].set_xlabel(r"streamwise $x$ (inlet $\to$ outlet)")
axs[2].plot(xg, M0, lw=1.8, label=r"$N=0$"); axs[2].plot(xg, M2, lw=1.8, label=r"$N=2$")
axs[2].set_xlabel(r"$x$"); axs[2].set_ylabel(r"segregation $M(x)$")
axs[2].set_title(r"Streamwise mixing"); axs[2].grid(True, alpha=0.3); axs[2].legend()
axs[0].set_title(r"Koch interface $c(x,y)$ at mid-$z$ (developing duct, inflow$\to$outflow)")
fig.tight_layout()
os.makedirs("results/figures", exist_ok=True)
fig.savefig("results/figures/inout_mixing.png", dpi=140, bbox_inches='tight')
print("wrote results/figures/inout_mixing.png", flush=True)
