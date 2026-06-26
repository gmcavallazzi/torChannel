"""Faithful fractal-surface test on a CIRCULAR orifice (developing duct, inflow->outflow).

Combines the two proposal ingredients on a round cross-section:
  (1) the fractal inlet SURFACE: the pipe wall is Koch-corrugated (area-balanced,
      azimuthal, generation N) and the corrugation is localised at the inlet; and
  (2) the Koch BAFFLE: the scalar interface at the inlet is itself Koch-folded (gen N).
Both scale together with N. Mixing is measured DOWNSTREAM as the fluid-masked
cross-sectional segregation M(x). If MERGE Eq.4 holds, M should collapse much faster
in x for higher N (L_mix(N)/L_mix(0) ~ r^{-Df N}); the null is an N-insensitive M(x)."""
import os, sys, copy, tempfile, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_JIT", "0")
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from solver import ChannelFlow
from utils import compute_divergence

usetex = os.environ.get("TORCHANNEL_USETEX", "1") == "1"
plt.rcParams.update({"text.usetex": usetex, "font.family": "serif", "font.size": 10})

Lx, Ly, Lz = 10.0, 1.0, 1.0
BASE = dict(
    grid={'nx': 96, 'ny': 48, 'nz': 48},
    domain={'Lx': Lx, 'Ly': Ly, 'Lz': Lz, 'bc_y': 'wall', 'bc_x': 'inout'},
    flow={'Re': 40.0, 'Re_tau': 10.0, 'U_bulk': 1.0, 'gamma': 1.0},
    boundary_conditions={'top_wall': {'type': 'dirichlet'}},
    initialization={'type': 'parabolic', 'perturbation_intensity': 0.0},
    solver={'type': 'fft'},
    time={'dt': 0.002, 'n_steps': 1, 't_max': 1e9, 'CFL_target': 0.25, 'dt_update_interval': 0,
          'dt_max': 0.006, 'dt_min': 0.0001, 'scheme': 'IMEX'},
    compute={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
    output={'results_folder': '/tmp/torchannel_pipe_koch', 'n_out': 10**9, 'n_save': 10**9},
    statistics={'enabled': False, 'n_stats': 0},
    immersed={'enabled': True, 'kind': 'pipe_koch', 'eta': 1.0e-4,
              'pipe_R': 0.42, 'pipe_yc': 0.5, 'pipe_zc': 0.5,
              'N': 0, 'r': 3.0, 'koch_amp': 0.15, 'n_lobes': 1, 'inlet_len': 1.0},
    scalar={'enabled': True, 'Sc': 0.5, 'wall_bc': 'neumann', 'scheme': 'central',
            'init_type': 'koch', 'N': 0, 'r': 3.0, 'eps_cells': 1.0, 'theta': 0.5})
NSTEPS = 9000


def Mx(sim):
    """Fluid-masked segregation M(x): cross-sectional std(c)/std_max per x-slice."""
    nx, ny, nz = sim.nx, sim.ny, sim.nz
    c = sim.scalar[1:nx+1, 1:ny+1, 1:nz+1]
    fl = (1.0 - sim.chi_c[1:nx+1, 1:ny+1, 1:nz+1]) if sim.immersed_enabled \
        else torch.ones_like(c)
    wz = sim.dz_f[0:nz].view(1, 1, -1)
    vol = (fl * wz).clamp(min=0)
    A = vol.sum(dim=(1, 2)).clamp(min=1e-30)
    mean = (c * vol).sum(dim=(1, 2)) / A
    var = (((c - mean.view(-1, 1, 1))**2) * vol).sum(dim=(1, 2)) / A
    std = torch.sqrt(var.clamp(min=0))
    std_max = torch.sqrt((mean * (1 - mean)).clamp(min=1e-30))
    return (std / std_max).detach().cpu().numpy()


def run(N):
    c = copy.deepcopy(BASE)
    c['immersed']['N'] = N
    c['scalar']['N'] = N
    f = tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False); yaml.safe_dump(c, f); f.close()
    sim = ChannelFlow(f.name)
    for _ in range(NSTEPS):
        sim.step_imex(sim.dt)
    nx, ny, nz = sim.nx, sim.ny, sim.nz
    div = float(torch.max(torch.abs(compute_divergence(sim.u, sim.v, sim.w, nx, ny, nz, sim.dx, sim.dy, sim.dz_f))))
    cf = sim.scalar[1:nx+1, 1:ny+1, 1:nz+1]
    fl = (1.0 - sim.chi_c[1:nx+1, 1:ny+1, 1:nz+1]).bool()
    cmin = float(cf[fl].min()); cmax = float(cf[fl].max())
    M = Mx(sim)
    mid = sim.scalar[1:nx+1, 1:ny+1, nz//2].detach().cpu().numpy()        # (x,y) midplane
    solid_mid = sim.chi_c[1:nx+1, 1:ny+1, nz//2].detach().cpu().numpy().astype(bool)
    print(f"  N={N}: div={div:.2e}  c_fluid in [{cmin:.3e},{cmax:.4f}]  "
          f"M(in)={M[0]:.3f} M(out)={M[-1]:.3f}", flush=True)
    return mid, solid_mid, M


def main():
  print("=== circular fractal-surface + Koch-baffle mixing (Re=40, Sc=0.5) ===", flush=True)
  results = {N: run(N) for N in (0, 1, 2)}
  xg = np.linspace(0, Lx, len(results[0][2]))

  fig, axs = plt.subplots(4, 1, figsize=(8.5, 7.4),
                          gridspec_kw={'height_ratios': [1, 1, 1, 1.2]})
  for ax, N in [(axs[0], 0), (axs[1], 1), (axs[2], 2)]:
      mid, solid_mid, _ = results[N]
      m = np.ma.array(mid, mask=solid_mid)
      ax.imshow(m.T, origin='lower', extent=[0, Lx, 0, Ly], aspect='auto',
                cmap='RdBu_r', vmin=0, vmax=1)
      ax.set_facecolor('0.6')
      ax.set_ylabel((r"$N=%d$\\ $y$" % N) if usetex else f"N={N} y"); ax.set_xticks([])
  axs[2].set_xlabel(r"streamwise $x$ (inlet $\to$ outlet)")
  for N, lw in [(0, 1.8), (1, 1.8), (2, 1.8)]:
      axs[3].plot(xg, results[N][2], lw=lw, label=r"$N=%d$" % N)
  axs[3].set_xlabel(r"$x$"); axs[3].set_ylabel(r"segregation $M(x)$")
  axs[3].set_title(r"Streamwise mixing (circular orifice)")
  axs[3].grid(True, alpha=0.3); axs[3].legend()
  axs[0].set_title(r"Fractal-wall + Koch-baffle $c(x,y)$ at mid-$z$ (round duct, inflow$\to$outflow)")
  fig.tight_layout()
  os.makedirs("results/figures", exist_ok=True)
  fig.savefig("results/figures/pipe_koch_mixing.png", dpi=140, bbox_inches='tight')
  print("wrote results/figures/pipe_koch_mixing.png", flush=True)


if __name__ == "__main__":
  main()
