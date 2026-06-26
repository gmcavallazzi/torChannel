"""Cross-sectional (y,z) scalar snapshots developing DOWNSTREAM in a circular orifice
with the fractal inlet WALL surface + Koch baffle. Rows = generation N (0 vs 2),
columns = streamwise stations x. Solid (pipe corners + inlet corrugation) shown grey.
Shows the round section, the Koch-folded interface at the inlet, and its decay in x."""
import os, sys, copy, tempfile, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_JIT", "0")
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from solver import ChannelFlow
import scripts.run_pipe_koch_mixing as RP

usetex = os.environ.get("TORCHANNEL_USETEX", "1") == "1"
plt.rcParams.update({"text.usetex": usetex, "font.family": "serif", "font.size": 10})

X_STATIONS = [0.15, 0.5, 1.0, 2.0, 4.0, 8.0]
NSTEPS = 9000
Ly, Lz = RP.Ly, RP.Lz


def run_sections(N):
    c = copy.deepcopy(RP.BASE); c['immersed']['N'] = N; c['scalar']['N'] = N
    f = tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False); yaml.safe_dump(c, f); f.close()
    sim = ChannelFlow(f.name)
    for _ in range(NSTEPS):
        sim.step_imex(sim.dt)
    nx = sim.nx
    dx = sim.dx
    secs = []
    for xs in X_STATIONS:
        i = int(min(nx, max(1, round(xs / dx))))            # cell index for this station
        cc = sim.scalar[i, 1:sim.ny+1, 1:sim.nz+1].detach().cpu().numpy()
        solid = sim.chi_c[i, 1:sim.ny+1, 1:sim.nz+1].detach().cpu().numpy() > 0.5
        secs.append(np.ma.array(cc, mask=solid))
    return secs


print("=== developing (y,z) cross-sections, round fractal orifice ===", flush=True)
data = {N: run_sections(N) for N in (0, 2)}

nrows, ncols = 2, len(X_STATIONS)
fig, axs = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 4.6))
for ri, N in enumerate((0, 2)):
    for ci, (xs, sec) in enumerate(zip(X_STATIONS, data[N])):
        ax = axs[ri, ci]
        ax.imshow(sec.T, origin='lower', extent=[0, Ly, 0, Lz], aspect='equal',
                  cmap='RdBu_r', vmin=0, vmax=1)
        ax.set_facecolor('0.6')
        ax.set_xticks([]); ax.set_yticks([])
        if ri == 0:
            ax.set_title(r"$x=%.2f$" % xs)
        if ci == 0:
            ax.set_ylabel(r"$N=%d$" % N, fontsize=12)
fig.suptitle(r"Scalar $c(y,z)$ developing downstream (round duct, fractal inlet wall)", y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96])
os.makedirs("results/figures", exist_ok=True)
fig.savefig("results/figures/pipe_koch_xsections.png", dpi=150, bbox_inches='tight')
print("wrote results/figures/pipe_koch_xsections.png", flush=True)
