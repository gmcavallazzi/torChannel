"""Cross-sectional (y,z) scalar snapshots developing downstream in a circular orifice
with the fractal inlet WALL surface + Koch baffle. Rows = generation N (0 vs 2),
columns = streamwise stations x.

Rendered on the TRUE (stretched-z) physical coordinates so the section is round (not
an index-grid ellipse); the fluid field is gouraud-interpolated for smoothness and
CLIPPED to the analytic boundary, which removes the cell-staircase and draws the
intended circle / Koch-fractal wall crisply."""
import os, sys, copy, tempfile, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_JIT", "0")
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from solver import ChannelFlow
from immersed import _koch_zigzag_disp
import scripts.run_pipe_koch_mixing as RP

usetex = os.environ.get("TORCHANNEL_USETEX", "1") == "1"
plt.rcParams.update({"text.usetex": usetex, "font.family": "serif", "font.size": 10})

X_STATIONS = [0.15, 0.5, 1.0, 2.0, 4.0, 8.0]
NSTEPS = 9000
Ly, Lz = RP.Ly, RP.Lz
IB = RP.BASE['immersed']
R, cy, cz = IB['pipe_R'], IB['pipe_yc'], IB['pipe_zc']
amp0, r, n_lobes, inlet_len = IB['koch_amp'] * R, IB['r'], IB['n_lobes'], IB['inlet_len']


def boundary(N, x):
    """Analytic wall boundary (y,z) at streamwise station x for generation N."""
    th = np.linspace(0, 2 * np.pi, 4001)
    tn = ((th + np.pi) / (2 * np.pi) * n_lobes) % 1.0
    d = _koch_zigzag_disp(N, r, tn)
    m = np.max(np.abs(d))
    dhat = d / m if m > 0 else d
    env = 0.5 * (1.0 + np.cos(np.pi * min(x, inlet_len) / inlet_len)) if x < inlet_len else 0.0
    Rw = R + amp0 * env * dhat
    return cy + Rw * np.cos(th), cz + Rw * np.sin(th)


def run_sections(N):
    c = copy.deepcopy(RP.BASE); c['immersed']['N'] = N; c['scalar']['N'] = N
    f = tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False); yaml.safe_dump(c, f); f.close()
    sim = ChannelFlow(f.name)
    for _ in range(NSTEPS):
        sim.step_imex(sim.dt)
    nx, ny, nz = sim.nx, sim.ny, sim.nz
    dy = sim.dy
    yc = (np.arange(ny) + 0.5) * dy                       # true fluid-cell centres in y
    zc = sim.z_c[1:nz + 1].detach().cpu().numpy()         # true (stretched) z centres
    secs = []
    for xs in X_STATIONS:
        i = int(min(nx, max(1, round(xs / sim.dx))))
        cc = sim.scalar[i, 1:ny + 1, 1:nz + 1].detach().cpu().numpy()   # full field (incl. solid)
        secs.append(cc)
    return yc, zc, secs


print("=== developing (y,z) cross-sections, round fractal orifice ===", flush=True)
data = {N: run_sections(N) for N in (0, 2)}

nrows, ncols = 2, len(X_STATIONS)
fig, axs = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 4.4))
for ri, N in enumerate((0, 2)):
    yc, zc, secs = data[N]
    Yc, Zc = np.meshgrid(yc, zc, indexing='ij')
    for ci, (xs, cc) in enumerate(zip(X_STATIONS, secs)):
        ax = axs[ri, ci]
        pcm = ax.pcolormesh(Yc, Zc, cc, shading='gouraud', cmap='RdBu_r', vmin=0, vmax=1)
        yb, zb = boundary(N, xs)
        clip = PathPatch(Path(np.column_stack([yb, zb])), transform=ax.transData,
                         facecolor='none', edgecolor='none')
        ax.add_patch(clip); pcm.set_clip_path(clip)
        ax.plot(yb, zb, color='k', lw=0.7)                       # true local boundary
        if N > 0:                                                # inlet fractal footprint (ghost ref)
            yf, zf = boundary(N, 0.0)
            ax.plot(yf, zf, color='0.35', lw=0.6, ls='--', alpha=0.7)
        ax.set_aspect('equal'); ax.set_xlim(0, Ly); ax.set_ylim(0, Lz)
        ax.set_xticks([]); ax.set_yticks([])
        if ri == 0:
            ax.set_title(r"$x=%.2f$" % xs)
        if ci == 0:
            ax.set_ylabel(r"$N=%d$" % N, fontsize=12)
fig.suptitle(r"Cross-sections $c(y,z)$", y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.97])
os.makedirs("results/figures", exist_ok=True)
fig.savefig("results/figures/pipe_koch_xsections.png", dpi=150, bbox_inches='tight')
print("wrote results/figures/pipe_koch_xsections.png", flush=True)
