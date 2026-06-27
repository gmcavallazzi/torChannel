"""Developing (y,z) scalar cross-sections down the round baffle pipe, from the campaign
checkpoints (no rerun). Rows = generation N, columns = streamwise station x. The field is
gouraud-interpolated and CLIPPED to the analytic pipe circle (R=0.42), so everything
OUTSIDE the pipe (solid corners, where the scalar is unconstrained) is masked away --
same faithful rendering as scripts/snapshot_pipe_xsections.py. usetex on."""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from immersed import _koch_zigzag_disp

usetex = os.environ.get("TORCHANNEL_USETEX", "1") == "1"
plt.rcParams.update({"text.usetex": usetex, "font.family": "serif", "font.size": 10})

# pipe geometry from mixing_campaign.base_config: inscribed disc in unit square
R, CY, CZ = 0.42, 0.5, 0.5
LY = LZ = 1.0
# surface_baffle fractal-wall params (immersed kind='pipe_koch')
KOCH_AMP, R_RATIO, N_LOBES, INLET_LEN = 0.15, 3.0, 6, 1.0


def wall_boundary(mode, N, x):
    """Analytic wall outline (y,z) at streamwise station x for generation N.
    'baffle' -> plain circle. 'surface_baffle' -> Koch-corrugated orifice near the
    inlet (half-cosine streamwise envelope over inlet_len; smooth disc downstream),
    matching immersed kind='pipe_koch' / scripts.snapshot_pipe_xsections."""
    th = np.linspace(0, 2 * np.pi, 2001)
    if mode != 'surface_baffle' or N <= 0:
        return CY + R * np.cos(th), CZ + R * np.sin(th)
    tn = ((th + np.pi) / (2 * np.pi) * N_LOBES) % 1.0
    d = _koch_zigzag_disp(N, R_RATIO, tn)
    m = np.max(np.abs(d))
    dhat = d / m if m > 0 else d
    env = 0.5 * (1.0 + np.cos(np.pi * min(x, INLET_LEN) / INLET_LEN)) if x < INLET_LEN else 0.0
    Rw = R + KOCH_AMP * R * env * dhat
    return CY + Rw * np.cos(th), CZ + Rw * np.sin(th)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='baffle')
    ap.add_argument('--Sc', type=float, default=10.0)
    ap.add_argument('--Ns', type=int, nargs='+', default=[0, 1, 2, 3])
    ap.add_argument('--xs', type=float, nargs='+', default=[0.0, 0.5, 1.0, 2.0, 4.0, 6.0])
    ap.add_argument('--indir', default='results/campaign')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    rows = []
    for N in a.Ns:
        fp = os.path.join(a.indir, f"{a.mode}_Sc{int(a.Sc)}_N{N}_final.npz")
        if not os.path.exists(fp):
            continue
        d = np.load(fp, allow_pickle=True)
        rows.append((N, d['scalar'], float(d['Lx']), d['Mx']))   # scalar (nx,ny,nz), Mx (nx,)
    if not rows:
        print("no final.npz checkpoints found"); return

    nr, nc = len(rows), len(a.xs)
    fig, axs = plt.subplots(nr, nc, figsize=(1.9 * nc, 1.9 * nr + 0.4), squeeze=False)
    for ri, (N, sc, Lx, Mx) in enumerate(rows):
        nx, ny, nz = sc.shape
        dx = Lx / nx
        yc = (np.arange(ny) + 0.5) * LY / ny
        zc = (np.arange(nz) + 0.5) * LZ / nz
        Yc, Zc = np.meshgrid(yc, zc, indexing='ij')
        for ci, xs in enumerate(a.xs):
            ax = axs[ri][ci]
            i = int(min(nx - 1, max(0, round(xs / dx))))
            cc = sc[i, :, :]
            pcm = ax.pcolormesh(Yc, Zc, cc, shading='gouraud', cmap='RdBu_r',
                                vmin=0, vmax=1)
            yb, zb = wall_boundary(a.mode, N, xs)          # faithful per-(N,x) outline
            clip = PathPatch(Path(np.column_stack([yb, zb])), transform=ax.transData,
                             facecolor='none', edgecolor='none')
            ax.add_patch(clip); pcm.set_clip_path(clip)
            ax.plot(yb, zb, color='k', lw=0.7)
            ax.set_aspect('equal'); ax.set_xlim(0, LY); ax.set_ylim(0, LZ)
            ax.set_xticks([]); ax.set_yticks([])
            # tiny M(x) label in the top-right square corner, outside the circle
            ax.text(0.99, 0.99, r"$M{=}%.2f$" % float(Mx[i]), transform=ax.transAxes,
                    ha='right', va='top', fontsize=9)
            if ri == 0:
                ax.set_title(r"$x=%.1f$" % xs)
            if ci == 0:
                ax.set_ylabel(r"$N=%d$" % N, fontsize=12)
    fig.suptitle(r"Cross-sections $c(y,z)$, Sc$=%d$" % int(a.Sc), y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = a.out or f"results/figures/campaign_{a.mode}_Sc{int(a.Sc)}_xsections.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print("wrote", out)


if __name__ == "__main__":
    main()
