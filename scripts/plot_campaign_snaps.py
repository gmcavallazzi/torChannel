"""Inlet (y,z) scalar cross-sections of the round baffle pipe, from the campaign
checkpoints. One row per generation N: the real injected interface (checkpoint
scalar[0]), gouraud-interpolated and CLIPPED to the analytic pipe circle (R=0.42) so
everything outside the pipe (solid corners) is masked. usetex on."""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

usetex = os.environ.get("TORCHANNEL_USETEX", "1") == "1"
plt.rcParams.update({"text.usetex": usetex, "font.family": "serif", "font.size": 10})

# pipe geometry from mixing_campaign.base_config('baffle'): inscribed disc in unit square
R, CY, CZ = 0.42, 0.5, 0.5
LY = LZ = 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='baffle')
    ap.add_argument('--Sc', type=float, default=10.0)
    ap.add_argument('--Ns', type=int, nargs='+', default=[0, 1, 2, 3])
    ap.add_argument('--indir', default='results/campaign')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    th = np.linspace(0, 2 * np.pi, 400)
    yb, zb = CY + R * np.cos(th), CZ + R * np.sin(th)
    circle = np.column_stack([yb, zb])

    rows = []
    for N in a.Ns:
        fp = os.path.join(a.indir, f"{a.mode}_Sc{int(a.Sc)}_N{N}_final.npz")
        if not os.path.exists(fp):
            continue
        rows.append((N, np.load(fp, allow_pickle=True)['scalar'][0, :, :]))   # inlet (y,z)
    if not rows:
        print("no final.npz checkpoints found"); return

    nr = len(rows)
    fig, axs = plt.subplots(nr, 1, figsize=(2.6, 2.6 * nr), squeeze=False)
    for ri, (N, cc) in enumerate(rows):
        ny, nz = cc.shape
        yc = (np.arange(ny) + 0.5) * LY / ny
        zc = (np.arange(nz) + 0.5) * LZ / nz
        Yc, Zc = np.meshgrid(yc, zc, indexing='ij')
        ax = axs[ri][0]
        pcm = ax.pcolormesh(Yc, Zc, cc, shading='gouraud', cmap='RdBu_r', vmin=0, vmax=1)
        clip = PathPatch(Path(circle), transform=ax.transData,
                         facecolor='none', edgecolor='none')
        ax.add_patch(clip); pcm.set_clip_path(clip)
        ax.plot(yb, zb, color='k', lw=0.8)
        ax.set_aspect('equal'); ax.set_xlim(0, LY); ax.set_ylim(0, LZ)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(r"$N=%d$" % N, fontsize=12)

    fig.subplots_adjust(right=0.82)
    cax = fig.add_axes([0.86, 0.15, 0.04, 0.7])
    fig.colorbar(pcm, cax=cax, label=r"$c$")
    out = a.out or f"results/figures/campaign_{a.mode}_Sc{int(a.Sc)}_inlet.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print("wrote", out)


if __name__ == "__main__":
    main()
