"""
Standard 4-cut snapshot figure from a saved 3D field (fields*.npz).

Layout (Giorgio's preferred default for snapshot requests):
  row 1: x-z cut of u at mid-span (canopy tip line marked)
  row 2: y-z cut of u at mid-x
  row 3: two x-y cuts side by side — mid-canopy (left) and above the tips (right)
Sequential colormap, robust percentile color limits shared by all panels,
gouraud-interpolated shading, usetex (module load texlive first on HPC).

Usage:
    python scripts/plot_snapshot.py results_canopy_monti/fields.npz
    python scripts/plot_snapshot.py fields.npz --h 0.25 --z-cuts 0.125 0.30 \
        --out figures_local/snap.png
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 11})


def main():
    ap = argparse.ArgumentParser(description='4-cut snapshot figure from fields npz')
    ap.add_argument('fields', help='path to fields*.npz')
    ap.add_argument('--h', type=float, default=0.25, help='canopy height to mark (0 = no line)')
    ap.add_argument('--z-cuts', type=float, nargs=2, default=None,
                    help='heights of the two x-y cuts (default: h/2 and h+0.05)')
    ap.add_argument('--out', default=None, help='output png (default: figures_local/<name>_cuts.png)')
    ap.add_argument('--field', default='u', choices=['u', 'v', 'w'], help='component to plot')
    ap.add_argument('--y-cut', type=float, default=None,
                    help='y location of the x-z cut (default: mid-span)')
    args = ap.parse_args()

    d = np.load(args.fields)
    f, z_c = d[args.field], d['z_c']
    step, t = int(d['step']), float(d['time'])
    Lx, Ly = float(d['Lx']), float(d['Ly'])
    # interior sizes from the ghosted staggered shapes
    if args.field == 'u':
        nx, ny, nz = f.shape[0] - 1, f.shape[1] - 2, f.shape[2] - 2
    elif args.field == 'v':
        nx, ny, nz = f.shape[0] - 2, f.shape[1] - 1, f.shape[2] - 2
    else:
        nx, ny, nz = f.shape[0] - 2, f.shape[1] - 2, f.shape[2] - 1

    fi = f[1:nx + 1, 1:ny + 1, 1:nz + 1]
    x = np.arange(nx) * Lx / nx
    y = (np.arange(ny) + 0.5) * Ly / ny
    zc = z_c[1:nz + 1] if args.field != 'w' else d['z_f'][1:nz + 1]

    z1, z2 = args.z_cuts if args.z_cuts else (0.5 * args.h, args.h + 0.05)
    k1 = int(np.argmin(np.abs(zc - z1)))
    k2 = int(np.argmin(np.abs(zc - z2)))
    j_mid = ny // 2 if args.y_cut is None else int(np.argmin(np.abs(y - args.y_cut)))
    i_mid = nx // 2

    vmin, vmax = np.percentile(fi, 0.5), np.percentile(fi, 99.5)
    cmap, shade = 'viridis', 'gouraud'
    lbl = rf'${args.field}/U_b$'

    fig = plt.figure(figsize=(12.0, 10.0), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.45])

    ax = fig.add_subplot(gs[0, :])
    p = ax.pcolormesh(x, zc, fi[:, j_mid, :].T, shading=shade, cmap=cmap, vmin=vmin, vmax=vmax)
    if args.h > 0:
        ax.axhline(args.h, color='w', ls='--', lw=0.8)
    ax.set_xlabel(r'$x/H$'); ax.set_ylabel(r'$z/H$')
    ax.set_title(lbl + rf', $x$--$z$ at $y/H={y[j_mid]:.2f}$ (step {step}, $t={t:.2f}$)')
    fig.colorbar(p, ax=ax, pad=0.01)

    ax = fig.add_subplot(gs[1, :])
    p = ax.pcolormesh(y, zc, fi[i_mid, :, :].T, shading=shade, cmap=cmap, vmin=vmin, vmax=vmax)
    if args.h > 0:
        ax.axhline(args.h, color='w', ls='--', lw=0.8)
    ax.set_xlabel(r'$y/H$'); ax.set_ylabel(r'$z/H$')
    ax.set_title(lbl + rf', $y$--$z$ at $x/H={x[i_mid]:.2f}$')
    fig.colorbar(p, ax=ax, pad=0.01)

    for col, k, tag in ((0, k1, 'mid-canopy'), (1, k2, 'above tips')):
        ax = fig.add_subplot(gs[2, col])
        p = ax.pcolormesh(x, y, fi[:, :, k].T, shading=shade, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel(r'$x/H$'); ax.set_ylabel(r'$y/H$')
        ax.set_title(lbl + rf' at $z/H={zc[k]:.3f}$ ({tag})')
        ax.set_aspect('equal')
        fig.colorbar(p, ax=ax, pad=0.02, shrink=0.9)

    out = args.out
    if out is None:
        base = os.path.splitext(os.path.basename(args.fields))[0]
        os.makedirs('figures_local', exist_ok=True)
        out = f'figures_local/{base}_step{step:06d}_cuts.png'
    fig.savefig(out, dpi=170, bbox_inches='tight')
    print(f'saved {out} | clim [{vmin:.2f}, {vmax:.2f}]')


if __name__ == '__main__':
    main()
