"""Phase-1 milestone: does the oblique-groove wall produce secondary flow?

Runs the corrugated channel (immersed.py grooves) at one or more Reynolds
numbers and reports the transverse kinetic-energy fraction

    f_perp = <v^2 + w^2>_fluid / <u^2>_fluid,

the signature of the helical (y,z) secondary flow that folds a cross-stream
interface. A flat channel gives f_perp = 0 exactly (no transverse velocity);
any f_perp > 0 here is the advective folding mechanism the proposal's Eq. 5
relies on. Also saves a (y,z) secondary-flow quiver at mid-x.

Usage (GPU needs PYTORCH_JIT=0 on the GB10, see memory):
    PYTORCH_JIT=0 python scripts/run_secondary_flow.py configs/corrugated_channel.yaml --Re 10 50 100
"""

import argparse
import copy
import os
import sys
import tempfile

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
torch.set_default_dtype(torch.float64)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from solver import ChannelFlow


def transverse_ke_fraction(flow):
    """Fluid-masked <v^2+w^2>/<u^2> on the staggered components."""
    nx, ny, nz = flow.nx, flow.ny, flow.nz

    def msq(field, chi):
        fi = field[1:nx + 1, 1:ny + 1, 1:nz + 1]
        m = (1.0 - chi[1:nx + 1, 1:ny + 1, 1:nz + 1])
        return float((m * fi ** 2).sum() / m.sum())

    u2 = msq(flow.u, flow.chi_u)
    v2 = msq(flow.v, flow.chi_v)
    w2 = msq(flow.w, flow.chi_w)
    return (v2 + w2) / u2, u2, v2, w2


def run_one(base_cfg, Re, nsteps, out):
    cfg = copy.deepcopy(base_cfg)
    cfg['flow'] = dict(cfg['flow']); cfg['flow']['Re'] = Re
    cfg['time'] = dict(cfg['time']); cfg['time']['n_steps'] = nsteps
    cfg['output'] = dict(cfg.get('output', {}))
    cfg['output']['results_folder'] = os.path.join(out, f'Re{Re:g}')
    os.makedirs(cfg['output']['results_folder'], exist_ok=True)
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as fh:
        yaml.safe_dump(cfg, fh); path = fh.name

    flow = ChannelFlow(path)
    fracs = []
    for n in range(1, nsteps + 1):
        flow.step_imex(flow.dt); flow.time += flow.dt
        if n % 200 == 0:
            f_perp, *_ = transverse_ke_fraction(flow)
            fracs.append((flow.time, f_perp))
    f_perp, u2, v2, w2 = transverse_ke_fraction(flow)
    print(f"  Re={Re:g}: f_perp = {f_perp:.3e}  (<u^2>={u2:.3e}, <v^2>={v2:.2e}, "
          f"<w^2>={w2:.2e})", flush=True)
    return flow, f_perp, np.array(fracs)


def secondary_flow_figure(flow, Re, out):
    """Quiver of (v,w) with u-contour in a (y,z) plane at mid-x."""
    nx, ny, nz = flow.nx, flow.ny, flow.nz
    i = nx // 2
    # interpolate staggered components to cell centres in the y,z plane
    vc = 0.5 * (flow.v[i, 0:ny, 1:nz + 1] + flow.v[i, 1:ny + 1, 1:nz + 1])
    wc = 0.5 * (flow.w[i, 1:ny + 1, 0:nz] + flow.w[i, 1:ny + 1, 1:nz + 1])
    uc = flow.u[i, 1:ny + 1, 1:nz + 1]
    chi = flow.chi_c[i, 1:ny + 1, 1:nz + 1]
    y = (np.arange(ny) + 0.5) * (flow.Ly / ny)
    z = flow.z_c[1:nz + 1].detach().cpu().numpy()
    Y, Z = np.meshgrid(y, z, indexing='ij')
    uc = uc.detach().cpu().numpy(); vc = vc.detach().cpu().numpy()
    wc = wc.detach().cpu().numpy(); chi = chi.detach().cpu().numpy()
    uc = np.ma.masked_where(chi > 0.5, uc)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    cf = ax.contourf(Y, Z, uc, 30, cmap='viridis')
    plt.colorbar(cf, ax=ax, label='u (streamwise)')
    s = max(1, ny // 24)
    ax.quiver(Y[::s, ::s], Z[::s, ::s], vc[::s, ::s], wc[::s, ::s],
              color='w', scale=None, width=0.003)
    ax.contour(Y, Z, chi, levels=[0.5], colors='r', linewidths=1.2)
    ax.set_xlabel('y'); ax.set_ylabel('z')
    ax.set_title(f'Secondary flow (v,w) at mid-x, Re={Re:g}')
    p = os.path.join(out, f'secondary_flow_Re{Re:g}.png')
    fig.savefig(p, dpi=130, bbox_inches='tight'); plt.close(fig)
    print(f"  wrote {p}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--Re', type=float, nargs='+', default=[10, 50, 100])
    ap.add_argument('--nsteps', type=int, default=3000)
    ap.add_argument('--out', default='/tmp/torchannel_corrugated')
    args = ap.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)
    os.makedirs(args.out, exist_ok=True)

    results = {}
    for Re in args.Re:
        print(f"=== Re={Re:g} ===", flush=True)
        flow, f_perp, hist = run_one(base_cfg, Re, args.nsteps, args.out)
        secondary_flow_figure(flow, Re, args.out)
        results[Re] = f_perp

    print("\n--- transverse KE fraction vs Re ---", flush=True)
    for Re in args.Re:
        print(f"  Re={Re:6g}:  f_perp = {results[Re]:.3e}", flush=True)
    nonzero = all(results[Re] > 1e-8 for Re in args.Re)
    print(f"\nSecondary flow present (f_perp > 0 at all Re): {nonzero}", flush=True)


if __name__ == "__main__":
    main()
