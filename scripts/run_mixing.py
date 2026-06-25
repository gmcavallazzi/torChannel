"""Mixing-decay driver: evolve a Koch-interface scalar and measure L_mix.

For each Koch generation N, run the channel solver (temporal-mixing framing:
periodic box, no-flux scalar walls), record the intensity-of-segregation
M(t) = std(c)/std_max, and report the mixing time t_mix (first M < threshold)
and the mixing length L_mix = U_bulk * t_mix.

Usage:
    python scripts/run_mixing.py configs/koch_mix_demo.yaml --Ns 0 1 2 --nsteps 1200

This is the temporal-mixing analogue of the proposal's downstream mixing length.
In a PLAIN laminar channel the flow is unidirectional and the interface is
homogeneous in x, so mixing is diffusion-limited (a 3D cross-check of the
diffusive-limit result); the Re-dependent advective mechanism requires the
corrugated wall (immersed boundary) or a turbulent field.
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
from scalar import scalar_stats

# system LaTeX for all figure text
plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                     "font.serif": ["Computer Modern Roman"],
                     "axes.labelsize": 12, "axes.titlesize": 12, "legend.fontsize": 9})


def run_one(base_cfg, N, nsteps, sample_every, results_root):
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault('scalar', {})
    cfg['scalar'].update({'enabled': True, 'init_type': 'koch', 'N': N})
    cfg['output'] = dict(cfg.get('output', {}))
    cfg['output']['results_folder'] = os.path.join(results_root, f'N{N}')
    os.makedirs(cfg['output']['results_folder'], exist_ok=True)

    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
        yaml.safe_dump(cfg, f)
        cfg_path = f.name

    flow = ChannelFlow(cfg_path)
    U = flow.U_bulk
    ts, Ms = [0.0], [scalar_stats(flow.scalar, flow.nx, flow.ny, flow.nz, flow.dz_f)['M']]
    for n in range(1, nsteps + 1):
        flow.step_imex(flow.dt)
        flow.time += flow.dt
        if n % sample_every == 0:
            M = scalar_stats(flow.scalar, flow.nx, flow.ny, flow.nz, flow.dz_f)['M']
            ts.append(flow.time); Ms.append(M)
            if torch.isnan(flow.scalar).any():
                print(f"  N={N}: NaN at step {n}", flush=True); break
    ts, Ms = np.array(ts), np.array(Ms)

    # mixing time: first crossing of M = thresh (linear interpolation)
    thresh = flow.Mthresh if hasattr(flow, 'Mthresh') else 0.05
    t_mix = float('nan')
    for i in range(1, len(Ms)):
        if Ms[i-1] > thresh >= Ms[i]:
            frac = (Ms[i-1] - thresh) / (Ms[i-1] - Ms[i])
            t_mix = ts[i-1] + frac * (ts[i] - ts[i-1]); break
    L_mix = U * t_mix
    print(f"  N={N}: final M={Ms[-1]:.4f}  t_mix={t_mix:.3f}  L_mix={L_mix:.3f}", flush=True)
    return ts, Ms, t_mix, L_mix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--Ns', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--nsteps', type=int, default=1200)
    ap.add_argument('--sample_every', type=int, default=20)
    ap.add_argument('--out', default='/tmp/torchannel_mixing')
    args = ap.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)
    os.makedirs(args.out, exist_ok=True)

    results = {}
    for N in args.Ns:
        print(f"=== running N={N} ===", flush=True)
        results[N] = run_one(base_cfg, N, args.nsteps, args.sample_every, args.out)

    # figure: M(t) decay per N
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    thresh = 0.05
    for N in args.Ns:
        ts, Ms, t_mix, L_mix = results[N]
        ax.semilogy(ts, Ms, 'o-', ms=3, lw=1.6,
                    label=rf"$N={N}$ ($L_{{\mathrm{{mix}}}}={L_mix:.1f}$)")
    ax.axhline(thresh, ls=':', color='k', lw=1)
    ax.text(ax.get_xlim()[1]*0.98, thresh*1.1, r"$M=0.05$", ha='right', fontsize=8)
    ax.set_xlabel(r"time $t$  ($L_{\mathrm{mix}} = U_{\mathrm{bulk}}\,t_{\mathrm{mix}}$)")
    ax.set_ylabel(r"segregation $M(t) = \sigma_c/\sigma_{\max}$")
    re = base_cfg['flow']['Re']
    sc = base_cfg.get('scalar', {}).get('Sc', 1.0)
    ax.set_title(rf"Koch-interface mixing decay (laminar, $\mathrm{{Re}}={re:g}$, "
                 rf"$\mathrm{{Sc}}={sc:g}$)")
    ax.legend(); ax.grid(True, which='both', alpha=0.3)
    fig_path = os.path.join(args.out, 'mixing_decay.png')
    fig.savefig(fig_path, dpi=130, bbox_inches='tight')
    print(f"wrote {fig_path}", flush=True)

    np.savez(os.path.join(args.out, 'mixing_results.npz'),
             **{f'N{N}_t': results[N][0] for N in args.Ns},
             **{f'N{N}_M': results[N][1] for N in args.Ns},
             Lmix={N: results[N][3] for N in args.Ns})


if __name__ == "__main__":
    main()
