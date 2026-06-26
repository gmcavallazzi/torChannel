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
from scalar import scalar_stats, scalar_dissipation, apply_scalar_bc

# System LaTeX for all figure text (HPC: `module load texlive` first). Set
# TORCHANNEL_USETEX=0 to fall back to mathtext on machines without a TeX install.
_usetex = os.environ.get("TORCHANNEL_USETEX", "1") == "1"
plt.rcParams.update({"text.usetex": _usetex, "font.family": "serif",
                     "font.serif": ["Computer Modern Roman"], "mathtext.fontset": "cm",
                     "axes.labelsize": 12, "axes.titlesize": 12, "legend.fontsize": 9})


def run_one(base_cfg, N, nsteps, sample_every, results_root, thresh=0.05):
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
    chi_c = flow.chi_c if getattr(flow, 'immersed_enabled', False) else None

    # Report the cell-Peclet number (U*dx/D): the false-diffusion / dispersive-wiggle
    # control parameter. Our scalar advection is 2nd-order central (zero false
    # diffusion, certified) so cell-Pe > 2 risks dispersive over/undershoots, not
    # artificial mixing; we watch min/max(c) for wiggles below.
    D = flow.scalar_D
    dz_min = float(flow.dz_f.min())
    cell_pe = U * max(flow.dx, flow.dy, dz_min) / D
    print(f"  N={N}: D={D:.2e}, cell-Pe={cell_pe:.1f} "
          f"(dx={flow.dx:.3f}, dz_min={dz_min:.3f})", flush=True)

    bc_y = getattr(flow, 'bc_y', 'periodic')
    wall_bc = getattr(flow, 'scalar_wall_bc', 'neumann')

    def M_of(c):
        return scalar_stats(c, flow.nx, flow.ny, flow.nz, flow.dz_f, chi_c=chi_c)['M']

    def chi_of(c):
        # area-sensitive observable <|grad c|^2>; needs up-to-date ghosts
        apply_scalar_bc(c, wall_bc, bc_y)
        return scalar_dissipation(c, flow.nx, flow.ny, flow.nz,
                                  flow.dx, flow.dy, flow.dz_f, chi_c=chi_c)

    ts, Ms, chis = [0.0], [M_of(flow.scalar)], [chi_of(flow.scalar)]
    cmin, cmax = 1.0, 0.0
    for n in range(1, nsteps + 1):
        flow.step_imex(flow.dt)
        flow.time += flow.dt
        if n % sample_every == 0:
            ts.append(flow.time); Ms.append(M_of(flow.scalar)); chis.append(chi_of(flow.scalar))
            cmin = min(cmin, float(flow.scalar.min()))
            cmax = max(cmax, float(flow.scalar.max()))
            if torch.isnan(flow.scalar).any():
                print(f"  N={N}: NaN at step {n}", flush=True); break
    ts, Ms, chis = np.array(ts), np.array(Ms), np.array(chis)
    overshoot = max(0.0, cmax - 1.0) + max(0.0, -cmin)
    print(f"  N={N}: c in [{cmin:.3f}, {cmax:.3f}]  overshoot={overshoot:.2e}"
          f"{'  <-- WIGGLES (consider TVD)' if overshoot > 0.05 else ''}", flush=True)

    # mixing time: first crossing of M = thresh (linear interpolation)
    t_mix = float('nan')
    for i in range(1, len(Ms)):
        if Ms[i-1] > thresh >= Ms[i]:
            frac = (Ms[i-1] - thresh) / (Ms[i-1] - Ms[i])
            t_mix = ts[i-1] + frac * (ts[i] - ts[i-1]); break
    L_mix = U * t_mix
    chi0 = chis[1] if len(chis) > 1 else chis[0]   # earliest post-IC sample
    print(f"  N={N}: final M={Ms[-1]:.4f}  t_mix={t_mix:.3f}  L_mix={L_mix:.3f}  "
          f"chi(early)={chi0:.3e}", flush=True)
    return ts, Ms, chis, t_mix, L_mix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--Ns', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--nsteps', type=int, default=1200)
    ap.add_argument('--sample_every', type=int, default=20)
    ap.add_argument('--Sc', type=float, default=None, help='override scalar Schmidt number')
    ap.add_argument('--Re', type=float, default=None, help='override flow Reynolds number')
    ap.add_argument('--dt', type=float, default=None, help='override time step')
    ap.add_argument('--out', default='/tmp/torchannel_mixing')
    ap.add_argument('--thresh', type=float, default=0.05, help='M threshold for t_mix')
    args = ap.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)
    if args.Sc is not None:
        base_cfg.setdefault('scalar', {})['Sc'] = args.Sc
    if args.Re is not None:
        base_cfg.setdefault('flow', {})['Re'] = args.Re
    if args.dt is not None:
        base_cfg.setdefault('time', {})['dt'] = args.dt
    os.makedirs(args.out, exist_ok=True)

    results = {}
    for N in args.Ns:
        print(f"=== running N={N} ===", flush=True)
        results[N] = run_one(base_cfg, N, args.nsteps, args.sample_every, args.out,
                             thresh=args.thresh)

    re = base_cfg['flow']['Re']
    sc = base_cfg.get('scalar', {}).get('Sc', 1.0)
    r = base_cfg.get('scalar', {}).get('r', 3.0)
    Df = np.log(4.0) / np.log(r)   # area-balanced zigzag generator: D_f = log4/log r

    # ---- ratio table: measured vs the proposal's prediction r^{-Df*N} ----
    N0 = args.Ns[0]
    Lmix0 = results[N0][4]
    chi0_ref = results[N0][2][1] if len(results[N0][2]) > 1 else results[N0][2][0]
    print(f"\n=== RESULT (Re={re:g}, Sc={sc:g}, r={r:g}, Df={Df:.3f}) ===", flush=True)
    print(f"{'N':>3} {'L_mix':>9} {'L_mix(N)/L_mix(0)':>18} {'predicted r^-DfN':>17} "
          f"{'chi_early':>11} {'chi(N)/chi(0)':>13}", flush=True)
    for N in args.Ns:
        ts, Ms, chis, t_mix, L_mix = results[N]
        chi_e = chis[1] if len(chis) > 1 else chis[0]
        lr = L_mix / Lmix0 if np.isfinite(L_mix) and np.isfinite(Lmix0) else float('nan')
        pred = r ** (-Df * N)
        print(f"{N:>3} {L_mix:>9.3f} {lr:>18.3f} {pred:>17.3f} "
              f"{chi_e:>11.3e} {chi_e/chi0_ref:>13.3f}", flush=True)

    # figure: M(t) decay and chi(t) per N
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for N in args.Ns:
        ts, Ms, chis, t_mix, L_mix = results[N]
        ax.semilogy(ts, Ms, 'o-', ms=3, lw=1.6,
                    label=rf"$N={N}$ ($L_{{\mathrm{{mix}}}}={L_mix:.1f}$)")
        ax2.plot(ts, chis, 'o-', ms=3, lw=1.6, label=rf"$N={N}$")
    ax.axhline(args.thresh, ls=':', color='k', lw=1)
    ax.set_xlabel(r"time $t$"); ax.set_ylabel(r"segregation $M(t)=\sigma_c/\sigma_{\max}$")
    ax2.set_xlabel(r"time $t$"); ax2.set_ylabel(r"$\chi=\langle|\nabla c|^2\rangle$")
    ax.set_title(rf"mixing length ($\mathrm{{Re}}={re:g}$, $\mathrm{{Sc}}={sc:g}$)")
    ax2.set_title(r"scalar dissipation (area-sensitive)")
    ax.legend(); ax.grid(True, which='both', alpha=0.3)
    ax2.legend(); ax2.grid(True, alpha=0.3)
    fig_path = os.path.join(args.out, 'mixing_decay.png')
    fig.savefig(fig_path, dpi=130, bbox_inches='tight')
    print(f"wrote {fig_path}", flush=True)

    np.savez(os.path.join(args.out, 'mixing_results.npz'),
             **{f'N{N}_t': results[N][0] for N in args.Ns},
             **{f'N{N}_M': results[N][1] for N in args.Ns},
             **{f'N{N}_chi': results[N][2] for N in args.Ns},
             Lmix={N: results[N][4] for N in args.Ns},
             Re=re, Sc=sc, r=r, Df=Df)


if __name__ == "__main__":
    main()
