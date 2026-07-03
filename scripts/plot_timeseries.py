"""
Plot run diagnostics parsed from the solver's stdout (slurm-*.out):
dt, u_tau (bed), forcing + canopy Fx, and u_tau,tip vs time.

Accepts one or more log files (e.g. across restarts); each subsequent file's
time axis is offset to continue where the previous ended, and restart points
are marked with a vertical line.

Usage:
    python scripts/plot_timeseries.py slurm-canopy-290.out
    python scripts/plot_timeseries.py slurm-canopy-286.out slurm-canopy-290.out \
        --out figures_local/timeseries.png --target-re-tau 1157
"""
import argparse
import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 11})

# step time dt max_div u_bulk u_tau forcing [canopy_Fx u_tau_tip]
ROW = re.compile(r'^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.eE+-]+)\s+([\d.]+)\s+([\d.]+)'
                 r'\s+([\d.eE+-]+)(?:\s+([\d.eE+-]+)\s+([\d.]+))?\s*$')


def parse_log(path):
    cols = {k: [] for k in ('step', 'time', 'dt', 'u_tau', 'forcing', 'Fx', 'u_tau_tip')}
    with open(path) as f:
        for line in f:
            m = ROW.match(line)
            if not m:
                continue
            g = m.groups()
            cols['step'].append(int(g[0]))
            cols['time'].append(float(g[1]))
            cols['dt'].append(float(g[2]))
            cols['u_tau'].append(float(g[5]))
            cols['forcing'].append(float(g[6]))
            cols['Fx'].append(float(g[7]) if g[7] is not None else np.nan)
            cols['u_tau_tip'].append(float(g[8]) if g[8] is not None else np.nan)
    return {k: np.asarray(v) for k, v in cols.items()}


def main():
    ap = argparse.ArgumentParser(description='Plot solver stdout diagnostics')
    ap.add_argument('logs', nargs='+', help='slurm-*.out file(s), oldest first')
    ap.add_argument('--out', default=None, help='output png')
    ap.add_argument('--target-re-tau', type=float, default=1157.0,
                    help='target Re_tau,out for the equilibrium reference lines (0 = off)')
    ap.add_argument('--H', type=float, default=1.0, help='outer height (tip to surface)')
    ap.add_argument('--Re', type=float, default=6000.0, help='bulk Reynolds number (nu = 1/Re)')
    ap.add_argument('--Lx', type=float, default=6.283185307179586)
    ap.add_argument('--Ly', type=float, default=4.71238898038469)
    ap.add_argument('--Lz', type=float, default=1.25, help='total channel height (H + h)')
    ap.add_argument('--u-tau-in-ratio', type=float, default=0.41, dest='r_in',
                    help='ESTIMATED u_tau,in / u_tau,out for the bed and Fx references. '
                         'Default 0.41 = the ratio of the paper\'s validation case '
                         '(Shimizu 1991: Re_tau,in/out = 535/1310, lambda=0.41, h/H=0.65) — '
                         'an analogy, NOT a published value for the lambda=0.35 case')
    ap.add_argument('--all', action='store_true',
                    help='keep every sample (default: skip the first 10 of each log, '
                         'which carry the restart spike)')
    args = ap.parse_args()

    # Equilibrium references from the momentum balance at the target Re_tau,out:
    #   u_tau,out = Re_tau,out * nu / H ; tau_tip = u_tau,out^2 = forcing * H
    #   Fx = tau_bed * Lx*Ly - forcing * Lx*Ly*Lz   (tau_bed from the ESTIMATED
    #   u_tau,in ratio -- the paper does not tabulate Re_tau,in)
    if args.target_re_tau > 0:
        u_out_eq = args.target_re_tau / args.Re * args.H
        forcing_eq = u_out_eq ** 2 / args.H
        u_in_eq = args.r_in * u_out_eq
        A = args.Lx * args.Ly
        Fx_eq = u_in_eq ** 2 * A - forcing_eq * A * args.Lz
    else:
        u_out_eq = forcing_eq = u_in_eq = Fx_eq = None

    # concatenate runs on a continuous time axis
    data, restarts, t_off = None, [], 0.0
    for path in args.logs:
        d = parse_log(path)
        if not args.all:
            d = {k: v[10:] for k, v in d.items()}   # drop the restart spike
        if len(d['time']) == 0:
            print(f'warning: no diagnostic rows in {path}')
            continue
        d['time'] = d['time'] + t_off
        if data is None:
            data = d
        else:
            restarts.append(d['time'][0])
            data = {k: np.concatenate([data[k], d[k]]) for k in data}
        t_off = data['time'][-1]
    if data is None:
        raise SystemExit('no data parsed')

    t = data['time']

    def add_ref(ax, series, val, color, label, force=False):
        """Reference line only if it doesn't wreck the axis: the axes always
        follow the data; a far-away reference becomes an off-scale note.
        force=True always draws the line and widens the axis to include it."""
        lo, hi = np.nanmin(series), np.nanmax(series)
        span = max(hi - lo, 0.05 * abs(0.5 * (hi + lo)), 1e-12)
        if force:
            ax.axhline(val, color=color, ls='--', lw=1, label=label)
            cur = ax.get_ylim()
            pad = 0.06 * (max(hi, val) - min(lo, val))
            ax.set_ylim(min(cur[0], lo - pad, val - pad),
                        max(cur[1], hi + pad, val + pad))
            return True
        if lo - 1.5 * span <= val <= hi + 1.5 * span:
            ax.axhline(val, color=color, ls='--', lw=1, label=label)
            return True
        ax.set_ylim(lo - 0.15 * span, hi + 0.15 * span)
        n_notes = getattr(ax, '_ref_notes', 0)
        ax.text(0.98, 0.95 - 0.09 * n_notes, label + rf' $= {val:.4g}$ (off scale)',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=9, color=color)
        ax._ref_notes = n_notes + 1
        return False

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    (ax_dt, ax_ut), (ax_f, ax_tip) = axes

    ax_dt.plot(t, data['dt'], 'k-', lw=1)
    ax_dt.set_ylabel(r'$\Delta t$')
    ax_dt.set_title(r'time step')

    ax_ut.plot(t, data['u_tau'], 'k-', lw=1)
    if u_in_eq is not None:
        add_ref(ax_ut, data['u_tau'], u_in_eq, 'C0',
                rf'est.\ eq.\ ({args.r_in:.2f}$\,u_{{\tau,out}}$)', force=True)
        # curiosity marker: half the expected (tip) friction velocity
        add_ref(ax_ut, data['u_tau'], 0.5 * u_out_eq, 'C2',
                rf'$u_{{\tau,out}}/2$', force=True)
        ax_ut.legend(loc='best', fontsize=9)
    ax_ut.set_ylabel(r'$u_{\tau,\mathrm{bed}}$')
    ax_ut.set_title(r'bed friction velocity')

    ax_f.plot(t, data['forcing'], 'k-', lw=1, label=r'forcing $-\partial P/\partial x$')
    if forcing_eq is not None:
        add_ref(ax_f, data['forcing'], forcing_eq, 'C0',
                rf'eq.\ $u_{{\tau,out}}^2/H$')
    ax_f.set_ylabel(r'forcing')
    ax_f.legend(loc='upper left', fontsize=9)
    ax_fx = ax_f.twinx()
    ax_fx.plot(t, data['Fx'], 'C3-', lw=1, label=r'canopy $F_x$')
    if Fx_eq is not None:
        add_ref(ax_fx, data['Fx'], Fx_eq, 'C3', rf'est.\ eq.\ $F_x$')
    ax_fx.set_ylabel(r'canopy $F_x$', color='C3')
    ax_fx.tick_params(axis='y', colors='C3')
    ax_fx.legend(loc='lower right', fontsize=9)
    ax_f.set_title(r'driving vs canopy drag')
    ax_f.set_xlabel(r'$t\, U_b/H$')

    ax_tip.plot(t, data['u_tau_tip'], 'k-', lw=1)
    if args.target_re_tau > 0:
        if add_ref(ax_tip, data['u_tau_tip'], args.target_re_tau / args.Re * args.H,
                   'C0', rf'$Re_{{\tau,out}}={args.target_re_tau:.0f}$'):
            ax_tip.legend(loc='best', fontsize=9)
    ax_tip.set_ylabel(r'$u_{\tau,\mathrm{tip}}$')
    ax_tip.set_title(r'tip friction velocity (momentum balance)')
    ax_tip.set_xlabel(r'$t\, U_b/H$')

    for ax in (ax_dt, ax_ut, ax_f, ax_tip):
        ax.grid(alpha=0.3)
        for tr in restarts:
            ax.axvline(tr, color='gray', ls=':', lw=0.8)

    fig.tight_layout()
    out = args.out
    if out is None:
        os.makedirs('figures_local', exist_ok=True)
        out = 'figures_local/timeseries_diagnostics.png'
    fig.savefig(out, dpi=170, bbox_inches='tight')
    print(f'saved {out} ({len(t)} samples, t = {t[0]:.2f} .. {t[-1]:.2f})')


if __name__ == '__main__':
    main()
