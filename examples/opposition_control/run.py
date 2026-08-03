#!/usr/bin/env python
"""Opposition control at Re_tau = 180 (Choi, Moin & Kim 1994).

Demonstrates driving torChannel as a control environment: the solver is stepped
from a Python loop, the wall shear field is read out each action, and
blowing/suction is written back to the wall. No co-simulation layer, no
file-based coupling -- the solver is a PyTorch program in the same process.

The control law is v_wall(x, z) = -v(x, y_d, z): oppose the wall-normal velocity
sensed at a detection plane. At y_d+ ~ 10-15 this is known to give ~20-25% drag
reduction at Re_tau = 180, which makes it a self-validating check on the API.

Usage:
    python examples/opposition_control/run.py --field results_re180_open/fields.npz

Start from an EQUILIBRATED turbulent field. Starting from a synthetic initial
condition measures the transition transient, not the control.
"""

import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

torch.set_default_dtype(torch.float64)

from torchannel.control import ChannelFlowEnv, OppositionControl


def build_config(base_config, field_file, out_dir, tmp_path):
    """Derive a control config from the validation case: restart, no stats."""
    cfg = yaml.safe_load(open(base_config))
    cfg['initialization'] = {'type': 'parabolic', 'field_file': field_file,
                             'reset_time': True}
    cfg['statistics'] = {'n_stats': 0}
    cfg['output'] = {'results_folder': out_dir, 'n_out': 10**9, 'n_save': 10**9}
    cfg['time']['t_max'] = 1e9
    cfg['time']['n_steps'] = 10**9
    with open(tmp_path, 'w') as fh:
        yaml.safe_dump(cfg, fh)
    return tmp_path


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--config', default='examples/re180_open/config.yaml')
    p.add_argument('--field', required=True,
                   help='equilibrated turbulent field to start from')
    p.add_argument('--detection-z-plus', type=float, default=15.0)
    p.add_argument('--action-interval', type=float, default=0.05,
                   help='physical time each action is held')
    p.add_argument('--n-actions', type=int, default=400)
    p.add_argument('--warmup', type=int, default=40,
                   help='uncontrolled actions used to fix the baseline drag')
    p.add_argument('--max-action', type=float, default=0.05)
    p.add_argument('--out', default='results_opposition')
    args = p.parse_args()

    if not os.path.exists(args.field):
        sys.exit(f"initial field not found: {args.field}\n"
                 f"Run the Re_tau=180 case first (see examples/re180_open/).")

    os.makedirs(args.out, exist_ok=True)
    cfg = build_config(args.config, args.field, args.out,
                       os.path.join(args.out, '_control.yaml'))

    env = ChannelFlowEnv(cfg, action_interval=args.action_interval,
                         max_action=args.max_action)
    policy = OppositionControl(env, detection_z_plus=args.detection_z_plus)

    zero = torch.zeros(env.control_shape, device=env.flow.device,
                       dtype=env.flow.dtype)

    # --- uncontrolled warm-up: establishes the baseline drag ---------------
    env.reset()
    print(f"{'action':>7} {'t':>9} {'tau_w':>12} {'u_tau':>10} {'DR %':>8}")
    taus = []
    for i in range(args.warmup):
        _, _, _, info = env.step(zero)
        taus.append(info['tau_wall'])
        if i % 10 == 0:
            print(f"{i:7d} {info['time']:9.3f} {info['tau_wall']:12.5e} "
                  f"{info['u_tau']:10.6f} {'--':>8}")
    baseline = float(np.mean(taus[len(taus) // 2:]))
    print(f"\nBaseline tau_w (second half of warm-up) = {baseline:.6e}\n")

    # --- controlled ---------------------------------------------------------
    env._baseline_tau = baseline
    hist = {'t': [], 'tau': [], 'dr': [], 'u_tau': []}
    obs = env.observe()
    for i in range(args.n_actions):
        obs, dr, _, info = env.step(policy(obs))
        hist['t'].append(info['time'])
        hist['tau'].append(info['tau_wall'])
        hist['dr'].append(dr)
        hist['u_tau'].append(info['u_tau'])
        if i % 10 == 0:
            print(f"{i:7d} {info['time']:9.3f} {info['tau_wall']:12.5e} "
                  f"{info['u_tau']:10.6f} {100 * dr:8.2f}")

    # Report the settled value, not the transient.
    tail = hist['dr'][len(hist['dr']) // 2:]
    dr_mean = 100 * float(np.mean(tail))
    dr_std = 100 * float(np.std(tail))
    print(f"\nDrag reduction over the second half: {dr_mean:.1f} +/- {dr_std:.1f} %")
    print("Expected ~20-25% at Re_tau=180 with y_d+ ~ 10-15 "
          "(Choi, Moin & Kim 1994).")

    npz = os.path.join(args.out, 'opposition_history.npz')
    np.savez(npz, baseline_tau=baseline, detection_z_plus=args.detection_z_plus,
             **{k: np.asarray(v) for k, v in hist.items()})
    print(f"wrote {npz}")


if __name__ == '__main__':
    main()
