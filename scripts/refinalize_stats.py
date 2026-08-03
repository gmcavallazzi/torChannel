#!/usr/bin/env python
"""Re-finalize a turbulence_stats.npz from its running-sum state file.

The state file holds the raw accumulated sums; the final statistics file holds
those sums divided by n_samples, plus derived scalars (u_tau) and metadata
(Lz, delta, top_wall_bc_type, nu). If a run was produced by an older build, the
derived scalars can be wrong or the metadata missing even though the sums
themselves are perfectly good. Re-finalizing recomputes the derived quantities
with the current code, without re-running the simulation.

Concretely, this repairs runs launched before the open-channel u_tau fix, where
finalize_statistics averaged the bottom wall with the FREE SURFACE and reported
u_tau several times too large. Only the scalar was affected -- the profiles and
spectra in the state file are untouched.

Usage:
    python scripts/refinalize_stats.py \
        results_re180_open/turbulence_stats_state.npz \
        --config examples/re180_open/config.yaml \
        -o results_re180_open/turbulence_stats.npz
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch.set_default_dtype(torch.float64)

from torchannel.turbstats import TurbulenceStats
from torchannel.utils import (generate_double_stretched_grid, generate_grid,
                              generate_hybrid_grid)


def build_grid(cfg):
    """Rebuild the wall-normal grid exactly as ChannelFlow.__init__ does."""
    dom, grd = cfg['domain'], cfg['grid']
    Lz = dom['Lz']
    stretching = dom.get('stretching_type', 'symmetric')

    if stretching == 'double':
        return generate_double_stretched_grid(
            grd['nz_canopy'], grd['nz_outer'], dom['z_transition'], Lz,
            dom.get('gamma_canopy', 2.0), dom.get('gamma_outer', 'auto'))
    if stretching == 'hybrid':
        return generate_hybrid_grid(
            grd['nz_uniform'], grd['nz_stretched'], dom['z_transition'], Lz,
            dom.get('gamma_stretched', 1.8))
    return generate_grid(cfg['flow']['gamma'], grd['nz'], Lz,
                         stretching_type=stretching)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('state_file')
    p.add_argument('--config', required=True,
                   help='the config the run was produced with')
    p.add_argument('-o', '--output', default=None,
                   help='destination (default: turbulence_stats.npz beside the state file)')
    p.add_argument('--force', action='store_true',
                   help='overwrite the output if it already exists')
    a = p.parse_args(argv)

    cfg = yaml.safe_load(open(a.config))
    z_f, z_c, dz_f, dz_c = build_grid(cfg)
    nz = len(dz_f)

    dom, grd, flow = cfg['domain'], cfg['grid'], cfg['flow']
    Lx, Ly, Lz = dom['Lx'], dom['Ly'], dom['Lz']
    nx, ny = grd['nx'], grd['ny']
    nu = 1.0 / flow['Re']

    bc = (cfg.get('boundary_conditions', {}).get('top_wall', {})
          .get('type', 'dirichlet'))
    canopy_cfg = cfg.get('canopy', {})
    if canopy_cfg.get('enabled', False):
        delta = Lz - float(canopy_cfg.get('h', 0.25))
    elif bc == 'neumann':
        delta = Lz
    else:
        delta = Lz / 2.0

    stats_cfg = cfg.get('statistics', {})
    st = TurbulenceStats(
        nx, ny, nz, Lx, Ly, Lz, z_c, z_f, dz_c, dz_f,
        Lx / nx, Ly / ny, nu, flow['Re_tau'],
        z_plus_target=stats_cfg.get('z_plus_target', 15.0), device='cpu',
        spectra_z=stats_cfg.get('spectra_z', None),
        top_wall_bc_type=bc, delta=delta)

    st.load_state(a.state_file)

    out = a.output or os.path.join(os.path.dirname(a.state_file),
                                   'turbulence_stats.npz')
    if os.path.exists(out) and not a.force:
        # Report what the existing file says, so the difference is visible
        # rather than silently overwritten.
        try:
            old = np.load(out)
            if 'u_tau' in old.files:
                print(f"Existing {out}: u_tau = {float(old['u_tau']):.6e} "
                      f"(Re_tau = {float(old['u_tau']) * delta / nu:.2f})")
        except Exception:
            pass
        sys.exit(f"{out} exists; pass --force to overwrite")

    stats = st.finalize_statistics()
    np.savez_compressed(out, **stats)

    u_tau = float(stats['u_tau'])
    print(f"\n  n_samples = {st.n_samples}")
    print(f"  top_wall  = {bc}   delta = {delta:.6f}")
    print(f"  u_tau     = {u_tau:.6e}   ->  Re_tau = {u_tau * delta / nu:.2f}")
    print(f"  wrote {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
