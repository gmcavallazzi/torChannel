"""Validation for the volume-penalization immersed boundary (immersed.py).

A flat solid slab (z < z1) imposed by penalization must recover ANALYTIC
Poiseuille flow in the reduced gap [z1, Lz]:

  * the streamwise profile is parabolic with peak u_max / U_bulk = 3/2,
  * the velocity deep inside the solid is ~ 0 and scales LINEARLY with eta
    (steady balance (chi/eta) u ~ forcing => u_solid = O(eta) interior
    suppression; the classic O(sqrt(eta)) error is the interfacial slip length,
    not this deep-solid velocity),
  * the field stays divergence-free at projection level.

This is the Phase-0 quantitative certificate that the IB forcing is correct and
leaves the FFT-Poisson projection intact. Run: python tests/test_immersed.py
"""

import copy
import os
import sys
import tempfile

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
torch.set_default_dtype(torch.float64)

from solver import ChannelFlow
from immersed import solid_fraction

CFG = os.path.join(os.path.dirname(__file__), '..', 'configs', 'penalization_slab_test.yaml')


def _run(eta, nsteps):
    """Run the slab config to (near) steady state and return the solver."""
    with open(CFG) as f:
        cfg = yaml.safe_load(f)
    cfg['immersed']['eta'] = eta
    cfg['time']['n_steps'] = nsteps
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as fh:
        yaml.safe_dump(cfg, fh)
        path = fh.name
    flow = ChannelFlow(path)
    for _ in range(nsteps):
        flow.step_imex(flow.dt)
        flow.time += flow.dt
    os.unlink(path)
    return flow, cfg


def _profile(flow):
    """x,y-averaged streamwise profile u(z) on interior cell centres."""
    ui = flow.u[1:flow.nx + 1, 1:flow.ny + 1, 1:flow.nz + 1]
    uz = ui.mean(dim=(0, 1)).detach().cpu().numpy()          # (nz,)
    zc = flow.z_c[1:flow.nz + 1].detach().cpu().numpy()
    dzf = flow.dz_f[0:flow.nz].detach().cpu().numpy()
    chi = flow.chi_c[1:flow.nx + 1, 1:flow.ny + 1, 1:flow.nz + 1].mean(dim=(0, 1))
    chi = chi.detach().cpu().numpy()                         # 1 in solid, 0 in fluid
    return uz, zc, dzf, chi


def main():
    Lz = 2.0
    z1 = 0.5
    print("=" * 70)
    print("Immersed-boundary penalization validation (flat slab z<0.5)")
    print("=" * 70)

    # ---- Geometry sanity: solid fraction ~ z1/Lz -------------------------
    flow, cfg = _run(eta=1e-4, nsteps=4000)
    phi = solid_fraction(flow.chi_c, flow.nx, flow.ny, flow.nz, flow.dz_f)
    phi_exp = z1 / Lz
    print(f"[1] solid fraction = {phi:.4f}  (expected ~ {phi_exp:.4f})")
    assert abs(phi - phi_exp) < 0.03, "solid fraction off"

    uz, zc, dzf, chi = _profile(flow)
    fluid = chi < 0.5

    # ---- Fluid bulk velocity (volume-weighted over fluid cells) ----------
    w = dzf * fluid
    U_bulk = (uz * w).sum() / w.sum()
    print(f"    fluid U_bulk = {U_bulk:.4f}  (target 1.0)")

    # ---- Parabolic signature: u_max / U_bulk = 3/2 -----------------------
    u_max = uz[fluid].max()
    ratio = u_max / U_bulk
    print(f"[2] u_max/U_bulk = {ratio:.4f}  (analytic Poiseuille = 1.5000)")
    assert abs(ratio - 1.5) < 0.05, "profile not parabolic"

    # ---- Compare full profile to analytic parabola on [z1, Lz] -----------
    H = Lz - z1
    zf = zc[fluid]
    u_analytic = 6.0 * U_bulk * (zf - z1) * (Lz - zf) / H**2   # mean = U_bulk
    rel_l2 = np.sqrt(((uz[fluid] - u_analytic) ** 2).sum() / (u_analytic ** 2).sum())
    print(f"[3] profile vs analytic parabola: rel-L2 = {rel_l2:.4f}")
    assert rel_l2 < 0.06, "profile deviates from analytic parabola"

    # ---- Deep-solid velocity ~ 0, scaling LINEARLY with eta --------------
    deep = zc < (z1 - 0.1)                  # well inside the solid
    slip_ref = np.abs(uz[deep]).max() / U_bulk
    print(f"[4] max|u|/U_bulk in solid (eta=1e-4) = {slip_ref:.2e}")
    assert slip_ref < 0.05, "velocity not suppressed inside solid"

    flow2, _ = _run(eta=1e-3, nsteps=4000)
    uz2, zc2, dzf2, chi2 = _profile(flow2)
    w2 = dzf2 * (chi2 < 0.5)
    U2 = (uz2 * w2).sum() / w2.sum()
    slip_hi = np.abs(uz2[zc2 < (z1 - 0.1)]).max() / U2
    rate = slip_hi / slip_ref
    print(f"    max|u|/U_bulk in solid (eta=1e-3) = {slip_hi:.2e};  "
          f"ratio(1e-3/1e-4) = {rate:.2f}  (linear O(eta) => ~10 expected)")
    assert slip_hi > slip_ref, "deep-solid velocity should grow with eta"
    assert 6.0 < rate < 15.0, "deep-solid velocity not ~linear in eta"

    # ---- Divergence stays at projection level ----------------------------
    from utils import compute_divergence
    div = compute_divergence(flow.u, flow.v, flow.w, flow.nx, flow.ny, flow.nz,
                             flow.dx, flow.dy, flow.dz_f)
    max_div = float(torch.max(torch.abs(div)))
    print(f"[5] max|div| = {max_div:.2e}")
    assert max_div < 1e-8, "divergence too large"

    print("\nALL IMMERSED-BOUNDARY TESTS PASSED")


if __name__ == "__main__":
    main()
