"""Tests for the control API (torchannel.control).

Self-contained: builds its own small config with a synthetic initial condition,
so it needs no external seed field and runs on CPU in CI.
"""

import os
import sys

import pytest
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch.set_default_dtype(torch.float64)

from torchannel.control import ChannelFlowEnv, OppositionControl
from torchannel.utils import compute_divergence


def _config(tmp_path, nx=16, ny=16, nz=24):
    cfg = {
        "grid": {"nx": nx, "ny": ny, "nz": nz},
        "domain": {"Lx": 4.0, "Ly": 2.0, "Lz": 1.0, "stretching_type": "bottom"},
        "flow": {"Re": 2870.0, "Re_tau": 180.0, "U_bulk": 1.0, "gamma": 1.6},
        "boundary_conditions": {"top_wall": {"type": "neumann"}},
        "initialization": {"type": "vortices", "perturbation_intensity": 0.05,
                           "n_vortices": 2, "seed": 7},
        "solver": {"type": "fft"},
        "time": {"dt": 2e-3, "n_steps": 10**9, "t_max": 1e9,
                 "CFL_target": 0.25, "dt_update_interval": 0,
                 "dt_max": 5e-3, "dt_min": 1e-5, "scheme": "IMEX"},
        "compute": {"device": "cpu"},
        "output": {"results_folder": str(tmp_path / "out"),
                   "n_out": 10**9, "n_save": 10**9},
        "statistics": {"n_stats": 0},
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return str(p)


@pytest.fixture
def env(tmp_path):
    return ChannelFlowEnv(_config(tmp_path), action_interval=0.01, max_action=0.05)


def test_observation_shape_and_type(env):
    obs = env.reset()
    assert obs.shape == (env.flow.nx, env.flow.ny)
    assert obs.dtype == torch.float32
    # Normalised by its own mean.
    assert float(obs.mean()) == pytest.approx(1.0, rel=1e-5)


def test_action_is_mean_subtracted(env):
    """Global mass conservation: a net wall flux has no pressure solution.

    The all-Neumann pressure problem's compatibility condition is exactly zero
    net flux, so a biased action must be corrected rather than accepted.
    """
    env.reset()
    biased = torch.ones(env.control_shape) * 0.03   # entirely one-signed
    env.step(biased)
    wv = env.flow._wall_velocity
    assert abs(float(wv.mean())) < 1e-15, "wall velocity was not mean-subtracted"
    assert float(wv.abs().max()) > 0.0, "the action was zeroed out entirely"


def test_actuation_preserves_divergence_free_field(env):
    """Blowing/suction must not break the projection."""
    env.reset()
    f = env.flow
    torch.manual_seed(0)
    for _ in range(3):
        action = torch.randn(env.control_shape) * 0.02
        env.step(action)
        div = compute_divergence(f.u, f.v, f.w, f.nx, f.ny, f.nz,
                                 f.dx, f.dy, f.dz_f_c)
        assert float(div.abs().max()) < 1e-10, \
            f"divergence {float(div.abs().max()):.2e} after actuation"
        flux = float(f.w[1:f.nx + 1, 1:f.ny + 1, 0].sum())
        assert abs(flux) < 1e-12, f"net wall flux {flux:.2e}"


def test_zero_action_matches_no_actuation(tmp_path):
    """A zero action must reproduce the uncontrolled flow bit-for-bit."""
    cfg = _config(tmp_path)
    e1 = ChannelFlowEnv(cfg, action_interval=0.01)
    e1.reset()
    e1.step(torch.zeros(e1.control_shape))
    u_controlled = e1.flow.u.clone()

    second = tmp_path / "b"
    second.mkdir()
    e2 = ChannelFlowEnv(_config(second), action_interval=0.01)
    e2.reset()
    e2.flow.set_wall_velocity(None)
    t_end = e2.flow.time + e2.action_interval
    while e2.flow.time < t_end:
        dt = min(float(e2.flow.compute_cfl_dt()), t_end - e2.flow.time)
        e2.flow.step_imex(dt)
        e2.flow.time += dt
        e2.flow.current_step += 1

    assert torch.equal(u_controlled, e2.flow.u), \
        "a zero action perturbed the flow"


def test_action_is_clipped(env):
    env.reset()
    env.step(torch.full(env.control_shape, 10.0))
    assert float(env.flow._wall_velocity.abs().max()) <= env.max_action + 1e-12


def test_coarse_control_grid_upsamples(tmp_path):
    env = ChannelFlowEnv(_config(tmp_path, nx=16, ny=16), control_shape=(4, 4),
                         action_interval=0.005)
    env.reset()
    a = torch.zeros(4, 4)
    a[0, 0] = 0.04
    env.step(a)
    wv = env.flow._wall_velocity
    assert wv.shape == (16, 16)
    # The actuated patch is 4x4 cells of the wall grid and uniform within it.
    patch = wv[0:4, 0:4]
    assert torch.allclose(patch, patch[0, 0].expand_as(patch))


def test_indivisible_control_shape_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="divide"):
        ChannelFlowEnv(_config(tmp_path, nx=16, ny=16), control_shape=(5, 4))


def test_non_imex_scheme_is_rejected(tmp_path):
    p = _config(tmp_path)
    cfg = yaml.safe_load(open(p))
    cfg["time"]["scheme"] = "FE"
    open(p, "w").write(yaml.safe_dump(cfg))
    with pytest.raises(NotImplementedError, match="IMEX"):
        ChannelFlowEnv(p)


def test_opposition_control_opposes_the_detection_plane(env):
    """The action must be the negation of w at the detection plane."""
    env.reset()
    pol = OppositionControl(env, detection_z_plus=15.0, gain=1.0)
    w_d = env.velocity_at_z_plus(15.0, component='w')
    action = pol()
    assert torch.allclose(action, -w_d), "opposition control is not opposing"


def test_drag_reduction_is_reported(env):
    env.reset()
    obs, reward, done, info = env.step(torch.zeros(env.control_shape))
    assert done is False
    for key in ("drag_reduction", "tau_wall", "u_tau", "time", "substeps"):
        assert key in info
    assert info["substeps"] >= 1
    assert reward == pytest.approx(info["drag_reduction"])
