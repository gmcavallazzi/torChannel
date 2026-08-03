"""Tests for configurable precision (compute.precision).

Covers the plumbing that has no other symptom when it goes wrong: a metric left
at float64 makes the hot kernels run fp64 arithmetic and silently downcast --
correct answers, several times slower, invisible in the output.

Run:
    pytest tests/test_precision.py -v
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch.set_default_dtype(torch.float64)

from torchannel.turbstats import TurbulenceStats
from torchannel.utils import compute_bulk_velocity, compute_u_tau, generate_grid


# --------------------------------------------------------------------------
# Reductions must stay exact regardless of field precision
# --------------------------------------------------------------------------

def test_bulk_velocity_matches_across_precision():
    """The float32 path must agree with float64 to well under controller noise.

    u_bulk drives an integral controller whose gain is 0.1/dt (~385 at
    production dt), so error here is amplified straight into the forcing.
    """
    nx, ny, nz = 24, 20, 32
    _, _, dz_f, _ = generate_grid(1.6, nz, 1.0, stretching_type="bottom")
    dx = dy = 0.05
    cell_vol = (dx * dy * dz_f.view(1, 1, -1)).expand(nx, ny, nz)
    total_volume = nx * dx * ny * dy * 1.0

    torch.manual_seed(0)
    u64 = torch.rand(nx + 1, ny + 2, nz + 2) + 1.0

    b64 = float(compute_bulk_velocity(u64, cell_vol, total_volume))
    b32 = float(compute_bulk_velocity(u64.to(torch.float32), cell_vol, total_volume))

    assert abs(b32 - b64) / abs(b64) < 1e-6, f"{b32} vs {b64}"


def test_bulk_velocity_float64_path_is_untouched():
    """Guards gate G0: the all-float64 branch must be the original expression."""
    nx, ny, nz = 8, 8, 12
    _, _, dz_f, _ = generate_grid(1.6, nz, 1.0, stretching_type="bottom")
    dx = dy = 0.1
    cell_vol = (dx * dy * dz_f.view(1, 1, -1)).expand(nx, ny, nz)
    total_volume = nx * dx * ny * dy * 1.0

    torch.manual_seed(1)
    u = torch.rand(nx + 1, ny + 2, nz + 2)
    expected = torch.sum(u[1:nx + 1, 1:ny + 1, 1:nz + 1] * cell_vol) / total_volume
    got = compute_bulk_velocity(u, cell_vol, total_volume)
    assert torch.equal(got, expected), "float64 bulk velocity is no longer bit-identical"


def test_bulk_velocity_does_not_promote_the_field():
    """A float32 field must not be widened to float64 in bulk.

    At production resolution the promoted temporary is ~850 MB, allocated only
    to be reduced away. Detected here by peak-allocation on CUDA; skipped on CPU.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA to measure allocation")
    nx, ny, nz = 128, 128, 64
    _, _, dz_f, _ = generate_grid(1.6, nz, 1.0, stretching_type="bottom")
    dz_f = dz_f.cuda()
    dx = dy = 0.05
    cell_vol = (dx * dy * dz_f.view(1, 1, -1)).expand(nx, ny, nz)
    u32 = torch.rand(nx + 1, ny + 2, nz + 2, device="cuda", dtype=torch.float32)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.max_memory_allocated()
    compute_bulk_velocity(u32, cell_vol, 1.0)
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - before

    field_bytes = u32.numel() * 4
    assert extra < field_bytes, (
        f"allocated {extra / 2**20:.0f} MiB for a {field_bytes / 2**20:.0f} MiB "
        f"field -- the float32 field is being promoted to float64")


def test_u_tau_top_wall_uses_its_own_spacing():
    """Closed channel: each wall must use its own distance to the first centre.

    On a non-symmetric grid the two spacings differ by orders of magnitude, so
    reusing the bottom spacing for the top wall is badly wrong.
    """
    nz, Lz, nu = 48, 1.0, 1e-3
    _, z_c, _, _ = generate_grid(1.6, nz, Lz, stretching_type="bottom")

    # Uniform-shear field: U = S*z, so both walls have the same true gradient.
    S = 3.0
    u = torch.zeros(6, 6, nz + 2)
    u[:, :, 1] = S * float(z_c[1])
    u[:, :, -2] = S * float(Lz - z_c[-2])

    got = float(compute_u_tau(u, z_c, nu, top_wall_bc_type="dirichlet"))
    expected = float(np.sqrt(nu * S))
    assert got == pytest.approx(expected, rel=1e-9), (
        f"{got} vs {expected}: the top wall is not using its own spacing")


# --------------------------------------------------------------------------
# Statistics accumulators
# --------------------------------------------------------------------------

@pytest.mark.parametrize("field_dtype", [torch.float64, torch.float32])
def test_statistics_accumulators_are_always_float64(field_dtype):
    """Running sums stay float64 whatever the fields do.

    They must not inherit the global default dtype: a run that flipped it would
    silently degrade every accumulated statistic with no other symptom.
    """
    nx = ny = 8
    nz = 16
    z_f, z_c, dz_f, dz_c = generate_grid(1.6, nz, 1.0, stretching_type="bottom")
    st = TurbulenceStats(nx, ny, nz, 1.0, 1.0, 1.0, z_c, z_f, dz_c, dz_f,
                         0.1, 0.1, 1e-3, Re_tau_target=180.0, device="cpu",
                         top_wall_bc_type="neumann")

    for name in ("U_sum", "uu_sum", "vv_sum", "ww_sum", "uw_sum",
                 "uuu_sum", "www_sum", "fx_profile_sum",
                 "E_uu_2d_sum", "E_vv_2d_sum", "E_ww_2d_sum", "E_uw_2d_sum"):
        assert getattr(st, name).dtype == torch.float64, f"{name} is not float64"

    torch.manual_seed(3)
    u = torch.rand(nx + 1, ny + 2, nz + 2, dtype=field_dtype)
    v = torch.rand(nx + 2, ny + 1, nz + 2, dtype=field_dtype)
    w = torch.rand(nx + 2, ny + 2, nz + 1, dtype=field_dtype)
    st.accumulate_statistics(u, v, w, 0.06)

    # Accumulating a reduced-precision field must not change the sums' dtype.
    for name in ("U_sum", "uu_sum", "uw_sum"):
        assert getattr(st, name).dtype == torch.float64, \
            f"{name} was demoted by accumulating a {field_dtype} field"


def test_statistics_agree_between_precisions():
    """Same field in float64 and float32 must give the same profiles."""
    nx = ny = 16
    nz = 24
    z_f, z_c, dz_f, dz_c = generate_grid(1.6, nz, 1.0, stretching_type="bottom")

    torch.manual_seed(4)
    u = torch.rand(nx + 1, ny + 2, nz + 2) + 1.0
    v = torch.rand(nx + 2, ny + 1, nz + 2) * 0.1
    w = torch.rand(nx + 2, ny + 2, nz + 1) * 0.1

    out = {}
    for dt in (torch.float64, torch.float32):
        st = TurbulenceStats(nx, ny, nz, 1.0, 1.0, 1.0, z_c, z_f, dz_c, dz_f,
                             0.1, 0.1, 1e-3, Re_tau_target=180.0, device="cpu",
                             top_wall_bc_type="neumann")
        st.accumulate_statistics(u.to(dt), v.to(dt), w.to(dt), 0.06)
        out[dt] = st.finalize_statistics()

    for key in ("U_mean", "uu_mean", "vv_mean", "ww_mean", "uw_mean"):
        a, b = out[torch.float64][key], out[torch.float32][key]
        rel = np.abs(a - b).max() / (np.abs(a).max() + 1e-300)
        assert rel < 1e-5, f"{key}: relative difference {rel:.2e}"


# --------------------------------------------------------------------------
# Config validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["float16", "bfloat16", "double", "fp32", ""])
def test_invalid_precision_is_rejected(bad, tmp_path):
    """Bad values must fail at config parse, not halfway through a run."""
    import yaml

    from torchannel.solver import ChannelFlow

    cfg = {
        "grid": {"nx": 8, "ny": 8, "nz": 8},
        "domain": {"Lx": 1.0, "Ly": 1.0, "Lz": 1.0, "stretching_type": "bottom"},
        "flow": {"Re": 1000.0, "Re_tau": 180.0, "U_bulk": 1.0, "gamma": 1.6},
        "boundary_conditions": {"top_wall": {"type": "neumann"}},
        "initialization": {"type": "parabolic"},
        "time": {"dt": 1e-3, "n_steps": 1, "CFL_target": 0.2},
        "compute": {"device": "cpu", "precision": bad},
        "output": {"results_folder": str(tmp_path / "out")},
        "statistics": {"n_stats": 0},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))

    with pytest.raises(ValueError) as exc:
        ChannelFlow(config_file=str(path))

    msg = str(exc.value)
    assert "precision" in msg.lower()
    if bad in ("float16", "bfloat16"):
        # The message should say WHY half precision is impossible here.
        assert "65504" in msg or "overflow" in msg.lower()
