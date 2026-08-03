"""Assertion-based regression tests, runnable by pytest and in CI.

The ~60 other scripts in this directory are standalone diagnostics that print
their results for a human to read. This file is the machine-checkable subset:
every check has an explicit tolerance and fails loudly.

Run:
    pytest tests/test_regression.py -v          # everything
    pytest tests/test_regression.py -m "not gpu"  # CI subset (no CUDA needed)
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch.set_default_dtype(torch.float64)

from torchannel.projection_fft import initialize_fft_solver, solve_poisson_fft
from torchannel.tridiag import pcr_solve
from torchannel.turbstats import TurbulenceStats
from torchannel.utils import compute_divergence, generate_grid


# --------------------------------------------------------------------------
# Grid generation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("stretching", ["symmetric", "bottom"])
@pytest.mark.parametrize("nz,Lz,gamma", [(32, 2.0, 2.6), (64, 1.0, 1.6)])
def test_grid_is_consistent(stretching, nz, Lz, gamma):
    z_f, z_c, dz_f, dz_c = generate_grid(gamma, nz, Lz, stretching_type=stretching)

    assert z_f.shape == (nz + 1,)
    assert z_c.shape == (nz + 2,), "z_c must carry one ghost cell at each end"
    assert dz_f.shape == (nz,)
    assert dz_c.shape == (nz + 1,)

    assert torch.all(dz_f > 0), "faces must be monotonically increasing"
    # dz_f sums to Lz exactly -- plot_statistics relies on this to recover Lz
    # on non-symmetric grids, where z_c[0] + z_c[-1] != Lz.
    assert torch.allclose(dz_f.sum(), torch.tensor(float(Lz)))
    assert abs(float(z_f[0])) < 1e-14
    assert abs(float(z_f[-1]) - Lz) < 1e-12

    # Interior centres bisect their faces; ghosts mirror across each boundary.
    assert torch.allclose(z_c[1:-1], 0.5 * (z_f[:-1] + z_f[1:]))
    assert abs(float(z_c[0] + z_c[1])) < 1e-12, "bottom ghost is not mirrored"


def test_symmetric_grid_is_symmetric():
    _, _, dz_f, _ = generate_grid(2.6, 64, 2.0, stretching_type="symmetric")
    assert torch.allclose(dz_f, dz_f.flip(0), atol=1e-14)


def test_bottom_grid_clusters_at_the_bottom_only():
    _, _, dz_f, _ = generate_grid(1.6, 64, 1.0, stretching_type="bottom")
    assert dz_f[0] < dz_f[-1], "expected fine spacing at the wall"
    # Monotone coarsening away from the single wall.
    assert torch.all(dz_f[1:] - dz_f[:-1] > -1e-15)


# --------------------------------------------------------------------------
# Tridiagonal solver
# --------------------------------------------------------------------------

def test_pcr_matches_dense_solve():
    torch.manual_seed(0)
    n = 64
    # Diagonally dominant, as the implicit-diffusion systems are.
    a = torch.rand(n) * 0.4 - 0.2
    c = torch.rand(n) * 0.4 - 0.2
    b = 2.0 + torch.rand(n)
    a[0] = 0.0
    c[-1] = 0.0

    d = torch.rand(3, 5, n)
    x = pcr_solve(a, b, c, d)

    A = torch.diag(b) + torch.diag(c[:-1], 1) + torch.diag(a[1:], -1)
    expected = torch.linalg.solve(A, d.reshape(-1, n).T).T.reshape(d.shape)
    assert torch.allclose(x, expected, atol=1e-10), \
        f"max abs diff {float((x - expected).abs().max()):.3e}"


# --------------------------------------------------------------------------
# Poisson solver / projection
# --------------------------------------------------------------------------

def _random_solenoidal_setup(nx, ny, nz, Lz, top_bc):
    dx, dy = 2 * np.pi / nx, np.pi / ny
    _, _, dz_f, dz_c = generate_grid(1.6, nz, Lz, stretching_type="bottom")
    fft = initialize_fft_solver(nx, ny, nz, dx, dy, dz_c, dz_f,
                                top_wall_bc_type=top_bc)
    return dx, dy, dz_f, fft


@pytest.mark.parametrize("top_bc", ["dirichlet", "neumann"])
def test_projection_makes_velocity_divergence_free(top_bc):
    """The contract the solver actually depends on, for both pressure BCs.

    Tested end-to-end (solve_poisson_fft -> project_velocity ->
    compute_divergence) rather than by reconstructing the discrete Laplacian
    here, so it validates the solver's own stencil instead of a copy of it.
    In float64 the residual should land near machine precision.
    """
    from torchannel.projection import project_velocity

    nx = ny = 16
    nz, Lz = 24, 1.0
    dt = 1e-3
    dx, dy = 2 * np.pi / nx, np.pi / ny
    _, _, dz_f, dz_c = generate_grid(1.6, nz, Lz, stretching_type="bottom")
    fft = initialize_fft_solver(nx, ny, nz, dx, dy, dz_c, dz_f,
                                top_wall_bc_type=top_bc)

    # An arbitrary, emphatically non-solenoidal field. Ghost cells are set by the
    # solver's own fused BC kernel rather than by hand: each component has a
    # different staggering (u is nx+1 in x, v is ny+1 in y, w is nz+1 in z), so
    # hand-rolled periodic ghosts get the seam wrong and leave a residual the
    # Poisson solver cannot represent.
    from torchannel.solver import apply_bc_all

    torch.manual_seed(1)
    u = torch.randn(nx + 1, ny + 2, nz + 2)
    v = torch.randn(nx + 2, ny + 1, nz + 2)
    w = torch.randn(nx + 2, ny + 2, nz + 1)
    apply_bc_all(u, v, w, top_bc)

    div0 = compute_divergence(u, v, w, nx, ny, nz, dx, dy, dz_f)
    assert float(div0.abs().max()) > 1.0, "test field should not start solenoidal"

    p = solve_poisson_fft(div0 / dt, fft)
    assert p.shape == (nx + 2, ny + 2, nz + 2)
    assert torch.isfinite(p).all(), "Poisson solve produced non-finite pressure"

    project_velocity(u, v, w, p, nx, ny, nz, dx, dy, dz_c, dz_f, dt)
    apply_bc_all(u, v, w, top_bc)   # refresh ghosts, as step_imex does
    div1 = compute_divergence(u, v, w, nx, ny, nz, dx, dy, dz_f)

    rel = float(div1.abs().max() / div0.abs().max())
    assert rel < 1e-12, f"divergence not removed: relative residual {rel:.3e}"


def test_poisson_pins_the_singular_mode():
    """The (0,0) mode is pinned, so a constant source must not produce NaN."""
    nx = ny = 8
    _, _, _, fft = _random_solenoidal_setup(nx, ny, 16, 1.0, "neumann")
    p = solve_poisson_fft(torch.ones(nx, ny, 16), fft)
    assert torch.isfinite(p).all(), "constant source produced non-finite pressure"


def test_divergence_of_uniform_flow_is_zero():
    nx, ny, nz, Lz = 8, 8, 16, 1.0
    dx, dy = 1.0 / nx, 1.0 / ny
    _, _, dz_f, _ = generate_grid(1.6, nz, Lz, stretching_type="bottom")

    u = torch.ones(nx + 1, ny + 2, nz + 2)
    v = torch.zeros(nx + 2, ny + 1, nz + 2)
    w = torch.zeros(nx + 2, ny + 2, nz + 1)

    div = compute_divergence(u, v, w, nx, ny, nz, dx, dy, dz_f)
    assert float(div.abs().max()) < 1e-13


# --------------------------------------------------------------------------
# Statistics: open- vs closed-channel u_tau
# --------------------------------------------------------------------------

def _stats(top_bc, nz=48, Lz=1.0, nu=3.4843e-4):
    z_f, z_c, dz_f, dz_c = generate_grid(1.6, nz, Lz, stretching_type="bottom")
    return TurbulenceStats(
        8, 8, nz, 1.0, 1.0, Lz, z_c, z_f, dz_c, dz_f, 0.1, 0.1, nu,
        Re_tau_target=180.0, device="cpu", top_wall_bc_type=top_bc,
    ), z_c


def test_open_channel_delta_is_full_height():
    stats, _ = _stats("neumann")
    assert stats.delta == pytest.approx(1.0), \
        "open channel: delta must be Lz, not Lz/2"
    closed, _ = _stats("dirichlet")
    assert closed.delta == pytest.approx(0.5)


@pytest.mark.parametrize("top_bc,expected_delta", [("neumann", 1.0), ("dirichlet", 0.5)])
def test_statistics_record_their_own_geometry(top_bc, expected_delta):
    """Lz, delta and the wall BC must travel WITH the statistics.

    Post-processing used to reconstruct these -- Lz from z_c[0] + z_c[-1] (only
    valid on a symmetric grid) and delta as a hard-coded Lz/2 -- which is how
    the open-channel u_tau bug survived. Recording them removes the inference.
    """
    stats, _ = _stats(top_bc)
    stats.n_samples = 1
    nz = stats.nz
    for name in ("U_sum", "uu_sum", "vv_sum", "ww_sum", "uw_sum",
                 "uuu_sum", "www_sum", "fx_profile_sum"):
        setattr(stats, name, torch.zeros(nz, dtype=torch.float64))
    stats.U_sum = torch.linspace(0.01, 1.0, nz, dtype=torch.float64)

    out = stats.finalize_statistics()
    for key in ("Lz", "delta", "top_wall_bc_type"):
        assert key in out, f"{key} is not recorded in the statistics"
    assert out["Lz"] == pytest.approx(1.0)
    assert out["delta"] == pytest.approx(expected_delta)
    assert out["top_wall_bc_type"] == top_bc


def test_open_channel_u_tau_ignores_the_free_surface():
    """Regression: u_tau must not average in the free-surface velocity.

    Averaging U_mean[-1] as if it were a second wall inflated u_tau by ~7x on
    the Re_tau=180 open-channel case (0.4479 instead of 0.0629).
    """
    nu = 3.4843e-4
    stats, z_c = _stats("neumann", nu=nu)
    zi = z_c[1:-1]

    # A profile with a genuine wall gradient at the bottom and a large,
    # free-surface velocity at the top -- the configuration that broke.
    u_tau_true = 0.0629
    U = (u_tau_true**2 / nu) * zi.clamp(max=0.02) + 1.0 * (zi / zi.max()) ** 0.14
    U = U - U[0] + (u_tau_true**2 / nu) * zi[0]

    stats.U_sum = U.clone()
    stats.n_samples = 1
    for name in ("uu_sum", "vv_sum", "ww_sum", "uw_sum", "uuu_sum",
                 "www_sum", "fx_profile_sum"):
        setattr(stats, name, torch.zeros_like(U))

    out = stats.finalize_statistics()
    u_tau = float(out["u_tau"])

    # The bottom-wall value, computed independently.
    expected = float(np.sqrt(nu * float(U[0]) / float(z_c[1])))
    assert u_tau == pytest.approx(expected, rel=1e-10)
    assert float(U[-1]) > 0.5, "test profile should have a large free-surface U"
    # And nowhere near the value the old two-wall average produced.
    bad = float(np.sqrt(nu * 0.5 * (float(U[0]) + float(U[-1])) / float(z_c[1])))
    assert u_tau < 0.5 * bad


# --------------------------------------------------------------------------
# Bundled reference data
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["mkm180", "mkm590", "lm550"])
def test_reference_profiles_are_physical(name):
    """Guards the y-wall-normal -> z-wall-normal remap in fetch_reference_data.

    Near the wall u' ~ z, v' ~ z and w' ~ z^2, so the wall-normal component
    (torChannel's ww) must be the smallest of the three. If the remap were
    reversed, the spanwise component would be the vanishing one instead.
    """
    import plot_statistics

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "torchannel", "data", "reference", f"{name}.csv")
    if not os.path.exists(path):
        pytest.skip(f"{name}.csv not present; run scripts/fetch_reference_data.py")

    ref = plot_statistics.load_reference(name)
    for key in ("z_plus", "U_plus", "uu_plus", "vv_plus", "ww_plus", "uw_plus"):
        assert key in ref, f"missing column {key}"

    z = ref["z_plus"]
    assert z[0] == pytest.approx(0.0, abs=1e-9)
    assert np.all(np.diff(z) > 0), "z+ must be increasing"

    near = (z > 0.5) & (z < 5.0)
    assert near.any()
    assert np.all(ref["ww_plus"][near] < ref["vv_plus"][near]), \
        "wall-normal stress is not the smallest near the wall: axes look swapped"
    assert np.all(ref["vv_plus"][near] < ref["uu_plus"][near])

    # U+ = z+ holds in the viscous sublayer.
    sub = (z > 0.3) & (z < 3.0)
    assert np.allclose(ref["U_plus"][sub], z[sub], rtol=0.03)


# --------------------------------------------------------------------------
# Packaging: the core install must actually work
# --------------------------------------------------------------------------

def test_solver_imports_without_optional_dependencies():
    """`pip install -e .` (torch, numpy, pyyaml) must be enough to run a case.

    Regression: utils.py imported matplotlib at module scope, and utils is on
    the core import path (solver -> utils). A minimal install therefore could
    not even import the solver, contradicting pyproject, where matplotlib lives
    in the optional [plot] extra. Caught by CI, which installs only [test].
    """
    import subprocess
    import textwrap

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = textwrap.dedent("""
        import builtins
        _real = builtins.__import__
        BLOCKED = ("matplotlib", "scipy", "skimage")

        def guard(name, *a, **k):
            if name.split(".")[0] in BLOCKED:
                raise ImportError("No module named %r (simulated core install)" % name)
            return _real(name, *a, **k)

        builtins.__import__ = guard
        import solver, operators, utils, turbstats, canopy
        import initflow, projection, projection_fft, tridiag
        import torchannel.solver
        from solver import ChannelFlow
        assert solver is torchannel.solver
        print("OK")
    """)
    env = dict(os.environ, PYTORCH_JIT="0")
    r = subprocess.run([sys.executable, "-c", code], cwd=repo, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, (
        "the solver cannot be imported with only the core dependencies:\n"
        + r.stderr[-1500:])


# --------------------------------------------------------------------------
# GPU
# --------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_poisson_cpu_gpu_agree():
    nx = ny = 16
    nz = 24
    dx, dy = 2 * np.pi / nx, np.pi / ny
    _, _, dz_f, dz_c = generate_grid(1.6, nz, 1.0, stretching_type="bottom")

    torch.manual_seed(2)
    div = torch.randn(nx, ny, nz)
    div -= div.mean()

    p_cpu = solve_poisson_fft(div, initialize_fft_solver(
        nx, ny, nz, dx, dy, dz_c, dz_f, top_wall_bc_type="neumann"))
    p_gpu = solve_poisson_fft(div.cuda(), initialize_fft_solver(
        nx, ny, nz, dx, dy, dz_c.cuda(), dz_f.cuda(), top_wall_bc_type="neumann"))

    assert torch.allclose(p_cpu, p_gpu.cpu(), atol=1e-10)
