# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TorChannel is a GPU-accelerated Direct Numerical Simulation (DNS) solver for incompressible turbulent channel flow, built with Python/PyTorch. `torch.float64` is the default and the reference path; `compute.precision` also offers `mixed` and `float32` (see Precision below).

## Commands

```bash
# Install (editable). The modules now live in the `torchannel` package.
pip install -e ".[plot,test]"

# Run a simulation (local, GB10 GPU: PYTORCH_JIT=0 is REQUIRED — see GPU notes)
PYTORCH_JIT=0 TORCHANNEL_COMPILE=1 TORCHANNEL_POISSON_CUDAGRAPH=1 \
    torchannel-run examples/re180_open/config.yaml
# `python main.py <config>` still works (top-level shims re-export the package)

# Submit on the local SLURM (single node spark-9ab9, partition "compute")
sbatch slurm/canopy_monti_gb10.sh        # writes slurm-canopy-<jobid>.out
# launch.sh is a STALE script for a former remote cluster — do not use it here

# Assertion-based tests (pytest). The other ~60 scripts in tests/ are standalone
# print-based diagnostics and are NOT collected -- see pyproject python_files.
PYTORCH_JIT=0 pytest -m "not gpu"        # CPU subset, what CI runs
PYTORCH_JIT=0 pytest                     # + the GPU agreement test
PYTORCH_JIT=0 python tests/test_derivative.py   # a legacy standalone diagnostic

# Post-process
python post_process.py results/fields.npz --config config550.yaml
python plot_statistics.py results/turbulence_stats.npz --config config550.yaml
python plot_statistics.py results/turbulence_stats_state.npz --checkpoint --config config550.yaml
# --reference {mkm180,mkm590,lm550} overlays published DNS; --open-channel is
# auto-detected from --config. `module load texlive` first (figures use usetex).
python scripts/plot_timeseries.py slurm-canopy-291.out       # parse solver stdout diagnostics
python scripts/plot_snapshot.py results_canopy_monti/fields.npz   # 4-cut snapshot figure
```

## GPU notes (GB10 / sm_121)

The local GPU (NVIDIA GB10) cannot NVRTC-compile the legacy TorchScript fuser, so all runs need `PYTORCH_JIT=0` (the `@torch.jit.script` decorators become passthroughs). Two opt-in speed layers replace it:
- `TORCHANNEL_COMPILE=1` — torch.compile (Inductor/Triton) on the hot kernels (needs `CC=gcc`)
- `TORCHANNEL_POISSON_CUDAGRAPH=1` — CUDA-graph capture of the FFT-Poisson solve

## Architecture

**Layout**: the solver lives in the `torchannel/` package. Top-level `solver.py`, `utils.py` etc. are back-compat shims that rebind `sys.modules`, so `import solver` and `import torchannel.solver` are the SAME module object — the ~60 legacy scripts in `tests/` keep working unchanged.

**Entry point**: `torchannel-run` (or `main.py`) → creates `ChannelFlow` (from `torchannel/solver.py`) and calls `run_simulation()`.

**Core modules**:
- `solver.py` — `ChannelFlow`: time-stepping loop, restart logic, adaptive dt (CFL), bulk-velocity PI forcing, fused BC kernel. Valid `time.scheme` values: `"IMEX"` (production; AB2 advection + xy-diffusion, Crank–Nicolson implicit z-diffusion), `"FE"` (fully explicit Forward Euler with instantaneous bulk-velocity correction — testing/small cases only), `"RK3"` (NotImplementedError placeholder). There is no standalone "AB2" scheme.
- `operators.py` — staggered-grid advection/diffusion, implicit z-diffusion tridiagonal solves, fused kernels (`compute_momentum_rhs_fused_imex` is the one used by IMEX), fused CFL
- `tridiag.py` — batched parallel-cyclic-reduction tridiagonal solver (used by the implicit diffusion)
- `projection_fft.py` — FFT (x,y) + Thomas (z) Poisson solver; modified wavenumbers; Neumann pressure BCs at both walls; singular (0,0) mode pinned
- `projection.py` — dense direct Poisson solver (small grids/testing only) + `project_velocity`
- `initflow.py` — init: `parabolic`, `uniform`, `vortices` (+ random perturbations), restart from file, and `interpolate` (staggered-aware trilinear regrid of a saved field from a DIFFERENT grid/domain — treated as a FRESH start: time resets, field rescaled to `U_bulk`)
- `utils.py` — grid generation (`symmetric`/`bottom` tanh, `hybrid`, `double` canopy grid), divergence, u_tau, bulk velocity, field I/O
- `turbstats.py` — `TurbulenceStats`: mean profile, Reynolds stresses, skewness, canopy drag profile, 2D spectra (legacy two-wall z⁺ plane or multi-plane via `statistics.spectra_z`); running sums + `n_samples` for restartable accumulation
- `canopy.py` — `RigidCanopyIBM`: RKPM direct-forcing IBM for rigid filamentous canopies (Monti et al. 2022); see `docs/CANOPY.md`. Only wired into the IMEX scheme; forcing applied after the implicit z-solve, before projection, never into the AB2 history.
- `control.py` — `ChannelFlowEnv`: steppable control environment (observation = wall shear stress, action = wall blowing/suction, reward = drag reduction), plus `OppositionControl` and an optional `to_gym()` adapter. IMEX only. See `examples/opposition_control/`.
- `cli.py` — `torchannel-run` / `torchannel-stats` console entry points.
- `data/reference/` — digitised MKM (1999) and Lee & Moser (2015) profiles, REMAPPED to torChannel axes (the references use y as wall-normal: their `R_vv`→`ww`, `R_ww`→`vv`, `R_uv`→`uw`). Regenerate with `scripts/fetch_reference_data.py`.

**Utility scripts**: `generate_grid.py` (grid + memory estimate from `config.yaml`, `--Re` override), `compute_gamma_from_grid.py <grid.csv>` (back-out gamma of a symmetric grid), `find_optimal_gamma.py` (edit constants in file; C1-continuity gamma sweep for hybrid grids — the `double` grid's `gamma_outer: auto` does this automatically), `visualize_structures.py` (Q-criterion isosurfaces), `scripts/plot_snapshot.py`, `scripts/plot_timeseries.py`.

**Staggered grid**: u at x-faces, v at y-faces, w at z-faces, p cell-centered; one ghost layer all around. Periodic in x,y; walls in z. Bottom wall always no-slip; top wall `boundary_conditions.top_wall.type`: `dirichlet` (no-slip, closed channel) or `neumann` (free-slip, open channel — used with the canopy). Grid arrays: `z_f` (nz+1 faces), `z_c` (nz+2 centers incl. ghosts), `dz_f` (nz cell heights = face spacing), `dz_c` (nz+1 center-to-center spacings).

**The periodic seam face is stored TWICE, and index 0 is the ghost.** `u` has shape `nx+1` in x and `apply_bc_all` does `u[0] = u[-1]`, so **`u[nx]` is the master copy of a physical face and `u[0]` is its ghost** — likewise `v[:, ny]` vs `v[:, 0]`. Any operator writing the u-RHS must cover `[1:nx+1]`, not `[1:nx]` (v: `[1:ny+1]`). Only `w` is different: `w[:, :, 0]` and `w[:, :, nz]` are the walls and must stay zero. Getting this wrong is silent — the plane still gets forcing, implicit z-diffusion and the pressure gradient, so nothing blows up; it just stops being advected, the flux form loses its telescoping property, and the operator injects `1/nx` of the applied force as spurious momentum (fixed in `47fe834`; it was worth +0.57% of `f·V` and a 0.9% error in the wall-stress balance). Faces `nx`/`ny` need a neighbour past the seam: `torch.cat([u, u[1:2]], dim=0)` supplies `u[nx+1] == u[1]`. Guarded by `test_advection_conserves_momentum_globally` and `test_every_physical_face_gets_a_right_hand_side`.

**Configuration**: YAML. `config550.yaml` = closed channel Re_τ=550; `config_canopy_monti*.yaml` = Monti canopy chain (fresh interpolate → run → stats). Key sections: `grid`, `domain`, `flow`, `boundary_conditions`, `initialization`, `time`, `solver`, `canopy`, `statistics`, `output`, `compute`.

## Key Conventions & Gotchas

- **Precision**: `compute.precision: float64` (default, bit-exact reference) | `mixed` (fp32 fields, fp64 Poisson) | `float32`. Statistics accumulators, grid metrics and the bulk-velocity controller stay fp64 in every mode. `main.py`'s `torch.set_default_dtype(torch.float64)` is left alone ON PURPOSE — it keeps every un-annotated factory call correct for free; only the large 3-D fields are threaded explicitly via `self.dtype`, with `dz_f_c`/`dz_c_c` compute-dtype metric copies. `_audit_dtypes()` asserts the layout at step 1, because a missed cast runs fp64 arithmetic and silently downcasts (correct, 2-5x slower, invisible). fp16/bf16 are rejected: `1/dz_min^2` ~6e6 overflows fp16's 65504.
- Reductions widen AFTER reducing over x,y. Passing `dtype=torch.float64` to a reduction upcasts the whole 3-D operand first (~500 MB of transient fp64 at production size).
- Performance kernels use `@torch.jit.script` (bypassed under `PYTORCH_JIT=0`) and optionally `torch.compile`. TF32 is deliberately OFF: no matmul in the hot path to accelerate, and it would degrade the RKPM moment solve.
- Boundary conditions via ghost cells (index 0 and -1); linear-reflection ghosts for Dirichlet (`u[0] = -u[1]`)
- **Statistics enable switch is `statistics.n_stats > 0`** — the `enabled:` key in some configs is IGNORED by the code
- **Open channel (`top_wall: neumann`): use `statistics.spectra_z` (physical heights), NOT `z_plus_target`.** The legacy path assumes two walls and δ=Lz/2, so it picks planes at 2x the intended z+ and pairs a near-wall plane with one at the free surface.
- Restart: set `initialization.field_file` (+ `reset_time: false`) → time/step continue, results folder preserved, `timeseries.npz` appended; restart statistics separately via `statistics.restart_state_file` (grid dims and spectra shapes are validated). `initialization.type: interpolate` + `field_file` is NOT a restart (fresh start, t=0, velocity rescaled).
- `fields.npz` is a rolling checkpoint overwritten every `n_save` steps; `output.n_snapshot > 0` archives `fields_t<time>.npz` snapshots; NaN detection saves `fields_error.npz` and aborts
- Re_τ delta convention: canopy → δ = Lz − h; open channel (neumann top) → δ = Lz; closed channel → δ = Lz/2. This is now threaded consistently: the solver passes `delta` and `top_wall_bc_type` into `TurbulenceStats`, and `plot_statistics.py` detects the open channel from `--config` (override with `--open-channel`). FIXED (was: u_tau averaged the bottom wall with the FREE SURFACE on open channels, inflating it ~7x; and δ was hard-coded Lz/2 in both files)
- Canopy runs: `canopy.h` should equal `domain.z_transition` (warning otherwise); post-process with `plot_statistics.py --canopy-height h` to get Re_τ,in/out per Monti conventions
- `rebuttal/` and `results/duct_*`, `results/herringbone*` etc. are artifacts of a separate fractal-mixing side project (LaTeX rebuttal document) — not part of the channel/canopy code path; all untracked
- `.gitignore` anchors `config.yaml` to the repo root (`/config.yaml`). Unanchored it matched at any depth and silently excluded `examples/*/config.yaml`
- Validation: `examples/re180_open/` is the reference case (open channel, Re_τ=180, seeded from CaNS `run0_theory_big/theory_dump/no_actuation_lumley/fld_0120.bin` via `scripts/cans_to_npz.py`). Use a LATE snapshot — that series spins up, and `fld_0001` sits at Re_τ=154 against a 179-187 plateau. The 207 MB seed is gitignored; regenerate per `examples/re180_open/README.md`
- Statistics accumulate as running sums divided by sample count, enabling restart without data loss
