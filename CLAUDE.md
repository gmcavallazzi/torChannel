# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TorChannel is a GPU-accelerated Direct Numerical Simulation (DNS) solver for incompressible turbulent channel flow, built with Python/PyTorch. It uses double precision (`torch.float64`) throughout for numerical stability.

## Commands

```bash
# Run a simulation
python main.py config550.yaml

# Run on HPC cluster (SLURM)
sbatch launch.sh

# Install dependencies
pip install -r requirements.txt

# Run a single test
python tests/test_derivative.py

# Post-process results
python post_process.py results/fields.npz --config config550.yaml
python plot_statistics.py results/turbulence_stats.npz --config config550.yaml
```

There is no test runner framework (no pytest). Tests are standalone scripts run individually with `python tests/test_<name>.py`.

## Architecture

**Entry point**: `main.py` → creates `ChannelFlow` (from `solver.py`) and calls `run_simulation()`.

**Core modules**:
- `solver.py` — Main `ChannelFlow` class: time-stepping loop, IMEX/AB2/FE schemes, adaptive dt, restart logic, bulk velocity forcing
- `operators.py` — Spatial discretization operators (advection, diffusion) on a staggered grid. Uses `@torch.jit.script` for fused GPU kernels
- `scalar.py` — Optional passive-scalar transport (concentration in [0,1]) at cell centres: conservative flux-form advection + IMEX diffusion (D = nu/Sc), periodic in x,y, configurable wall BC in z (`neumann` no-flux for mixing/decay studies, default; or `dirichlet`). Enabled via a `scalar:` config block; advanced each step by `ChannelFlow.advance_scalar`. Verified in `tests/test_scalar.py` (erf diffusion, mean conservation, zero numerical diffusion).
- `projection_fft.py` — FFT-based Poisson solver for pressure projection (primary solver)
- `projection.py` — Direct Poisson solver (fallback/testing)
- `initflow.py` — Flow field initialization (vortices, random, parabolic, laminar, or restart from file)
- `utils.py` — Grid generation (hyperbolic tangent stretching), I/O, diagnostic utilities
- `statistics.py` — `TurbulenceStats` class: on-the-fly Reynolds stresses, mean profiles, 2D energy spectra

**Staggered grid layout**: Velocities are staggered (u at x-faces, v at y-faces, w at z-faces). Pressure is cell-centered. Periodic in x,y; wall-bounded in z with no-slip or free-slip BCs.

**Configuration**: YAML files (see `config550.yaml` for Re_τ=550 example). Key sections: `grid`, `domain`, `flow`, `time`, `initialization`, `statistics`.

**Output**: `.npz` files for fields, timeseries, and statistics checkpoints. Restart is supported by pointing `initialization.field_file` to a saved checkpoint.

## Project context (MERGE fractal-mixing study)

This repo is being used to test the MERGE proposal on fractal-interface mixing
enhancement. The passive-scalar feature (`scalar.py`), the Koch fractal initial
condition, the mixing driver, current results, the key architectural constraints,
and the paused next steps are documented in **`docs/MERGE_CONTEXT.md`** — read it
first when resuming this work (it is the portable substitute for the workstation-local
Claude memory, which does not travel with the repo).

## Key Conventions

- All tensor operations use PyTorch with `torch.float64` default dtype
- Performance-critical operators use `@torch.jit.script` decorators
- Grid is non-uniform in wall-normal (z) direction via `gamma` stretching parameter
- Boundary conditions are applied via ghost cells (index 0 and -1 in z)
- Statistics accumulate as running sums divided by sample count, enabling restart without data loss
