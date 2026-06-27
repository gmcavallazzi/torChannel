# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TorChannel is a GPU-accelerated Direct Numerical Simulation (DNS) solver for incompressible turbulent channel flow, built with Python/PyTorch. It uses double precision (`torch.float64`) throughout for numerical stability.

## Commands

```bash
# Run the duct mixing campaign (baffle or surface_baffle, N-sweep)
python scripts/mixing_campaign.py --mode baffle --Sc 10 --dt 5e-4 --Ns 0 1 2

# Run on HPC cluster (SLURM)
sbatch slurm/baffle_sc100.sh

# Install dependencies
pip install -r requirements.txt

# Run a single test
python tests/test_scalar.py

# Plot the canonical campaign figures (needs `module load texlive`)
python scripts/plot_campaign_temp.py      --mode baffle --Sc 10 --Ns 0 1 2 --thr 0.7 --left-only
python scripts/plot_campaign_xsections.py --mode baffle --Sc 10 --Ns 0 1 2
```

There is no test runner framework (no pytest). Tests are standalone scripts run individually with `python tests/test_<name>.py`.

## Architecture

**Entry point**: `scripts/mixing_campaign.py` builds a config and drives `ChannelFlow`
(from `solver.py`) over an N-sweep (the duct mixing study). This is a duct-only repo —
the generic `main.py` channel driver and channel post-processing live in the public repo.

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

**Configuration**: YAML files (see `configs/duct_koch.yaml` for a duct example, or the in-code `base_config` in `scripts/mixing_campaign.py`). Key sections: `grid`, `domain`, `flow`, `time`, `initialization`, `scalar`, `immersed`.

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
