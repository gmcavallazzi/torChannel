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
- `operators.py` — Spatial discretization operators (advection, diffusion) on a staggered grid. Uses `@torch.jit.script` for fused GPU kernels. Includes pluggable tridiagonal solver interface (`_ext` wrappers)
- `projection_fft.py` — FFT-based Poisson solver for pressure projection (primary solver)
- `projection.py` — Direct Poisson solver (fallback/testing)
- `tridiagonal_cusparse.py` — cuSPARSE GPU tridiagonal solver wrapper (ctypes). Optional alternative to the Thomas algorithm
- `initflow.py` — Flow field initialization (vortices, random, parabolic, laminar, or restart from file)
- `utils.py` — Grid generation (hyperbolic tangent stretching), I/O, diagnostic utilities
- `statistics.py` — `TurbulenceStats` class: on-the-fly Reynolds stresses, mean profiles, 2D energy spectra

**Staggered grid layout**: Velocities are staggered (u at x-faces, v at y-faces, w at z-faces). Pressure is cell-centered. Periodic in x,y; wall-bounded in z with no-slip or free-slip BCs.

**Configuration**: YAML files (see `config550.yaml` for Re_τ=550 example). Key sections: `grid`, `domain`, `flow`, `time`, `initialization`, `statistics`, `solver`.

**Tridiagonal solver**: The `solver.tridiagonal` config option selects the tridiagonal solver used for both implicit z-diffusion and the FFT Poisson solve. Options: `"thomas"` (default, JIT-compiled) or `"cusparse"` (GPU-accelerated via NVIDIA cuSPARSE). Example:
```yaml
solver:
  type: "fft"
  tridiagonal: "cusparse"  # or "thomas" (default)
```

**Output**: `.npz` files for fields, timeseries, and statistics checkpoints. Restart is supported by pointing `initialization.field_file` to a saved checkpoint.

## Key Conventions

- All tensor operations use PyTorch with `torch.float64` default dtype
- Performance-critical operators use `@torch.jit.script` decorators
- Grid is non-uniform in wall-normal (z) direction via `gamma` stretching parameter
- Boundary conditions are applied via ghost cells (index 0 and -1 in z)
- Statistics accumulate as running sums divided by sample count, enabling restart without data loss
