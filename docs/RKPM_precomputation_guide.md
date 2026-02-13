# RKPM Coefficient Precomputation Guide

## Overview

RKPM (Reproducing Kernel Particle Method) coefficient computation is expensive for large numbers of Lagrangian points (e.g., 124k points for canopy simulation). To avoid recomputing these coefficients every time you run a simulation, we use a **separate precomputation script** that:

1. Computes RKPM coefficients offline (once)
2. Saves them to NPZ files
3. Supports **MPI parallelization** for massive speedup
4. Main simulation loads precomputed files instantly

## Quick Start

### Step 1: Precompute Coefficients

**Serial (single core):**
```bash
python compute_rkpm_coefficients.py config_canopy_monti.yaml
```

**Parallel (MPI, 8 cores):**
```bash
mpirun -np 8 python compute_rkpm_coefficients.py config_canopy_monti.yaml
```

Expected output:
```
================================================================================
RKPM COEFFICIENT PRECOMPUTATION (MPI)
================================================================================

Config file: config_canopy_monti.yaml
MPI ranks: 8

Generating grid and Lagrangian points...
✓ Grid: 288×216×150
✓ Lagrangian points: 124416
✓ Configuration hash: a1b2c3d4...

[Rank 0] Computing RKPM for u-velocity...
  Basis: linear (4 terms)
  Search range: 4
  Total Lagrangian points: 124416
  MPI ranks: 8
  Points per rank: 15552

[Rank 0] Progress: 15552/15552 (100.0%)
[Rank 1] Progress: 15552/15552 (100.0%)
...
[Rank 0] ✓ Completed RKPM for u-velocity

[Similar for v-velocity and w-velocity]

================================================================================
COMPUTING EPSILON VALUES
================================================================================

Computing epsilon for u-velocity...
  Building sparse matrix A...
  Matrix A: 124416×124416, 15387204 nonzeros
  Solving A * eps = 1...
  ✓ Epsilon computed (range: [8.234e-01, 1.156e+00])

[Similar for v-velocity and w-velocity]

================================================================================
SAVING RESULTS
================================================================================

Flattening support structures...
Saving coefficients to results_canopy_monti/rkpm_coefficients.npz...
✓ Saved coefficients (28.3 MB)

Saving epsilon values to results_canopy_monti/rkpm_epsilon.npz...
✓ Saved epsilon values (2.8 MB)

================================================================================
PRECOMPUTATION COMPLETE
================================================================================

Generated files:
  1. results_canopy_monti/rkpm_coefficients.npz (28.3 MB)
  2. results_canopy_monti/rkpm_epsilon.npz (2.8 MB)

Configuration hash: a1b2c3d4...
```

### Step 2: Update Config

The config file already points to these files:
```yaml
ibm:
  rkpm_coefficients_file: "results_canopy_monti/rkpm_coefficients.npz"
  rkpm_epsilon_file: "results_canopy_monti/rkpm_epsilon.npz"
```

### Step 3: Run Simulation

```bash
python main.py config_canopy_monti.yaml
```

The simulation will load the precomputed coefficients in **< 5 seconds** instead of recomputing them!

## Performance

| Method | Cores | Time (124k points) | Speedup |
|--------|-------|-------------------|---------|
| **Serial** | 1 | ~8-12 min | 1x |
| **MPI** | 4 | ~2-3 min | 4x |
| **MPI** | 8 | ~1-1.5 min | 8x |
| **MPI** | 16 | ~30-45 sec | 16x |

**Note:** Speedup is nearly linear with number of cores (excellent scaling!)

## Configuration Options

### RKPM Basis

In `config_canopy_monti.yaml`:
```yaml
ibm:
  rkpm_basis: "linear"  # or "quadratic"
```

- **linear** (4 terms): Faster, good accuracy for most applications [Recommended]
- **quadratic** (10 terms): Slower, highest accuracy

### Search Range

```yaml
ibm:
  rkpm_search_range: 4  # Grid cells, range 3-6
```

- **3**: Minimal neighbors, fastest, may reduce accuracy
- **4**: Good balance [Recommended]
- **6**: Maximum neighbors, slower, highest accuracy

## File Outputs

### rkpm_coefficients.npz

Contains interpolation/spreading weights for all Lagrangian points:
- `u_ix`, `u_iy`, `u_iz`: Grid indices for u-velocity
- `u_wdt`: RKPM weights for u-velocity
- `u_vol`: Volume elements for u-velocity
- `u_n_neighbors`: Number of neighbors per Lagrangian point
- Similar arrays for v-velocity and w-velocity
- `config_hash`: MD5 hash of configuration parameters
- `n_lag`: Total number of Lagrangian points

**Size:** ~20-30 MB for 124k points (linear basis)

### rkpm_epsilon.npz

Contains epsilon values for partition of unity correction:
- `epsilon_u`: Epsilon for u-velocity (shape: n_lag)
- `epsilon_v`: Epsilon for v-velocity (shape: n_lag)
- `epsilon_w`: Epsilon for w-velocity (shape: n_lag)
- `config_hash`: Configuration hash
- `n_lag`: Total number of Lagrangian points

**Size:** ~3-5 MB for 124k points

## When to Recompute

Recompute coefficients when you change:
- Grid resolution (nx, ny, nz)
- Domain size (Lx, Ly, Lz)
- Grid stretching parameters (gamma, z_transition, etc.)
- Lagrangian point count or distribution
- Canopy geometry (spacing, diameter, height, seed)
- RKPM settings (basis, search_range)

The configuration hash automatically tracks these changes. If you try to use incompatible coefficients, the simulation will error and tell you to recompute.

## Troubleshooting

### "RKPM coefficient files not specified in config"

Add to your config:
```yaml
ibm:
  rkpm_coefficients_file: "results_canopy_monti/rkpm_coefficients.npz"
  rkpm_epsilon_file: "results_canopy_monti/rkpm_epsilon.npz"
```

### "File not found"

Run the precomputation script:
```bash
python compute_rkpm_coefficients.py config_canopy_monti.yaml
```

### "Number of Lagrangian points mismatch"

Your config has changed since the coefficients were computed. Recompute:
```bash
mpirun -np 8 python compute_rkpm_coefficients.py config_canopy_monti.yaml
```

### "ImportError: No module named 'mpi4py'"

Install mpi4py:
```bash
# Option 1: Conda (recommended, includes MPI)
conda install -c conda-forge mpi4py

# Option 2: Pip (requires MPI already installed)
pip install mpi4py
```

**Note:** You need MPI installed on your system. On Mac: `brew install open-mpi`. On Linux: `apt-get install libopenmpi-dev` or `yum install openmpi-devel`.

### Serial computation is slow

Use MPI! Even 4 cores gives ~4x speedup:
```bash
mpirun -np 4 python compute_rkpm_coefficients.py config_canopy_monti.yaml
```

### MPI not scaling well

Check:
1. Number of cores ≤ number of Lagrangian points (124k points supports up to 124k cores theoretically!)
2. CPU cores are physical, not hyper-threaded
3. No resource contention from other processes

## Advanced: Inspecting Coefficients

```python
import numpy as np

# Load coefficients
data = np.load("results_canopy_monti/rkpm_coefficients.npz")
print(f"Lagrangian points: {data['n_lag']}")
print(f"Config hash: {data['config_hash']}")
print(f"Neighbors per point (u): min={data['u_n_neighbors'].min()}, "
      f"max={data['u_n_neighbors'].max()}, mean={data['u_n_neighbors'].mean():.1f}")

# Load epsilon
eps_data = np.load("results_canopy_monti/rkpm_epsilon.npz")
print(f"Epsilon u: min={eps_data['epsilon_u'].min():.3e}, "
      f"max={eps_data['epsilon_u'].max():.3e}, mean={eps_data['epsilon_u'].mean():.3e}")
```

## Comparison: Old vs New Workflow

### Old Workflow (Inline Computation)
```
Run simulation → Compute RKPM (8-12 min) → Time-stepping → Results
Every restart: 8-12 min wait
```

### New Workflow (Precomputed)
```
First time:
  Precompute RKPM (mpirun): 1-2 min → Save NPZ files

Every simulation:
  Run simulation → Load NPZ (< 5 sec) → Time-stepping → Results
Every restart: < 5 sec to start
```

**Advantage:** Once precomputed, you can run hundreds of simulations instantly without recomputation!

## Best Practices

1. **Use MPI** for first computation (8 cores recommended)
2. **Keep coefficient files** in version control or backup (small enough)
3. **Separate coefficients** for different configs (use different filenames)
4. **Document config hash** in your simulation notes for reproducibility
5. **Reuse coefficients** across parameter sweeps that don't change grid/geometry

## MPI Tips

### Running on HPC Cluster

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --time=00:10:00

module load openmpi
mpirun -np 32 python compute_rkpm_coefficients.py config_canopy_monti.yaml
```

### Running Locally

```bash
# Use all available cores
mpirun -np $(nproc) python compute_rkpm_coefficients.py config_canopy_monti.yaml

# Or specify explicitly
mpirun -np 8 python compute_rkpm_coefficients.py config_canopy_monti.yaml
```

### Checking MPI Installation

```bash
# Check MPI version
mpirun --version

# Test with 4 ranks
mpirun -np 4 python -c "from mpi4py import MPI; print(f'Rank {MPI.COMM_WORLD.rank} of {MPI.COMM_WORLD.size}')"
```

Expected output:
```
Rank 0 of 4
Rank 1 of 4
Rank 2 of 4
Rank 3 of 4
```
