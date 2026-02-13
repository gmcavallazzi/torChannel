# RKPM Workflow Summary

## New Architecture

RKPM coefficient computation has been separated from the main simulation into a standalone script that supports **MPI parallelization**.

## Files

1. **`compute_rkpm_coefficients.py`** - Standalone precomputation script (uses MPI)
2. **`ibm.py`** - Modified to load precomputed coefficients
3. **`config_canopy_monti.yaml`** - Updated to specify coefficient file paths
4. **`RKPM_precomputation_guide.md`** - Complete documentation

## Quick Workflow

### First Time: Precompute Coefficients

```bash
# Serial (8-12 minutes for 124k points)
python compute_rkpm_coefficients.py config_canopy_monti.yaml

# Parallel with MPI (1-2 minutes with 8 cores!)
mpirun -np 8 python compute_rkpm_coefficients.py config_canopy_monti.yaml
```

This generates:
- `results_canopy_monti/rkpm_coefficients.npz` (~28 MB)
- `results_canopy_monti/rkpm_epsilon.npz` (~3 MB)

### Every Run: Use Precomputed Files

```bash
python main.py config_canopy_monti.yaml
```

Loads coefficients in **< 5 seconds** instead of recomputing!

## Config File Changes

**Old (removed):**
```yaml
ibm:
  rkpm_cache_file: "..."
  rkpm_force_recompute: false
```

**New (required):**
```yaml
ibm:
  rkpm_coefficients_file: "results_canopy_monti/rkpm_coefficients.npz"
  rkpm_epsilon_file: "results_canopy_monti/rkpm_epsilon.npz"
  rkpm_basis: "linear"      # Used by compute_rkpm_coefficients.py
  rkpm_search_range: 4      # Used by compute_rkpm_coefficients.py
```

## Key Advantages

1. **MPI Parallelization**: ~8x speedup with 8 cores
2. **Cleaner Separation**: Precomputation is separate from simulation
3. **No More Waiting**: Subsequent runs start instantly
4. **Explicit Control**: You decide when to recompute
5. **Reproducibility**: Coefficient files can be version controlled

## MPI Requirements

**Install mpi4py:**
```bash
# Conda (recommended, includes MPI)
conda install -c conda-forge mpi4py

# Or pip (requires MPI already installed)
pip install mpi4py
```

**Install MPI (if needed):**
- Mac: `brew install open-mpi`
- Ubuntu: `apt-get install libopenmpi-dev`
- CentOS: `yum install openmpi-devel`

## When to Recompute

Recompute when you change:
- Grid resolution (nx, ny, nz)
- Domain size (Lx, Ly, Lz)
- Canopy geometry (spacing, diameter, height)
- RKPM settings (basis, search_range)

The code automatically detects mismatches and will error with instructions.

## Performance Summary

| Method | Time (124k points) |
|--------|-------------------|
| Old inline (serial) | 8-12 min per run |
| New precompute (serial) | 8-12 min once, then < 5 sec per run |
| New precompute (8 cores) | 1-2 min once, then < 5 sec per run |

## Backward Compatibility

**Breaking change:** The old `rkpm_cache_file` parameter is no longer supported. You must:

1. Run `compute_rkpm_coefficients.py` once
2. Update config to use `rkpm_coefficients_file` and `rkpm_epsilon_file`

## Example: Complete Fresh Start

```bash
# 1. Precompute with MPI (1-2 minutes)
mpirun -np 8 python compute_rkpm_coefficients.py config_canopy_monti.yaml

# 2. Run simulation (instant startup)
python main.py config_canopy_monti.yaml

# 3. Run again (instant startup, reuses coefficients)
python main.py config_canopy_monti.yaml

# 4. Change grid resolution in config, must recompute
# Edit config_canopy_monti.yaml: nx: 288 -> 384
mpirun -np 8 python compute_rkpm_coefficients.py config_canopy_monti.yaml

# 5. Run with new resolution (instant startup)
python main.py config_canopy_monti.yaml
```

## See Also

- **RKPM_precomputation_guide.md** - Complete documentation with examples
- **config_canopy_monti.yaml** - Example configuration
- **compute_rkpm_coefficients.py** - Precomputation script source
