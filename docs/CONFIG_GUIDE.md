# TorChannel Configuration Guide

This document provides a comprehensive reference for all configuration parameters in `config.yaml`.

---

## Table of Contents

1. [Configuration File Structure](#configuration-file-structure)
2. [Grid Parameters](#grid-parameters)
3. [Domain Size](#domain-size)
4. [Flow Parameters](#flow-parameters)
5. [Initialization](#initialization)
6. [Poisson Solver](#poisson-solver)
7. [Time Integration](#time-integration)
8. [Compute Device](#compute-device)
9. [Output Control](#output-control)
10. [Turbulence Statistics](#turbulence-statistics)
11. [Example Configurations](#example-configurations)
12. [Best Practices](#best-practices)

---

## Configuration File Structure

TorChannel uses YAML format for configuration. The file is organized into logical sections:

```yaml
grid:          # Grid resolution
domain:        # Physical domain size
flow:          # Flow parameters
initialization: # Initial conditions
solver:        # Poisson solver selection
time:          # Time integration settings
compute:       # CPU/GPU selection
output:        # Output control
statistics:    # Turbulence statistics
```

---

## Grid Parameters

Defines the computational grid resolution.

```yaml
grid:
  nx: 192    # Number of grid points in streamwise (x) direction
  ny: 192    # Number of grid points in spanwise (y) direction
  nz: 128    # Number of grid points in wall-normal (z) direction
```

### Parameters

- **nx** (integer): Streamwise grid points
  - Periodic direction
  - Must be even for FFT solver

- **ny** (integer): Spanwise grid points
  - Periodic direction
  - Must be even for FFT solver

- **nz** (integer): Wall-normal grid points
  - Non-uniform stretched grid
  - Points cluster near walls (z=0, z=Lz)

### Guidelines

- Grid spacing in wall units should satisfy:
  - $\Delta x^+ < 11$ (streamwise)
  - $\Delta y^+ < 6$ (spanwise)
  - $\Delta z^+ < 0.5$ near wall (wall-normal)

---

## Domain Size

Defines the physical domain dimensions.

```yaml
domain:
  Lx: 12.567  # Streamwise domain length (e.g., 4π)
  Ly: 4.189   # Spanwise domain length (e.g., 4π/3)
  Lz: 2.0     # Wall-normal domain height (channel full height, 2δ)
```

### Parameters

- **Lx** (float): Streamwise length
  - Periodic boundary conditions
  - Typical values: $2\pi\delta$ to $8\pi\delta$ (where $\delta = L_z/2$)
  - Larger domains capture longer wavelength structures

- **Ly** (float): Spanwise length
  - Periodic boundary conditions
  - Typical values: $\pi\delta$ to $4\pi\delta$
  - Must be large enough for spanwise correlations to decay

- **Lz** (float): Channel full height ($2\delta$)
  - Half-height $\delta = L_z/2$ is the characteristic length scale
  - No-slip walls at $z=0$ and $z=L_z$
  - Typical value: $L_z = 2.0$ (normalized by $\delta=1.0$)

---

## Flow Parameters

Defines the Reynolds number and flow characteristics.

```yaml
flow:
  Re: 2870.0      # Bulk Reynolds number (Re = U_bulk * Lz / ν)
  Re_tau: 180.0   # Target friction Reynolds number (Re_τ = u_τ * δ / ν)
  U_bulk: 1.0     # Target bulk velocity
  gamma: 2.6      # Grid stretching parameter
```

### Parameters

- **Re** (float): Bulk Reynolds number
  - Defined as: $Re = U_{\text{bulk}} \times \delta / \nu$
  - Determines kinematic viscosity: $\nu = U_{\text{bulk}} \times \delta / Re$
  - Example: $Re = 2870$ for $Re_\tau = 180$ at $L_z = 2.0$

- **Re_tau** (float): Target friction Reynolds number
  - Defined as: $Re_\tau = u_\tau \times \delta / \nu$ (where $\delta = L_z/2$)
  - Characterizes turbulence intensity
  - Common values: 180, 395, 550, 1000
  - Used for computing wall units ($z^+$) and initializing statistics

- **U_bulk** (float): Target bulk (mean) velocity
  - Maintained constant by adjusting pressure gradient forcing
  - Typically normalized to 1.0
  - Actual value: $U_{\text{bulk}} = \frac{1}{V} \iiint u \, dx \, dy \, dz$, where $V = L_x \times L_y \times L_z$

- **gamma** (float): Grid stretching parameter
  - Controls near-wall grid clustering
  - Uses hyperbolic tangent stretching: $z(\xi) = \frac{L_z}{2} \left[1 + \frac{\tanh(\gamma\xi)}{\tanh(\gamma)}\right]$
  - Larger $\gamma$ → more clustering near walls
  - Typical range: 1.5-3.5

### Relationship Between Re and Re_τ

For turbulent channel flow:
- $Re_\tau \approx 0.09 \times Re^{0.88}$ (empirical correlation)
- Example: $Re_\tau = 180 \rightarrow Re \approx 2870$

---

## Initialization

Controls how the flow field is initialized.

```yaml
initialization:
  # Initial flow field type (used when field_file is not specified)
  type: "vortices"  # Options: "vortices", "random", "parabolic", "laminar"

  # Parameters for vortex initialization
  perturbation_intensity: 0.08  # Amplitude of velocity perturbations
  n_vortices: 4                 # Number of counter-rotating vortex pairs

  # RESTART FROM CHECKPOINT (uncomment to enable)
  field_file: "results/fields.npz"  # Path to saved flow field
  reset_time: false                  # false: continue time, true: reset to t=0
```

### Initialization Types

#### 1. Vortices (Recommended)
```yaml
type: "vortices"
perturbation_intensity: 0.08
n_vortices: 4
```
- Base parabolic profile + counter-rotating vortex pairs
- Vortices generated using streamfunction
- Promotes faster transition to turbulence
- **perturbation_intensity**: Amplitude relative to U_bulk (0.05-0.15)
- **n_vortices**: Number of vortex pairs in spanwise direction (2-8)

#### 2. Parabolic
```yaml
type: "parabolic"
```
- Laminar parabolic profile: $u(z) = U_{\text{max}} \left[1 - \left(\frac{2z}{L_z} - 1\right)^2\right]$
- U_max = 1.5 × U_bulk (for channel flow)
- Useful for testing

#### 3. Random
```yaml
type: "random"
perturbation_intensity: 0.1
```
- Random velocity perturbations
- Fast transition but initially chaotic
- Requires longer settling time

#### 4. Laminar
```yaml
type: "laminar"
```
- Uniform velocity U = U_bulk
- Primarily for testing

### Reproducibility

- **seed** (int, optional): seeds the random perturbation added by the
  `parabolic`/`uniform` initializers. Without it the draw comes from PyTorch's
  global generator and a fresh start is **not** repeatable run to run. Ignored
  when restarting or interpolating, since those take the field from a file.

```yaml
initialization:
  type: "vortices"
  perturbation_intensity: 0.05
  seed: 20260803
```

Note the canopy has its own independent `canopy.seed` for filament placement.

### Restart from Checkpoint

To restart from a saved flow field:

```yaml
field_file: "results/fields.npz"
reset_time: false
```

- **field_file** (string): Path to checkpoint file
  - Can be: `fields.npz`, `fields_final.npz`, or any saved field
  - Loads: u, v, w, p, step, time

- **reset_time** (boolean):
  - `false`: Continue from saved time and step number
  - `true`: Reset time to 0 and step to 0 (keeps flow field)

**Important**: On restart, the results folder is preserved (no cleaning occurs). Timeseries data is automatically appended.

---

## Poisson Solver

Selects the Poisson solver for pressure projection.

```yaml
solver:
  type: "fft"  # Options: "fft" (recommended), "direct"
```

### Parameters

- **type** (string): Solver algorithm
  - **"fft"** (recommended): FFT-based solver
    - Fast: O(N log N) complexity
    - Uses modified wavenumbers for 2nd-order accuracy
    - Requires nx, ny to be even
    - Scales well on GPU

  - **"direct"**: Direct sparse matrix solver
    - Fallback for testing/validation
    - Slower: O(N²) to O(N^1.5) depending on method
    - Works for any grid size
    - Not recommended for production runs

### Recommendation

Always use `type: "fft"` unless:
- You need to validate against direct solver
- You're running very small test cases
- You need non-standard boundary conditions

---

## Time Integration

Controls timestepping and integration scheme.

```yaml
time:
  dt: 0.005               # Initial timestep
  n_steps: 2000000000     # Maximum number of steps
  t_max: 1000.0           # Maximum physical time

  # Adaptive timestepping
  CFL_target: 0.25        # Target CFL number
  dt_update_interval: 10  # Update dt every N steps (0 = disable)
  dt_max: 0.011           # Maximum allowed timestep
  dt_min: 0.002           # Minimum allowed timestep

  # Time integration scheme
  scheme: "IMEX"  # Options: "IMEX", "AB2", "FE"
```

### Basic Parameters

- **dt** (float): Initial timestep
  - Will be adjusted if adaptive timestepping is enabled
  - Smaller Re_τ → larger dt possible

- **n_steps** (integer): Maximum number of timesteps
  - Simulation stops when step > n_steps OR time ≥ t_max (whichever comes first)
  - Set large value (e.g., 2000000000) if using t_max as primary stopping criterion

- **t_max** (float): Maximum physical time
  - Simulation stops when time ≥ t_max OR step > n_steps (whichever comes first)
  - For statistics: need ~500-1000 δ/U_bulk time units after settling
  - Set large value if using n_steps as primary stopping criterion

### Adaptive Timestepping

- **CFL_target** (float): Target CFL number
  - $\text{CFL} = \Delta t \times \left(\frac{|u|}{\Delta x} + \frac{|v|}{\Delta y} + \frac{|w|}{\Delta z}\right)$
  - Typical range: 0.2-0.5
  - Smaller CFL → more stable but slower
  - Automatically adjusts $\Delta t$ to maintain target CFL

- **dt_update_interval** (integer): Update frequency
  - Check and update dt every N steps
  - Typical value: 10
  - Set to 0 to disable adaptive timestepping (use fixed dt)

- **dt_max** (float): Maximum timestep
  - Safety limit for adaptive timestepping
  - Prevents dt from growing too large

- **dt_min** (float): Minimum timestep
  - Safety limit for adaptive timestepping
  - If CFL requires dt < dt_min, simulation will warn or stop

### Time Integration Schemes

#### IMEX (Recommended)
```yaml
scheme: "IMEX"
```
- **I**mplicit-**Ex**plicit scheme
- **Implicit**: Wall-normal (z) diffusion (2nd-order Crank-Nicolson)
- **Explicit**: Advection + xy-diffusion (Adams-Bashforth 2)
- **Advantages**:
  - No timestep restriction from z-diffusion (can use larger dt)
  - Stable for stretched grids with fine near-wall spacing
  - Recommended for all production runs
- **Stability**: $\text{CFL} < 0.5$ typically safe

#### Adams-Bashforth 2 (AB2)
```yaml
scheme: "AB2"
```
- Fully explicit 2nd-order scheme
- All terms (advection + diffusion) treated explicitly
- **Advantages**: Simple, fast per step
- **Disadvantages**:
  - Stricter timestep restriction (especially z-diffusion)
  - Requires smaller dt for stretched grids
- **Stability**: $\text{CFL} < 0.2$ recommended for safety
- **Use case**: Testing, comparison with IMEX

#### Forward Euler (FE)
```yaml
scheme: "FE"
```
- Simple 1st-order explicit scheme
- Fully explicit treatment of all terms
- **Disadvantages**:
  - Low accuracy (1st-order)
  - Strict stability limits
  - Requires very small dt
- **Use case**: Basic testing and debugging only, not recommended for production

### Recommendations

- **For production runs**: Use `scheme: "IMEX"` with adaptive timestepping
  ```yaml
  scheme: "IMEX"
  dt: 0.005
  CFL_target: 0.25
  dt_update_interval: 10
  ```

- **For maximum stability**: Lower CFL_target to 0.2

- **For testing**: Can use `scheme: "AB2"` with smaller dt

---

## Compute Device

Selects CPU or GPU, and the working precision.

```yaml
compute:
  device: "auto"        # Options: "cuda", "cpu", "auto"
  precision: "float64"  # Options: "float64", "mixed", "float32"
```

### Parameters

- **device** (string): Compute device
  - **"auto"** (recommended): Automatically use GPU if available, otherwise CPU
  - **"cuda"**: Force GPU (will error if CUDA not available)
  - **"cpu"**: Force CPU (useful for debugging)

- **precision** (string, default `"float64"`): working precision for the
  velocity and pressure fields and the operators acting on them.

  | value | fields | Poisson solve | typical use |
  |---|---|---|---|
  | `float64` | float64 | float64 | **default**; the reference path, bit-exact |
  | `mixed` | float32 | float64 | production on memory- or time-limited hardware |
  | `float32` | float32 | float32 | maximum speed, validate before trusting |

  Statistics accumulators, grid metrics (`z_c`, `z_f`, `dz_c`, `dz_f`), the
  bulk-velocity integral and the forcing controller stay **float64 in every
  mode** — they are cheap and are where error would accumulate.

  `mixed` keeps the pressure solve in float64 because the Poisson wall-normal
  operator is ill-conditioned at the lowest horizontal wavenumbers, where
  float32 costs ~1e-5 relative error in the largest-scale pressure. Measured on
  the Re_τ=180 case, the divergence floor is ~7e-6 under `mixed` against ~2.4e-5
  under `float32` (and ~4e-14 in float64).

  **Expect ~2× faster, not 64×.** These kernels are memory-bandwidth bound, so
  halving the bytes is what you collect; the fp64:fp32 arithmetic ratio barely
  enters. Memory does halve, so ~1.26× the linear resolution fits on the same
  card — often the more useful half. Regenerate the numbers with
  `python tests/bench_precision.py`.

  **float16/bfloat16 are rejected**, with an explanation: `1/dz_min²` reaches
  ~6e6 on a Re_τ=550 grid, against an fp16 maximum of 65504.

  A note on trusting reduced precision: at Re_τ=180 the flow's own Lyapunov
  time means *any* two runs diverge pointwise within a few eddy turnovers, so
  pointwise agreement is not the test. Compare statistics (u_τ, U⁺, the stress
  profiles, the total-stress balance) over a converged window instead.

### GPU Requirements

- NVIDIA GPU with CUDA support
- PyTorch installed with CUDA
- Sufficient GPU memory:

### Performance

- GPU provides significant speedup (typically 10-50× vs CPU)
- Speedup increases with grid size
- Small grids (<64³) may not benefit so much from GPU

---

## Output Control

Controls diagnostic output and file saving.

```yaml
output:
  results_folder: results  # Output directory
  n_out: 200              # Print diagnostics every n_out steps
  n_save: 4000            # Save flow fields every n_save steps
```

### Parameters

- **results_folder** (string): Output directory path
  - Created automatically if doesn't exist
  - Can be absolute or relative path
  - On fresh start: folder is cleaned
  - On restart: folder is preserved

- **n_out** (integer): Diagnostic print interval
  - Print timestep, time, CFL, u_bulk, u_τ, forcing every n_out steps
  - Typical values: 100-500
  - Also triggers timeseries data collection

- **n_save** (integer): Field save interval
  - Save `fields.npz` checkpoint every n_save steps
  - Also saves `timeseries.npz` and `turbulence_stats_state.npz`
  - Typical values: 1000-10000
  - Smaller → more frequent checkpoints but larger storage

### Output Files

Created in `results_folder/`:

- **fields_init.npz**: Initial flow field
- **fields.npz**: Latest checkpoint (overwritten every n_save steps)
- **fields_final.npz**: Final flow field (at end of simulation)
- **timeseries.npz**: Time history (appended every n_save steps)
- **turbulence_stats_state.npz**: Statistics checkpoint (if enabled)
- **turbulence_stats.npz**: Final averaged statistics (if enabled, at end)
- **grid_plot.png**: Visualization of stretched grid
- **grid_info.csv**: Grid coordinates and spacings

---

## Turbulence Statistics

Controls collection of turbulence statistics.

```yaml
statistics:
  enabled: true  # Enable/disable statistics collection

  # Collection parameters
  n_stats: 200     # Collect statistics every n_stats steps
  t_stats: 50.0    # Start collecting after this time

  # Output files
  output_file: "turbulence_stats.npz"         # Final averaged statistics
  state_file: "turbulence_stats_state.npz"    # Checkpoint file

  # Spectral parameters
  z_plus_target: 15.0  # Wall distance in wall units for 2D spectra

  # RESTART (optional)
  restart_state_file: "results/turbulence_stats_state.npz"
```

### Basic Parameters

> **There is no `enabled` key.** Some older configs carry one; the code ignores
> it completely. Statistics are enabled by **`n_stats > 0`** and disabled by
> `n_stats: 0`. A config that sets `enabled: true` but omits `n_stats` collects
> nothing, silently.

- **n_stats** (integer): Collection interval — **this is the on/off switch**
  - `0` disables statistics entirely
  - Collect statistics every n_stats steps
  - Should be divisible by n_out for efficiency

- **t_stats** (float): Start time
  - Begin collecting statistics after time > t_stats
  - Typical: t_stats ≈ 50-100 if the flow is initialized with `vortices`

### Output Files

- **output_file** (string): Final statistics filename
  - Saved at end of simulation in results_folder
  - Contains time-averaged quantities divided by n_samples
  - Default: `"turbulence_stats.npz"`

- **state_file** (string): Checkpoint filename
  - Saved every n_save steps in results_folder
  - Contains running sums (NOT yet averaged)
  - Allows restart without losing accumulated statistics
  - Default: `"turbulence_stats_state.npz"`

### Spectral Parameters

Two modes. Prefer `spectra_z`.

- **spectra_z** (list of floats, recommended): PHYSICAL heights at which to
  accumulate 2D premultiplied energy spectra. Each height gets its own
  spectrum; no wall-mirroring, no assumption about the number of walls.

  ```yaml
  statistics:
    spectra_z: [0.0833, 0.25, 0.5]   # z+ = 15 at Re_tau=180 with delta=1
  ```

- **z_plus_target** (float, legacy): height in wall units for the 2D spectra.

  > **Do not use this on an open channel or a canopy run.** This path assumes a
  > closed channel with two no-slip walls: it hard-codes δ = L_z/2 and averages a
  > "bottom wall" plane with a "top wall" plane. With δ = L_z (free-slip top) it
  > therefore selects a plane at **half** the requested z⁺ and pairs it with one
  > at the free surface. On the Re_τ=180 open-channel case, `z_plus_target: 15.0`
  > actually lands on z⁺ = 7.4.

  It remains correct for a closed channel with `top_wall.type: dirichlet` and a
  symmetric grid.


## Example Configurations

### Re_τ = 180 (Wall-Resolved)

```yaml
grid:
  nx: 192
  ny: 192
  nz: 128

domain:
  Lx: 12.567  # 4π
  Ly: 4.189   # 1.5π
  Lz: 2.0

flow:
  Re: 2870.0
  Re_tau: 180.0
  U_bulk: 1.0
  gamma: 2.6

time:
  dt: 0.005
  CFL_target: 0.25
  dt_update_interval: 10
  scheme: "IMEX"

statistics:
  enabled: true
  n_stats: 200
  t_stats: 50.0
```

---

## Best Practices

### Grid Resolution

1. **Check grid spacing in wall units:**
   ```bash
   python generate_grid.py --Re [your_Re]
   ```
   Verify: $\Delta x^+ < 11$, $\Delta y^+ < 6$, $\Delta z^+_{\text{min}} < 0.5$

2. **Use stretched grid** (gamma > 1) to efficiently resolve near-wall region


### Time Integration

1. **Use IMEX scheme** for stability and efficiency

2. **Enable adaptive timestepping** to maintain target CFL

3. **Set conservative CFL** (0.2-0.3) for safety

4. **Allow settling time** before collecting statistics (t_stats ≈ 50-100)

### Statistics Collection

1. **Start collecting after turbulence is established:**
   - Monitor u_τ: should oscillate around target value
   - Check that mean flow profile matches log-law

2. **Collect enough samples:**
   - Need ~500-1000 samples for good statistics
   - Run for 500-1000 eddy turnover times after t_stats

3. **Use restart capability** for long runs:
   - Save checkpoint every ~50-100 time units
   - Allows resuming after interruptions

### Output and Storage

1. **Balance checkpoint frequency:**
   - More frequent → safer but more storage
   - Typical: n_save = 1000-5000 steps

2. **Monitor disk space:**
   - Each checkpoint: ~(nx × ny × nz × 32 bytes) × 4 fields
   - Example: 192³ grid ≈ 230 MB per checkpoint

3. **Archive old checkpoints** if running multiple restarts

### Performance

1. **Use GPU** for significant speedup (10-50×)

2. **Increase grid size** to improve GPU utilization

3. **Profile your runs:**
   - Note time per step for different configurations
   - GPU speedup increases with grid size

4. **Optimize n_out vs n_save:**
   - Frequent diagnostics (n_out) have minimal cost
   - File I/O (n_save) can be bottleneck → increase if needed

---

## Troubleshooting

### Simulation Diverges

**Symptoms**: CFL explodes, NaN values, negative dt warning

**Solutions**:
1. Reduce dt or CFL_target
2. Use IMEX scheme instead of AB2
3. Check initial conditions (reduce perturbation_intensity)
4. Increase grid resolution near walls (larger gamma)

### u_τ Not Reaching Target

**Symptoms**: $u_\tau$ converges to different value than $Re_\tau \times \nu / \delta$

**Solutions**:
1. Check Re calculation: should be consistent with Re_tau
2. Increase domain size (longer wavelengths need larger Lx)
3. Verify grid resolution is adequate
4. Allow more time for flow to settle

### Statistics Look Wrong

**Symptoms**: Noisy profiles, negative variances, incorrect scaling

**Solutions**:
1. Collect more samples (increase run time)
2. Start collecting later (increase t_stats)
3. Verify flow is fully turbulent before collecting
4. Check that domain is large enough

### Slow Performance

**Symptoms**: Simulation is slower than expected

**Solutions**:
1. Ensure GPU is being used (`device: "cuda"`)
2. Check CUDA installation and PyTorch compatibility
3. Increase grid size if GPU utilization is low
4. Reduce checkpoint frequency (increase n_save)

---

For more details on the numerical methods and implementation, see:
- [Numerical Methods Documentation](NUMERICAL_METHODS.md)
- [Implementation Details](IMPLEMENTATION.md)
