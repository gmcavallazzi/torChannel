# TorChannel Implementation Details

This document describes the Python/PyTorch implementation of TorChannel, including code architecture, data structures, and performance considerations.

---

## Table of Contents

1. [Code Architecture](#code-architecture)
2. [Module Descriptions](#module-descriptions)
3. [PyTorch Implementation](#pytorch-implementation)
4. [Staggered Grid Indexing](#staggered-grid-indexing)
5. [Data Structures](#data-structures)
6. [File Formats](#file-formats)
7. [Dependencies](#dependencies)
8. [Performance Considerations](#performance-considerations)
9. [Development Guidelines](#development-guidelines)

---

## Code Architecture

### Overview

TorChannel follows a modular architecture with clear separation of concerns:

```
main.py
   ↓
solver.py (ChannelFlow class)
   ├→ utils.py (grid, I/O, diagnostics)
   ├→ initflow.py (initialization)
   ├→ operators.py (spatial operators)
   ├→ projection_fft.py (FFT Poisson solver)
   ├→ projection.py (direct solver, fallback)
   └→ turbstats.py (turbulence statistics)

Post-processing:
   ├→ post_process.py (field visualization)
   ├→ plot_statistics.py (statistics plots)
   └→ generate_grid.py (grid analysis)
```

### Design Philosophy

1. **Modular**: Each module has a specific responsibility
2. **Vectorized**: All operations use PyTorch tensor operations
3. **Device-agnostic**: Code works on both CPU and GPU
4. **Readable**: Clear variable names and documentation
5. **Efficient**: Optimized kernels for performance-critical sections

---

## Module Descriptions

### main.py

**Purpose**: Entry point for simulations

**Key Functions**:
- Parse command-line arguments
- Initialize solver
- Run simulation

**Usage**:
```bash
torchannel-run <config.yaml>        # or: python main.py <config.yaml>
```

---

### solver.py

**Purpose**: Main simulation class and time-stepping loop

**Key Class**: `ChannelFlow`

**Responsibilities**:
- Load configuration
- Initialize grid, velocity fields, pressure
- Main time-stepping loop
- Adaptive timestepping
- Output and checkpointing
- Statistics collection

**Key Methods**:
- `__init__(config_file)`: Initialize simulation
- `run_simulation()`: Main time loop
- `step_AB2()`: Adams-Bashforth 2 time step
- `step_IMEX()`: IMEX time step
- `step_FE()`: Forward Euler time step

**Time Loop Structure**:
```python
for step in range(n_steps):
    # 1. Compute RHS (advection + diffusion)
    # 2. Time integration (AB2 or IMEX)
    # 3. Apply forcing
    # 4. Pressure projection (solve Poisson equation)
    # 5. Velocity correction (subtract pressure gradient)
    # 6. Adaptive timestep
    # 7. Diagnostics and output
    # 8. Statistics collection
```

---

### operators.py

**Purpose**: Spatial discretization operators

**Key Functions**:

**Diffusion**:
- `diffusion_u(u, ...)`: $\nabla^2 u$ on staggered grid
- `diffusion_v(v, ...)`: $\nabla^2 v$ on staggered grid
- `diffusion_w(w, ...)`: $\nabla^2 w$ on staggered grid

**Advection**:
- `advection_u(u, v, w, ...)`: $(\mathbf{u} \cdot \nabla)u$
- `advection_v(u, v, w, ...)`: $(\mathbf{u} \cdot \nabla)v$
- `advection_w(u, v, w, ...)`: $(\mathbf{u} \cdot \nabla)w$

**Implicit Diffusion (z-direction)**:
- `solve_implicit_diffusion_u()`: Crank-Nicolson z-diffusion for u
- `solve_implicit_diffusion_v()`: Crank-Nicolson z-diffusion for v
- `solve_implicit_diffusion_w()`: Crank-Nicolson z-diffusion for w

**Horizontal Diffusion (xy-directions)**:
- `diffusion_xy_u()`: $\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}$
- `diffusion_xy_v()`: $\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}$
- `diffusion_xy_w()`: $\frac{\partial^2 w}{\partial x^2} + \frac{\partial^2 w}{\partial y^2}$

**Fused Kernels** (optimized for GPU):
- `compute_momentum_rhs_fused_imex()`: Combined advection + xy-diffusion
- `compute_cfl_fused()`: Compute CFL number in single kernel

**Utilities**:
- `solve_tridiagonal_batch()`: Batched tridiagonal solver (Thomas algorithm)

**Implementation Notes**:
- All functions handle staggered grid indexing
- Ghost cells used for boundary conditions
- Finite differences on non-uniform z-grid
- Conservative discretization for advection

---

### projection_fft.py

**Purpose**: FFT-based Poisson solver (recommended)

**Key Functions**:

- `initialize_fft_solver(nx, ny, nz, dx, dy, dz_c, dz_f)`:
  - Precompute wavenumbers (kx, ky)
  - Build tridiagonal coefficients for all modes
  - Return dictionary with precomputed data

- `solve_poisson_fft(div, fft_data)`:
  - FFT divergence field in x, y
  - Solve tridiagonal systems in z for each mode
  - Inverse FFT to get pressure
  - Return pressure with ghost cells

**Algorithm**:
1. FFT: $\hat{\text{div}} = \text{FFT}_{xy}[\text{div}]$
2. For each $(k_x, k_y)$:
   - Solve: $\left[\frac{\partial^2}{\partial z^2} - k_x^2 - k_y^2\right] \hat{\varphi} = \hat{\text{div}}$
3. Inverse FFT: $\varphi = \text{IFFT}_{xy}[\hat{\varphi}]$

**Features**:
- Modified wavenumbers for 2nd-order accuracy
- Batched tridiagonal solves (all modes simultaneously)
- Pre-allocated workspace to avoid allocations
- Neumann BC at walls (∂φ/∂z = 0)

---

### projection.py

**Purpose**: Direct Poisson solver (fallback for testing)

**Key Functions**:

- `build_poisson_matrix(nx, ny, nz, dx, dy, dz_c, dz_f)`:
  - Build sparse matrix for $\nabla^2 \varphi = \text{RHS}$
  - 7-point stencil on 3D grid
  - Periodic BC in x, y; Neumann in z

- `solve_poisson(div, matrix, nx, ny, nz)`:
  - Solve Au = b using sparse solver
  - Return pressure field

- `project_velocity(u, v, w, p, dx, dy, dz_c, dt, nx, ny, nz)`:
  - Compute pressure gradient
  - Update velocity: $\mathbf{u} := \mathbf{u} - \Delta t \nabla p$

**Use Cases**:
- Validation against FFT solver
- Small test cases
- Non-standard boundary conditions (future)

**Disadvantages**:
- Slower than FFT solver (O(N²) vs O(N log N))
- Limited to small grids
- Not recommended for production

---

### initflow.py

**Purpose**: Flow field initialization

**Key Functions**:

- `initialize_flow(...)`:
  - Create initial velocity and pressure fields
  - Types: "vortices", "random", "parabolic", "laminar"
  - Apply no-slip BC at walls

- `initialize_flow_from_file(field_file, device, reset_time)`:
  - Load from checkpoint (.npz file)
  - Optionally reset time/step
  - Transfer to specified device

**Initialization Types**:

1. **Vortices** (recommended):
   - Parabolic base + counter-rotating vortex pairs
   - Streamfunction: $\psi = A \sin(k_y y) \sin^2(\pi z / L_z)$
   - $v = -\frac{\partial \psi}{\partial z}$, $w = \frac{\partial \psi}{\partial y}$

2. **Parabolic**:
   - $u(z) = U_{\text{max}} \left[1 - \left(\frac{2z}{L_z} - 1\right)^2\right]$

3. **Random**:
   - Random perturbations

4. **Laminar**:
   - Uniform u = U_bulk

---

### utils.py

**Purpose**: Utilities for grid, I/O, and diagnostics

**Key Functions**:

**Grid Generation**:
- `generate_grid(gamma, nz, Lz, device)`:
  - Hyperbolic tangent stretching
  - Returns z_f, z_c, dz_f, dz_c
  - Creates ghost cells

- `plot_grid(z_f, z_c, results_folder)`:
  - Visualize grid distribution

- `save_grid_csv(...)`:
  - Save grid data to CSV

**I/O**:
- `save_flow_fields(u, v, w, p, ..., filename)`:
  - Save velocity, pressure to .npz
  - Include metadata (step, time, u_tau, forcing)

- `load_flow_fields(filename, device)`:
  - Load from .npz file
  - Transfer to device

**Diagnostics**:
- `compute_u_tau(u, z_c, nu)`:
  - Compute friction velocity from velocity gradient at wall

- `compute_bulk_velocity(u, cell_vol, total_vol)`:
  - Volume-averaged velocity

- `compute_divergence(u, v, w, dx, dy, dz_c, ...)`:
  - Check divergence-free condition

**Plotting**:
- `plot_profile(...)`: Plot 1D profiles

---

### turbstats.py

**Purpose**: Turbulence statistics collection

**Key Class**: `TurbulenceStats`

**Features**:
- On-the-fly accumulation (running sums)
- Memory-efficient (no storage of snapshots)
- Checkpointing for restart
- Computed quantities:
  - Mean velocity U(z)
  - Reynolds stresses: uu, vv, ww, uw
  - 2D energy spectra at z⁺ ≈ 15

**Key Methods**:
- `accumulate_statistics(u, v, w, u_tau)`:
  - Add current snapshot to running sums
  - Compute fluctuations u' = u - U(z)
  - Update spectral energy at z⁺ location

- `finalize_statistics()`:
  - Divide sums by n_samples
  - Compute u_tau from velocity gradient
  - Return dictionary of statistics

- `save_statistics(filepath)`:
  - Save finalized statistics to .npz

- `save_state(filepath)` / `load_state(filepath)`:
  - Checkpoint running sums for restart
  - Preserves n_samples count

**Implementation Details**:
- All operations on GPU (if available)
- Only transfer to CPU for final save
- 2D FFT for spectral analysis
- Symmetric folding of negative wavenumbers

---

### post_process.py

**Purpose**: Flow field visualization

**Key Functions**:
- `plot_slices(...)`: 2D slices in xy, xz, yz planes
- `plot_profiles(...)`: Mean velocity profile
- `plot_timeseries(...)`: Time evolution of u_bulk, u_tau, forcing

**Usage**:
```bash
# Full post-processing
python post_process.py results/fields.npz --config examples/re180_open/config.yaml

# Timeseries only
python post_process.py results/timeseries.npz --timeseries-only --Re 2870
```

---

### plot_statistics.py

**Purpose**: Turbulence statistics visualization

**Key Functions**:
- `plot_velocity_profile()`: U⁺ vs z⁺ (log-law plot)
- `plot_normal_stresses()`: Reynolds stress components
- `plot_shear_vorticity()`: -⟨u'w'⟩ and dU/dz
- `plot_2d_spectra()`: Premultiplied energy spectra
- `plot_total_stress_decomposition()`: Reynolds + viscous stress

**Usage**:
```bash
python plot_statistics.py results/turbulence_stats.npz --config examples/re180_open/config.yaml
```

**Output**: Multiple PNG/PDF files with publication-quality plots

---

### generate_grid.py

**Purpose**: Grid analysis and memory estimation

**Usage**:
```bash
python generate_grid.py --Re 2870
```

**Output**:
- Grid spacing in physical and wall units
- Memory requirements for CPU/GPU
- Grid distribution plot

---

## PyTorch Implementation

### Why PyTorch?

1. **GPU Acceleration**: Native CUDA support
2. **Automatic Differentiation**: Not used currently, but available for future optimization
3. **Tensor Operations**: Efficient vectorized operations
4. **Flexibility**: Easy to prototype and modify
5. **Ecosystem**: Integration with scientific Python stack

### Tensor Types

**Data Type**: `torch.float64` (double precision)
- Set globally: `torch.set_default_dtype(torch.float64)`
- Essential for DNS stability and accuracy

**Device**: `torch.device('cuda')` or `torch.device('cpu')`
- Tensors created on specified device
- All operations stay on device (avoid transfers)

### Memory Management

**Pre-allocation**:
- Workspace arrays allocated once, reused
- Example: FFT pressure workspace

**In-place Operations**:
- Use `+=`, `-=`, `*=` when possible
- Reduces temporary allocations

**GPU Memory**:
- Monitor with `torch.cuda.memory_allocated()`
- Clear cache if needed: `torch.cuda.empty_cache()`

### Vectorization

**Key Principles**:
1. Avoid Python loops over grid points
2. Use tensor slicing for stencils
3. Broadcast operations when possible
4. Fuse operations into single kernels

**Example - Finite Difference**:
```python
# Bad: Python loop
for i in range(1, nx-1):
    dudx[i] = (u[i+1] - u[i-1]) / (2*dx)

# Good: Vectorized
dudx[1:-1] = (u[2:] - u[:-2]) / (2*dx)
```

### GPU Optimization

**Fused Kernels**:
- Combine multiple operations into single kernel
- Example: `compute_momentum_rhs_fused_imex()`
  - Computes advection + xy-diffusion in one pass
  - Reduces memory bandwidth

**Batched Operations**:
- Example: Tridiagonal solves for all FFT modes
  - Shape: (nx × nky, nz)
  - Single batched solve instead of loop

**Avoiding CPU-GPU Transfers**:
- Keep data on GPU as long as possible
- Only transfer for final output/diagnostics
- Use `.item()` sparingly (triggers transfer)

---

## Staggered Grid Indexing

### Array Shapes

Including ghost cells:
- **u**: (nx+1, ny+2, nz+2) - staggered in x
- **v**: (nx+2, ny+1, nz+2) - staggered in y
- **w**: (nx+2, ny+2, nz+1) - staggered in z
- **p**: (nx+2, ny+2, nz+2) - cell-centered

### Interior Points

Excluding ghost cells:
- **u interior**: `u[0:nx+1, 1:ny+1, 1:nz+1]`
- **v interior**: `v[1:nx+1, 0:ny+1, 1:nz+1]`
- **w interior**: `w[1:nx+1, 1:ny+1, 0:nz+1]`
- **p interior**: `p[1:nx+1, 1:ny+1, 1:nz+1]`

### Ghost Cells

**Purpose**:
- Implement boundary conditions
- Avoid special cases in stencils

**Periodic directions (x, y)**:
- Wrap-around: `u[0] = u[nx]`, `u[-1] = u[1]`
- Implemented automatically via periodic BC

**No-slip walls (z = 0, Lz)**:
- Bottom: `u[:,:,0]` (ghost cell below wall)
- Top: `u[:,:,-1]` (ghost cell above wall)
- Values set to enforce u = 0 at walls

### Interpolation

**To cell center from u-face**:
```python
u_center = 0.5 * (u[:-1, :, :] + u[1:, :, :])
```

**To cell center from z-face (w)**:
```python
w_center = 0.5 * (w[:, :, :-1] + w[:, :, 1:])
```

### Gradient Computation

**Pressure gradient at u-location**:
```python
dpdx = (p[1:, :, :] - p[:-1, :, :]) / dx
```

**Velocity gradient at cell center**:
```python
dudx = (u[1:, :, :] - u[:-1, :, :]) / dx
```

---

## Data Structures

### Configuration (YAML)

Loaded as nested dictionary:
```python
config = {
    'grid': {'nx': 192, 'ny': 192, 'nz': 128},
    'flow': {'Re': 2870.0, ...},
    ...
}
```

### Flow Fields (NPZ)

Saved as NumPy .npz (compressed):
```python
np.savez_compressed(filename,
    u=u_array,
    v=v_array,
    w=w_array,
    p=p_array,
    z_c=z_c_array,
    z_f=z_f_array,
    Lx=Lx,
    Ly=Ly,
    step=step,
    time=time,
    u_tau=u_tau,
    forcing=forcing
)
```

**Loading**:
```python
data = np.load(filename)
u = torch.tensor(data['u'], device=device)
```

### Statistics (NPZ)

**Finalized statistics**:
```python
{
    'n_samples': int,
    'z_c': array(nz),
    'U_mean': array(nz),
    'uu_mean': array(nz),
    'vv_mean': array(nz),
    'ww_mean': array(nz),
    'uw_mean': array(nz),
    'kx': array(nx//2),
    'ky': array(ny//2),
    'E_uu_2d': array(nx//2, ny//2),
    'E_vv_2d': array(nx//2, ny//2),
    'E_ww_2d': array(nx//2, ny//2),
    'E_uw_2d': array(nx//2, ny//2),
    'nu': float,
    'Re_tau_target': float,
    'u_tau': float
}
```

**State checkpoint**:
```python
{
    'n_samples': int,
    'U_sum': array(nz),
    'uu_sum': array(nz),
    ...,  # Running sums (not averaged)
    'E_uu_2d_sum': array(nx//2, ny//2),
    ...
}
```

### Timeseries (NPZ)

```python
{
    'time': array(n_samples),
    'step': array(n_samples),
    'u_bulk': array(n_samples),
    'u_tau': array(n_samples),
    'forcing': array(n_samples)
}
```

**Automatic appending** on restart:
- Load existing file
- Concatenate new data
- Save back to same file

---

## File Formats

### NPZ (NumPy Compressed)

**Advantages**:
- Compressed (saves disk space)
- Easy to load in Python/NumPy
- Supports multiple arrays in one file
- Portable across platforms

**Disadvantages**:
- Not human-readable
- Python-specific format

**Usage**:
```python
# Save
np.savez_compressed(filename, array1=a1, array2=a2, ...)

# Load
data = np.load(filename)
a1 = data['array1']
```

### YAML (Configuration)

**Advantages**:
- Human-readable and editable
- Supports nested structures
- Comments allowed
- Standard format

**Usage**:
```python
import yaml

# Load
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
```

### CSV (Grid Data)

**For grid coordinates**:
```
k,z_f,z_c,dz_f,dz_c,z_plus
0,0.0,0.005,0.01,0.01,0.0
1,0.01,0.015,0.01,0.01,0.1
...
```

**Advantages**:
- Human-readable
- Easy to import in Excel/MATLAB
- Portable

---

## Dependencies

### Required Packages

```
Python >= 3.8
PyTorch >= 1.10
NumPy >= 1.20
Matplotlib >= 3.3
PyYAML >= 5.4
```

### Installation

**CPU-only**:
```bash
pip install torch numpy matplotlib pyyaml
```

**GPU (CUDA 11.8)**:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib pyyaml
```

**Optional (for publication-quality plots)**:
```bash
# LaTeX support in Matplotlib
# Requires LaTeX installation on system
```

### Compatibility

**PyTorch versions**:
- Tested on PyTorch 1.10 - 2.1
- Should work on newer versions

**CUDA versions**:
- CUDA 11.x and 12.x supported
- Match PyTorch CUDA version with system CUDA

**Operating Systems**:
- Linux (primary development/testing)
- macOS (CPU-only)
- Windows (should work, less tested)

---

## Performance Considerations

### GPU Acceleration

**Speedup factors**:
- Grid size dependent
- Measured: 0.96 s/step at 576x432x260 (64.7M cells) in float64 on an
  NVIDIA GB10, with the canopy IBM active (~1.5e-8 s per cell per step).
  The previous "10-50x faster than CPU" claim here was never measured.
- Larger grids → better GPU utilization

**GPU Memory Requirements**:
- Re_τ = 180 (192³): ~2-4 GB
- Re_τ = 395 (256³): ~4-8 GB
- Re_τ = 550 (384³): ~8-16 GB

**Memory Formula** (approximate):
$$\text{Memory} \approx (n_x \times n_y \times n_z) \times 32 \text{ bytes/point} \times 10 \text{ fields}$$

### Optimization Techniques

**1. Fused Kernels**
- Combine operations to reduce memory bandwidth
- Example: advection + diffusion in one pass

**2. Pre-allocation**
- Allocate workspace arrays once
- Reuse buffers to avoid repeated allocations

**3. Batched Operations**
- Tridiagonal solves for all FFT modes simultaneously
- Better GPU utilization

**4. Minimize CPU-GPU Transfers**
- Keep computation on GPU
- Transfer only for output/diagnostics

**5. Vectorization**
- No Python loops over grid points
- Use tensor slicing and broadcasting

### Profiling

**PyTorch Profiler**:
```python
with torch.profiler.profile() as prof:
    # Code to profile
    solver.step_IMEX()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

**CUDA Events** (timing):
```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
# Code to time
end.record()
torch.cuda.synchronize()
print(f"Time: {start.elapsed_time(end)} ms")
```

---

## Development Guidelines

### Code Style

**Naming Conventions**:
- Variables: `snake_case`
- Functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_CASE`

**Documentation**:
- Docstrings for all functions/classes
- Type hints where helpful
- Comments for non-obvious logic

**Example**:
```python
def compute_u_tau(u: torch.Tensor, z_c: torch.Tensor, nu: float) -> float:
    """
    Compute friction velocity from wall velocity gradient.

    Args:
        u: Streamwise velocity field (nx+1, ny+2, nz+2)
        z_c: Cell center positions (nz+2)
        nu: Kinematic viscosity

    Returns:
        u_tau: Friction velocity
    """
    # Implementation
    ...
```

### Testing

**Unit Tests**:
- Test individual functions
- Use small grids for speed
- Check against analytical solutions

**Validation**:
- Compare with published DNS data
- Check conservation properties
- Verify boundary conditions

**Regression Tests**:
- Ensure changes don't break existing functionality
- Save reference outputs

### Version Control

**Branches**:
- `main`: Stable version
- `dev`: Development
- Feature branches for new capabilities

**Commits**:
- Clear, descriptive messages
- One logical change per commit

**Tags**:
- Version releases: `v1.0.0`, `v1.1.0`, etc.

### Documentation

**Keep Updated**:
- README for major changes
- Configuration guide for new parameters
- Implementation details for new modules

**Examples**:
- Provide working examples
- Document typical use cases
- Include troubleshooting tips

---

## Future Enhancements

Potential areas for development:

1. **Higher-order methods**: 4th-order finite differences
2. **Additional turbulence models**: LES capabilities
3. **Spectral methods**: Chebyshev in z-direction
4. **Multi-GPU**: Domain decomposition
5. **Post-processing tools**: Vortex identification, structure detection
6. **Boundary layers**: Additional geometries
7. **Compressible flow**: Extension to compressible Navier-Stokes

---

For questions about implementation or contributions, please open an issue on the GitHub repository.

For usage and configuration details, see:
- [Configuration Guide](CONFIG_GUIDE.md)
- [Numerical Methods](NUMERICAL_METHODS.md)
- [README](../README.md)
