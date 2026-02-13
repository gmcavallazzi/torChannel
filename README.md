# TorChannel

**GPU-accelerated Direct Numerical Simulation (DNS) for turbulent channel flow**

TorChannel is a high-performance Python/PyTorch implementation of DNS for incompressible turbulent channel flow. It features GPU acceleration, advanced time integration schemes, comprehensive turbulence statistics collection, and robust restart capabilities.

---

## Features

- **GPU Acceleration**: Fully vectorized PyTorch implementation with CUDA support
- **Advanced Time Integration**: IMEX scheme (implicit wall-normal diffusion, explicit advection) with adaptive timestepping
- **FFT-based Poisson Solver**: Fast pressure projection using FFT in periodic directions
- **Non-uniform Stretched Grids**: Hyperbolic tangent stretching for efficient near-wall resolution
- **Turbulence Statistics**: On-the-fly computation of Reynolds stresses, mean profiles, and 2D energy spectra
- **Restart Capability**: Save/load flow fields and accumulated statistics without losing progress
- **Bulk Velocity Forcing**: Maintains constant flow rate (Re_bulk) by adjusting pressure gradient
- **Flexible Configuration**: YAML-based configuration with extensive options

---

## Quick Start

### Installation

**Requirements:**
- Python >= 3.8
- PyTorch >= 1.10 (with optional CUDA support)
- NumPy >= 1.20
- Matplotlib >= 3.3
- PyYAML >= 5.4

**CPU-only setup:**
```bash
pip install torch numpy matplotlib pyyaml
```

**GPU setup (CUDA):**
```bash
# Install PyTorch with CUDA support (adjust CUDA version as needed)
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib pyyaml
```

### Running a Simulation

1. **Configure the simulation** by editing `config.yaml` (see [Configuration Guide](docs/CONFIG_GUIDE.md))

2. **Run the simulation:**
   ```bash
   python main.py config.yaml
   ```

3. **Visualize results:**
   ```bash
   # Plot flow field slices and mean velocity profile
   python post_process.py results/fields.npz --config config.yaml

   # Plot timeseries data only
   python post_process.py results/timeseries.npz --timeseries-only --Re 2870

   # Plot turbulence statistics
   python plot_statistics.py results/turbulence_stats.npz --config config.yaml
   ```

### Example Configuration

A typical configuration for Re_τ = 180:

```yaml
grid:
  nx: 192    # Streamwise points
  ny: 192    # Spanwise points
  nz: 128    # Wall-normal points

domain:
  Lx: 12.567  # 4π (streamwise length)
  Ly: 4.189   # 4π/3 (spanwise length)
  Lz: 2.0     # Channel height (2δ)

flow:
  Re: 2870.0      # Bulk Reynolds number
  Re_tau: 180.0   # Target friction Reynolds number
  U_bulk: 1.0     # Target bulk velocity
  gamma: 2.6      # Grid stretching parameter

time:
  dt: 0.005
  CFL_target: 0.25
  scheme: "IMEX"  # IMEX, AB2, or FE

output:
  results_folder: results
  n_out: 200      # Print diagnostics every 200 steps
  n_save: 4000    # Save fields every 4000 steps

statistics:
  enabled: true
  n_stats: 200     # Collect statistics every 200 steps
  t_stats: 50.0    # Start collecting after t=50
  z_plus_target: 15.0  # Height for 2D spectra
```

---

## Directory Structure

```
TorChannel/
├── README.md                   # This file
├── LICENSE                     # MIT License
├── config.yaml                 # Main configuration file
├── main.py                     # Entry point
├── solver.py                   # Main simulation class
├── operators.py                # Spatial operators
├── projection_fft.py           # FFT Poisson solver
├── projection.py               # Direct Poisson solver (fallback)
├── initflow.py                 # Flow initialization
├── utils.py                    # Grid generation, I/O, utilities
├── statistics.py               # Turbulence statistics
├── plot_statistics.py          # Statistics visualization
├── post_process.py             # Flow field visualization
├── generate_grid.py            # Grid generation utility
├── docs/
│   ├── CONFIG_GUIDE.md         # Configuration reference
│   ├── NUMERICAL_METHODS.md    # Equations and discretization
│   └── IMPLEMENTATION.md       # Code architecture details
└── results/                    # Output directory (created automatically)
    ├── fields.npz              # Latest flow field checkpoint
    ├── fields_final.npz        # Final flow field
    ├── timeseries.npz          # Time history data
    ├── turbulence_stats.npz    # Averaged statistics
    └── turbulence_stats_state.npz  # Statistics checkpoint
```

---

## Output Files

### Flow Fields
- **fields_init.npz**: Initial flow field
- **fields.npz**: Latest checkpoint (updated every `n_save` steps)
- **fields_final.npz**: Final flow field at end of simulation

### Time Series
- **timeseries.npz**: Time history of bulk velocity, friction velocity (u_τ), and forcing
- Automatically appended on restart

### Turbulence Statistics
- **turbulence_stats.npz**: Final averaged statistics
  - Mean velocity profile U(z)
  - Reynolds stresses: u'u'(z), v'v'(z), w'w'(z), u'w'(z)
  - 2D premultiplied energy spectra at z⁺ ≈ 15
- **turbulence_stats_state.npz**: Statistics checkpoint (running sums + sample count)
  - Allows restarting statistics accumulation without losing progress

### Visualization
- **grid_plot.png**: Non-uniform grid distribution
- **post_slices_*.png**: Flow field slices (xy, xz, yz planes)
- **post_profile.png**: Mean streamwise velocity profile
- **post_timeseries.png**: Time evolution of u_bulk, u_τ, forcing
- **turbulence_stats_plots_*.png**: Statistics plots (velocity, stresses, spectra)

---

## Restarting Simulations

TorChannel supports seamless restart from checkpoints:

**1. Restart flow field:**
```yaml
initialization:
  field_file: "results/fields.npz"  # Path to checkpoint
  reset_time: false                  # Continue from saved time
```

**2. Restart statistics accumulation:**
```yaml
statistics:
  restart_state_file: "results/turbulence_stats_state.npz"
```

On restart:
- Results folder is **preserved** (no cleaning)
- Timeseries data is **automatically appended**
- Statistics accumulation **continues** from checkpoint

---

## Documentation

- **[Configuration Guide](docs/CONFIG_GUIDE.md)**: Detailed explanation of all configuration parameters
- **[Numerical Methods](docs/NUMERICAL_METHODS.md)**: Governing equations, discretization schemes, and solution methods
- **[Implementation Details](docs/IMPLEMENTATION.md)**: Code architecture, PyTorch implementation, and performance considerations

---

## Validation

TorChannel has been validated against established DNS databases:

- **Re_τ = 180**: Results consistent with Moser et al. (1999) channel flow database
- Mean velocity profiles match log-law and viscous sublayer predictions
- Reynolds stress profiles show correct near-wall scaling
- Friction velocity u_τ converges to target value within < 1% error

---

## References

**Key DNS Literature:**

1. Kim, J., Moin, P., & Moser, R. (1987). "Turbulence statistics in fully developed channel flow at low Reynolds number." *Journal of Fluid Mechanics*, 177, 133-166.

2. Moser, R. D., Kim, J., & Mansour, N. N. (1999). "Direct numerical simulation of turbulent channel flow up to Re_τ = 590." *Physics of Fluids*, 11(4), 943-945.

3. Pope, S. B. (2000). *Turbulent Flows*. Cambridge University Press.

**DNS Database:**
- [Johns Hopkins Turbulence Database](http://turbulence.pha.jhu.edu/)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Author

Giorgio Cavallazzi

For questions, issues, or contributions, please open an issue on the GitHub repository.

---

## Acknowledgments

This code implements standard DNS methods for turbulent channel flow based on the extensive literature in computational fluid dynamics. The implementation emphasizes:

- **Performance**: GPU acceleration with PyTorch for efficient large-scale simulations
- **Accuracy**: Second-order spatial discretization with adaptive timestepping
- **Usability**: YAML configuration, restart capabilities, and comprehensive output

---

## Citation

If you use TorChannel in your research, please cite:

```
@software{torchannel2024,
  author = {Cavallazzi, Giorgio},
  title = {TorChannel: GPU-accelerated DNS for turbulent channel flow},
  year = {2024},
  url = {https://github.com/[your-username]/TorChannel}
}
```
