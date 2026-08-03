# torChannel

**GPU-accelerated DNS of turbulent channel flow, written in PyTorch.**

torChannel solves the incompressible Navier–Stokes equations on a staggered grid
with a second-order finite-volume discretisation, an IMEX time integrator, and an
FFT-based pressure projection. It runs on a single GPU and is written entirely in
Python — every field is a `torch.Tensor` and every operator is a tensor op, so the
solver can be read, modified, and driven from a Python loop rather than only run as
a batch job.

It also includes an RKPM immersed-boundary method for rigid filamentous canopies,
following Monti et al. (2022).

**Because the solver is a PyTorch program, it can be driven as a control
environment** — stepped from a Python loop, with sensors read and actuation
written in-process, no co-simulation layer:

```python
from torchannel.control import ChannelFlowEnv, OppositionControl

env = ChannelFlowEnv("examples/re180_open/config.yaml", action_interval=0.05)
obs = env.reset()                       # obs = wall shear stress field
policy = OppositionControl(env, detection_z_plus=15.0)

for _ in range(400):
    obs, reward, done, info = env.step(policy(obs))   # action = wall blowing/suction
    print(info["drag_reduction"])
```

See [`examples/opposition_control/`](examples/opposition_control/), which
reproduces the ~20–25 % drag reduction of Choi, Moin & Kim (1994).
`env.to_gym()` adapts it to `gymnasium` for RL libraries.

---

## Install

```bash
git clone https://github.com/gmcavallazzi/torChannel.git
cd torChannel
pip install -e ".[plot]"        # add ,test for pytest
```

Requires Python ≥ 3.9 and PyTorch ≥ 1.10. The core solver needs only `torch`,
`numpy` and `pyyaml`; `[plot]` adds `matplotlib`, `scipy` and `scikit-image` for
post-processing.

## Run

```bash
torchannel-run examples/re180_open/config.yaml
```

or equivalently `python main.py examples/re180_open/config.yaml`. The bundled
example is a Re_τ = 180 open channel on 192×192×128 — see
[`examples/re180_open/README.md`](examples/re180_open/README.md), which also
documents how to build its initial field.

Post-process:

```bash
# Statistics, overlaid on published DNS
python plot_statistics.py results_re180_open/turbulence_stats.npz \
    --config examples/re180_open/config.yaml --reference mkm180

# From a mid-run checkpoint instead of the final file
python plot_statistics.py results_re180_open/turbulence_stats_state.npz \
    --checkpoint --config examples/re180_open/config.yaml

# Fields, timeseries, snapshots
python post_process.py results_re180_open/fields.npz --config examples/re180_open/config.yaml
python scripts/plot_timeseries.py slurm-re180-open-319.out
python scripts/plot_snapshot.py results_re180_open/fields.npz
```

Figures use `text.usetex`; on a cluster you may need `module load texlive` first.

---

## What it does and does not do

Being explicit, so you can tell in one minute whether it fits your problem.

**Supported**

| | |
|---|---|
| Equations | Incompressible Navier–Stokes, constant density, constant viscosity |
| Geometry | Plane channel. Periodic in x and y; walls in z |
| Boundary conditions | Bottom wall always no-slip. Top wall `dirichlet` (closed channel) or `neumann` (free-slip/symmetry — open channel) |
| Time integration | `IMEX` (AB2 on advection + xy-diffusion, implicit z-diffusion) for production; `FE` for testing |
| Pressure | FFT in x,y + Thomas in z, with modified wavenumbers; a dense direct solver for small cases |
| Grids | Uniform in x,y. In z: `symmetric`, `bottom`, `hybrid`, or `double` (tanh clustering at a bed and at canopy tips) |
| Forcing | Constant flow rate, via a feedback controller on the bulk velocity |
| Initialisation | `parabolic`, `uniform`, `vortices` (+ seeded random perturbations), restart from a checkpoint, or `interpolate` from a field on a *different* grid |
| Immersed boundary | Rigid filamentous canopies (RKPM direct forcing) — see [`docs/CANOPY.md`](docs/CANOPY.md) |
| Control | Steppable environment with wall blowing/suction; optional `gymnasium` adapter — see [`torchannel/control.py`](torchannel/control.py) |
| Precision | `compute.precision: float64` (default, the reference path), `mixed`, or `float32` |
| Statistics | Mean profile, Reynolds stresses ⟨u'u'⟩ ⟨v'v'⟩ ⟨w'w'⟩ ⟨u'w'⟩, third moments, canopy drag profile, 2D premultiplied spectra at arbitrary heights. Accumulated as running sums, so restarts lose nothing |

**Not supported.** Backpropagation *through* the solver (in-place ops, CUDA-graph
capture and preallocated buffers all fight autograd — the control environment is
for gradient-free methods). No scalar transport, temperature or buoyancy. No LES
or wall model — this is DNS, and the resolution burden is yours. No inflow/outflow or
streamwise development. No constant-pressure-gradient forcing (flow rate only).
No flexible or moving immersed bodies. `RK3` is a stub that raises
`NotImplementedError`. **Single GPU only** — there is no MPI or domain
decomposition, which is the hard ceiling on problem size.

---

## Validation

The bundled Re_τ = 180 open-channel case is the reference configuration.
`plot_statistics.py --reference` overlays published DNS directly:

| dataset | flag | Re_τ |
|---|---|---|
| Moser, Kim & Mansour (1999) | `--reference mkm180` | 178.1 |
| Moser, Kim & Mansour (1999) | `--reference mkm590` | 587.2 |
| Lee & Moser (2015) | `--reference lm550` | 543.5 |

These ship as small CSVs under `torchannel/data/reference/`, regenerable with
`python scripts/fetch_reference_data.py`. That script exists rather than a raw
download because the published data use **y** as the wall-normal direction while
torChannel uses **z**, so the Reynolds stresses must be remapped
(`R_vv → ww`, `R_ww → vv`, `R_uv → uw`), not merely renamed.

Two caveats that belong on any figure you produce:

- The reference data are **closed**-channel profiles. Overlaid on an open channel
  they agree near the wall; the difference toward the centreline is physical, not
  an error — a free-slip top suppresses motions that cross a closed channel's
  centreline.
- The bundled example is seeded from a CaNS field in a different box, so
  comparison against that baseline is **statistical, not like-for-like**.

Beyond that, `pytest -m "not gpu"` runs an assertion-based suite covering grid
generation, the tridiagonal solver, divergence-free projection under both
pressure BCs, and the reference-data axis convention. CI runs it on every push.

---

## Performance notes

Measured on an NVIDIA GB10, `float64` throughout: **0.96 s/step at 576×432×260
(64.7 M cells)** with the canopy IBM active — about 1.5×10⁻⁸ s per cell per step.

`compute.precision` selects the working precision. `float64` is the default and
the reference path; `mixed` keeps the pressure solve in float64 (its z-operator
is ill-conditioned at the largest scales) while running the fields and momentum
kernels in float32; `float32` runs everything reduced. Statistics accumulators,
grid metrics and the bulk-velocity controller stay float64 in every mode.

Reduced precision halves memory, so roughly 1.26× the linear resolution fits on
the same card. Expect a ~2× speedup, not 64× — these kernels are
memory-bandwidth bound, so halving the bytes is what you actually collect.

### GPU notes (GB10 / sm_121)

The GB10 cannot NVRTC-compile the legacy TorchScript fuser, so runs there need
`PYTORCH_JIT=0` (the `@torch.jit.script` decorators become passthroughs). Two
opt-in speed layers replace it:

```bash
PYTORCH_JIT=0 TORCHANNEL_COMPILE=1 TORCHANNEL_POISSON_CUDAGRAPH=1 \
    torchannel-run examples/re180_open/config.yaml
```

- `TORCHANNEL_COMPILE=1` — torch.compile (Inductor/Triton) on the hot kernels
- `TORCHANNEL_POISSON_CUDAGRAPH=1` — CUDA-graph capture of the FFT-Poisson solve

A CUDA-capability warning at startup on this card is expected and harmless.

---

## Configuration

YAML. See [`docs/CONFIG_GUIDE.md`](docs/CONFIG_GUIDE.md) for the full reference and
`examples/` for working files. Three things that reliably catch people out:

- **Statistics are enabled by `statistics.n_stats > 0`.** Some older configs carry
  an `enabled:` key; the code ignores it.
- **On an open channel, use `statistics.spectra_z`** (a list of physical heights),
  not `z_plus_target`. The latter is a legacy two-wall path that assumes
  δ = L_z/2 and pairs a near-wall plane with one at the opposite wall.
- **`initialization.type: interpolate` is not a restart.** It regrids a field from
  a different grid or domain, resets time to zero, and rescales to `U_bulk`. A
  restart is `field_file` with `reset_time: false`.

### Restarting

```yaml
initialization:
  field_file: "results_re180_open/fields.npz"
  reset_time: false
statistics:
  restart_state_file: "results_re180_open/turbulence_stats_state.npz"
```

The results folder is preserved, `timeseries.npz` is appended, and statistics
continue from the accumulated running sums.

---

## Repository layout

```
torchannel/          the package
  solver.py            ChannelFlow: time loop, restart, adaptive dt, forcing
  operators.py         advection, diffusion, implicit z-diffusion, fused kernels
  projection_fft.py    FFT + Thomas Poisson solver
  turbstats.py         running-sum turbulence statistics
  canopy.py            RKPM rigid-canopy IBM
  utils.py             grids, divergence, field I/O
  data/reference/      digitised reference DNS profiles
examples/            runnable cases
tests/               test_regression.py (pytest) + ~60 standalone diagnostics
scripts/             converters and plotting utilities
docs/                configuration, numerical methods, canopy
```

Top-level `solver.py`, `utils.py` etc. are back-compat shims that re-export the
package modules.

---

## Citing

See [`CITATION.cff`](CITATION.cff). If you use the canopy IBM, please also cite
Monti, Nicholas, Omidyeganeh, Pinelli & Rosti (2022), *On the solidity parameter
in canopy flows*, JFM.

## License

MIT — see [`LICENSE`](LICENSE).
