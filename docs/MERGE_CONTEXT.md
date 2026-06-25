# MERGE project context (portable)

This file travels with the repo so the project context is not lost when the code is
moved to another machine (e.g. a GPU/SLURM cluster). It mirrors the working notes
that otherwise live only in a local Claude Code memory on the original workstation.

## Goal

Test the **MERGE** research proposal (*"Fractal Boundary Conditioning for Passive
Mixing Enhancement in Microfluidic Channels"*, de Oliveira & Scheid; `draft_BS.pdf`
on the original workstation). Central claim:

    L_mix(N) / L_mix(0)  ~  r^{-D_f N}        (proposal Eq. 4)

i.e. imposing a Koch-fractal fold (generation N, contraction ratio r, dimension
D_f = log m / log r) on the fluid-fluid interface at a microfluidic junction should
shorten the mixing length. Open question / main risk: do the fine fractal scales
survive molecular diffusion long enough to help, and is there an optimal N?

This `torChannel` DNS code (incompressible, staggered-FD, PyTorch) was chosen as the
Navier-Stokes base to probe the **Re-dependent advective mechanism** (proposal Eq. 5)
that a pure-diffusion model cannot capture.

## What was built on branch `feature/passive-scalar`

1. **`scalar.py`** — passive scalar transport (concentration c in [0,1]) at cell
   centres: conservative flux-form advection + IMEX diffusion with D = nu/Sc, periodic
   in x,y, configurable wall BC in z (`neumann` no-flux default for mixing/decay
   studies; or `dirichlet`). Wired into the time loop via `ChannelFlow.advance_scalar`.
2. **Koch fractal IC** — `scalar.init_type: 'koch'` builds a generation-N Koch
   interface (area-balanced zigzag generator) in the (z,y) cross-section, homogeneous
   in streamwise x. N=0 is the flat baseline.
3. **`scripts/run_mixing.py`** — temporal-mixing driver: evolves the scalar, records
   the intensity of segregation M(t)=std(c)/std_max, extracts t_mix (first M<0.05) and
   L_mix = U_bulk * t_mix, and plots M(t) per N.
4. **`tests/test_scalar.py`** — verification (all pass): pure-diffusion vs analytic
   erf (D_eff/D - 1 = -0.004%, i.e. ZERO numerical diffusion), mean conservation to
   machine precision, pure advection over one period (variance preserved to 0.04%).
5. Configs: `configs/scalar_mix_test.yaml`, `configs/koch_mix_demo.yaml`.

## Result so far

Laminar plain-channel demo (Re=20, Sc=1): **L_mix = 14.5 / 14.7 / 14.7 for N=0/1/2 —
identical.** A 3D DNS confirmation that a fractal interface gives NO mixing benefit in
unidirectional laminar flow: there is no transverse velocity to fold it, so only
diffusion acts (lowest-mode decay M ~ exp(-D k^2 t)). This is the control; a real
benefit needs an advective folding mechanism.

## Key architectural constraint (important)

The pressure Poisson solve (`projection_fft.py`) is **FFT-based and tied to PERIODICITY
in x and y**. Consequences for extending the code:
- Keeping the periodic box (e.g. **volume-penalization immersed boundary** for a
  fractal/corrugated wall) is MODERATE effort — the FFT-Poisson still works.
- Anything that BREAKS periodicity (inflow/outflow, a real **T-junction**) needs
  REPLACING the pressure solver (multigrid or sparse direct) + inflow/outflow BCs +
  dropping the bulk-flow forcing — a partial rewrite (weeks). Note the proposal's
  "baffle" variant leaves the downstream channel straight and only sets the interface
  at the junction plane, so the full T-junction geometry may be unnecessary.
- Turbulence only sustains at Re_bulk > ~1000 (Re_tau > ~180), far above microfluidic
  Re (1-200); a turbulent run tests the proposal's TNTI-entrainment ANALOGY, not the
  device, and needs a GPU/HPC (infeasible on a CPU-only machine).

## Active direction: (A) corrugated wall via volume-penalization IB

On a GPU+SLURM machine (NVIDIA GB10), pursuing **option (A)** with a staged Schmidt
plan (validate the mechanism at Sc~1-10 with cell-Pe<=2, then push Sc~100-1000 on the
GPU to hit the open DNS niche). Geometry: **oblique grooves first**, herringbone later.

### Phase progress
- **Phase 0 — DONE & VALIDATED.** Volume-penalization immersed boundary built:
  - `immersed.py` — staggered solid-mask builders (chi_u/v/w/c), implicit pointwise
    Brinkman penalization `u <- u/(1 + dt*chi/eta)` (no linear solve, unconditionally
    stable), fluid-volume helpers.
  - `solver.py` — reads an `immersed:` config block, builds masks, applies penalization
    BEFORE projection in `step_imex`/`step_forward_euler`, and retargets the bulk-forcing
    PI controller to the FLUID volume. `projection_fft.py` untouched (periodic box kept).
  - `configs/penalization_slab_test.yaml` + `tests/test_immersed.py` (all pass): a flat
    penalized slab recovers analytic Poiseuille in the reduced gap (u_max/U_bulk=1.475 vs
    1.5; profile rel-L2 4.7%), deep-solid velocity ~0 and LINEAR in eta (O(eta) interior
    suppression; sqrt(eta) is the interfacial slip length, not this), max|div|=0.
- **Phase 1 — DONE.** Oblique-groove wall (`immersed.py` kind='grooves',
  h=h0+A*sin(kx*x+ky*y)) drives a clear helical (v,w) secondary flow — the Eq. 5
  folding mechanism that a flat channel lacks entirely.
  - `configs/corrugated_channel.yaml` (grooves: h0=0.30, A=0.15, n_waves_x=2,
    n_waves_y=1; dt=0.006 under the 2D explicit xy-diffusion limit at low Re),
    `scripts/run_secondary_flow.py` (reports transverse KE fraction f_perp =
    <v^2+w^2>/<u^2>, saves a (y,z) secondary-flow quiver at mid-x).
  - Result: f_perp = 2.3e-3 / 1.7e-3 / 1.3e-3 at Re=10/50/100 (vs EXACTLY 0 for a
    flat wall). Roughly geometric/Stokes-like fraction, slightly decreasing with Re
    (high-Re cases also less developed at fixed t=24); the Re-dependence of MIXING
    enters via Peclet in Phase 2, not via this KE fraction. Qualitative milestone
    (no body-fitted reference, as the lit review flagged).
  - GPU note: run with `PYTORCH_JIT=0` on the GB10 (JIT nvrtc can't target sm_121;
    eager cuFFT/aten kernels work — verified). The capability UserWarning is benign.
- **Phase 2 — DONE (null result at Sc=1).** Koch scalar advected by the corrugated
  secondary flow; N sweep measured.
  - `scalar.py`: `scalar_stats` takes an optional `chi_c` so the M-diagnostic counts
    FLUID cells only. `scripts/run_mixing.py`: passes chi_c when immersed, reports
    cell-Pe=U*dx/D, tracks min/max(c) for dispersive wiggles, usetex via TORCHANNEL_USETEX
    (default 1; `module load texlive` on the HPC). `configs/corrugated_channel.yaml`
    scalar block (Sc=1, neumann, koch, r=3).
  - Result (Re=50, Sc=1, cell-Pe=4.9, ZERO overshoot so no TVD needed):
    **L_mix = 33.8 / 35.0 / 35.1 for N=0/1/2 — flat (marginally WORSE with N).**
    Even WITH secondary flow, no fractal benefit at Sc=1: the decay is diffusion-limited
    (dominated by the slowest large-scale mode), and the fine fractal scales diffuse away
    before the (weak, f_perp~1.7e-3) secondary flow can fold them. This is the proposal's
    stated central risk made concrete, and motivates Phase 3.
  - Figure: `results/figures/mixing_decay_corrugated.png` (persistent, git-ignored).
- **Phase 3 — DONE (decisive NEGATIVE: scaling not reproduced).** GPU N-sweep at
  higher Sc (where fine scales survive) on the GB10. Added `--Sc/--Re/--dt` overrides
  to `scripts/run_mixing.py`. Sc diagnostic: current 32x48x48 grid is clean at Sc<=4
  (cell-Pe~20, zero overshoot with the non-dissipative central scheme); Sc=10 wiggles
  (cell-Pe~49, overshoot 0.20 -> would need TVD or a finer grid).
  - L_mix(N)/L_mix(0): **Sc=1 -> 1.00/1.04/1.04; Sc=4 -> 1.00/1.14/1.15** (L_mix =
    83.9/95.4/96.8 at Sc=4). The proposal predicts r^{-D_f N} = 0.24/0.058 (4x cut per
    generation). We measure the OPPOSITE: the Koch fractal gives NO mixing-length benefit
    — a slight penalty that GROWS with Sc. r^{-D_f N} is NOT reproduced.
  - Mechanism: the oblique-groove secondary flow is STEADY and WEAK (f_perp~1e-3), not
    CHAOTIC. Without exponential stretching the fractal's fine scales just diffuse; the
    late-time decay is mode-limited and the extra area projects onto slow modes (mild
    penalty). Numerics clean (overshoot 0), so this is physics. Figures:
    `results/figures/mixing_decay_corrugated{,_Sc4}.png` (persistent, git-ignored).
  - This is the proposal's central risk confirmed for a steady laminar secondary flow.

## Recommended next (Phase 4): chaotic folding (staggered herringbone)

The remaining FAIR test of the proposal is a CHAOTIC advection flow (exponential
stretching), which is what real passive micromixers (SHM) use and what the fractal
needs to cascade its fine scales before diffusion erases them. Add a staggered-
herringbone groove mask (alternating oblique groove sets along x) — the immersed-
boundary machinery already supports arbitrary h(x,y); only the mask shape changes —
raise the groove amplitude, and rerun the N-sweep at Sc=1..4. If L_mix(N) finally
drops there, the proposal holds in the chaotic regime but not the steady one; if it
stays flat/worse, that is a strong negative result for the claim. This is the decisive
strongest-case experiment.

### Other options (not being pursued now)
- **(B) 'vortices' transient-secondary-flow demo** — cheap illustration that advection
  folds the interface (not sustained, not the device).
- **(C) Turbulent case** (perturbation init at Re_tau~180 + fractal scalar IC) on
  GPU/HPC — tests the TNTI analogy.

Also pending: a literature review of numerical requirements for microfluidic mixing
(high-Schmidt-number false-diffusion pitfall, grid-Peclet criteria, schemes,
validation) — see the project notes on the original workstation if available.

## Running on GPU/SLURM

Set `compute: {device: cuda}` in the config, then `sbatch launch.sh` (check the
`#SBATCH` lines and the config name inside it). Quick check: `python tests/test_scalar.py`.
Mixing run: `python scripts/run_mixing.py configs/koch_mix_demo.yaml --Ns 0 1 2`.
