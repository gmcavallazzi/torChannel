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

## MAJOR PIVOT: framing flaw found -> moved definitive test to CaNS

A deep audit (prompted by "is it resolved / are parameters wiping out the effect?")
showed the torChannel results are largely a FRAMING ARTIFACT, not a physics refutation:
- Koch IC resolution is FINE (N=1/2 capture 99%/91% of the fractal interfacial area).
- But the metric is structurally blind to N: initial M is ~identical across N
  (0.997/0.991/0.988) — the fractal carries negligible variance; the variance-based M
  decaying in a PERIODIC box is dominated by the gravest mode (set by the box), N-insensitive.
- Spanwise PERIODICITY adds a spurious flat seam interface (c jumps 0->1 at y=0/Ly).
- The area-sensitive observable, scalar dissipation <|grad c|^2>, IS N-sensitive
  (1.0/1.98/2.08x) — so the right observable + a seam-free geometry are needed.

Fix = **CaNS** (`/home/giorgio/CaNS`, fresh clone, built on the GB10): a DUCT has walls
in y AND z (no spanwise seam), built-in scalar transport, GPU. See memory `cans-gb10-build`.
WORKING case `run_merge/` (periodic duct + passive Koch scalar, GPU): added `iniscal='koc'`
(reads data/scalar_ic.bin from `utils/gen_koch_ic.py`, reuses our koch_interface_yz) and a
`mixing_stats` diagnostic writing data/mixing_s_001.out = [time, M, var, chi=<|grad c|^2>].
Verified end-to-end on GPU: M and chi both decay. CaNS limitation: developing (inflow/outflow)
duct = spatial mixing length is CPU-ONLY (GPU only supports periodic streamwise); CaNS has
NO immersed boundary (patterned wall would need penalization ported to Fortran).
NEXT in CaNS: N-sweep comparing chi(t) & M(t) across N=0/1/2; then high Sc; folding needs a
turbulent duct or a ported penalization wall.

### CaNS N-sweep — DONE (seam-free duct, Sc=1): confirms the audit's two-part story
Driver `run_merge/run_sweep.sh` (loops N=0/1/2: regen `scalar_ic.bin` via
`utils/gen_koch_ic.py --N`, truncate `data/mixing_s_001.out` — CaNS opens it in APPEND
mode, so it MUST be removed between runs — run `./cans`, archive `sweep_out/mixing_N{N}.out`).
Grid 32x64x64, duct Lx,Ly,Lz=2,1,1, Poiseuille init, nu=alpha=0.01 (Sc=1), 10000 steps to
t~31.9. Runs are independent (restart=F, fresh IC each); ~70 s for all three on the GB10.
- **chi (scalar dissipation) IS N-sensitive early — as predicted.** chi(t=0.064) ratio =
  **1.00 / 1.24 / 1.27** (N0/N1/N2): the extra fractal interface really does raise the
  dissipation rate. But the enhancement EVAPORATES fast: curves collapse by t~1, and by t~2
  N>0 is marginally BELOW N0. Fine scales (N=2 striations ~0.13 wide) diffuse in t~w^2/D~1-2
  t.u. — a few % of t_mix.
- **Mixing length is N-insensitive — NOT the proposal's scaling.** L_mix(N)/L_mix(0) (M=0.05)
  = **1.00 / 0.98 / 0.98** (t_mix = 29.3/28.8/28.7). Same at every threshold (M=0.5/0.2/0.1).
  Proposal predicts r^{-Df N} = 1.00/0.25/0.06 (4x cut per generation). We measure ~2%,
  essentially flat.
- **Why:** a straight laminar duct has NO secondary/transverse flow (duct secondary flow is a
  turbulent effect) — pure cross-stream diffusion + streamwise advection. Without folding the
  fractal's fine scales just diffuse before they matter; global M is set by the gravest
  diffusive mode (tau~(Lz/pi)^2/D~10), which is N-blind. This is the SAME negative as
  torChannel but now in a seam-free geometry AND with the right observable shown explicitly:
  the area-sensitive chi sees the fractal; the mixing length does not, because there is no
  stretching to keep the fine scales alive.
- Figure: `results/figures/cans_duct_koch_Nsweep_Sc1.png` (M(t) collapse + chi(t) early split).
- Confirms: the proposal needs a CHAOTIC/stretching flow, not just a seam-free duct. NEXT
  remains high-Sc (TVD scalar) and/or a folding flow (turbulent duct, or penalization wall
  ported to CaNS Fortran).

torChannel additions kept for the record: seam-free `koch_strip` IC + `scalar_dissipation`
in scalar.py (the cheap reframe, now superseded by the CaNS duct).

## Active direction (superseded by CaNS pivot above): (A) corrugated wall via volume-penalization IB

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

## Phase 4 — DONE (herringbone built); plus a RESOLUTION/PARAMETER AUDIT

Staggered-herringbone (SHM) geometry added: `immersed.py` kind='herringbone'
(chevron grooves, off-centre apex, apex side flips every streamwise cycle ->
alternating counter-rotating rolls = chaotic-advection potential). `solver.py`
plumbs apex_frac/stagger; `configs/herringbone_channel.yaml` (nx=48, h0=0.30,
A=0.20, 3 chevron cycles). Geometry + flow verified (mask map + (y,z)
recirculation figures in results/figures/).

- Herringbone Koch N-sweep, Sc=1: **L_mix = 36.0/35.8/35.8 — FLAT** (no benefit,
  no penalty). Crucially N=0 (36.0) is NO faster than the oblique-groove N=0 (33.8)
  despite a completely different flow.

### Audit (prompted by "is it resolved / are parameters wiping out the effect?")
- **Spatial resolution OK (not the cause).** On-grid Koch interfacial length
  (integral |grad c|) vs a refined ny=384 reference resolves **N=1: 99%, N=2: 91%**
  of the true fractal area — the extra interface really is in the IC. (N>=3 aliases;
  do not use.)
- **No hidden early-time benefit.** Time to M=0.5/0.2/0.1/0.05 is identical or
  slightly worse for N>0 at every threshold (Sc=4 oblique): not a tail artifact.
- **Real masking factor #1 — Sc too low.** At Sc<=4 the finest N=2 striations
  (~0.13 wide) diffuse away in t~w^2/D ~ 1-3 t.u. out of t_mix~36-84 — gone in a few
  % of the run. The proposal's regime is Sc~1e3; unreachable here without a TVD
  scalar + much finer grid (Sc=10 already wiggled, overshoot 0.20).
- **Real masking factor #2 — folding localized in a tall box.** Grooves influence
  z<~0.8 but the channel is Lz=2 and the interface spans the full height; the global
  M is set by the gravest DIFFUSIVE mode of the unstirred upper ~60% (~Lz^2/pi^2 D),
  which is why both geometries and all N collapse to L_mix~34-36 at Sc=1. We are
  measuring background diffusion, not mixing enhancement.

**Conclusion:** the negative result is solid *within the accessible low-Pe,
floor-stirred regime*, but that regime is stacked against the proposal — it has NOT
been fairly tested in its intended high-Pe, fully-stirred chaotic regime.

### Next (Phase 5): give the mechanism a fair test
1. **Thin channel** (Lz ~ 0.8-1.0, grooves filling the gap) so the chaotic flow
   stirs the WHOLE domain and the gravest mode IS the stirred scale — fixes #2 cheaply.
2. Then **high Sc with a TVD/flux-limited scalar** so fine scales survive — fixes #1
   (the expensive, decisive step).

---

## (superseded) earlier recommendation — chaotic folding (staggered herringbone)

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
