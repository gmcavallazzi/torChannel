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

## Phase 5 ENABLERS — IMPLEMENTED in torChannel (2026-06-26)

Two additive, flag-gated capabilities were added so the temporal periodic-box test
becomes a FAIR test WITHOUT inlet/outlet (which are non-essential: in the diffusive
limit the temporal duct equals the spatial developing problem via x=U·t, and the
L_mix(N)/L_mix(0) ratio is provably Sc-independent). Both default OFF; the existing
fully-periodic channel path is unchanged (verified bit-for-bit by the regression tests).

1. **Seam-free DUCT — no-slip walls in y (`domain.bc_y: periodic|wall`).** A spanwise
   DCT-II replaces the y-FFT in the pressure Poisson (`projection_fft.py`): y is uniform,
   so a cosine transform (cell-centred Neumann pressure) decouples into the SAME
   tridiagonal-in-z solves; implemented with a precomputed cosine matrix
   (`ky_mod = (2/dy)·sin(πk/(2ny))`), round-trips to ~1e-15. Wall velocity BCs (no-slip
   reflection for u,w; v=0 at y-faces) added to `apply_bc_all` + a wall variant of the v
   y-diffusion in `operators.py` (`_d2v_dy2_raw`); scalar gets no-flux y-walls in
   `apply_scalar_bc`. **With real y-walls the spanwise seam is gone, so the proposal's
   single two-stream interface `init_type:'koch'` is now the correct IC** (the
   `koch_strip` workaround is no longer needed). Validated: manufactured-solution Poisson
   exact to 1e-15; live duct run max|div|=0, spanwise symmetry 2e-16, no-slip enforced,
   u_max/u_bulk≈1.9 (square-duct ~2.1, developing).

2. **High-Sc TVD scalar (`scalar.scheme: central|tvd`).** `advection_scalar_tvd` in
   `scalar.py`: conservative flux-form with a van Leer MUSCL limiter (2-cell halo built
   per direction from the BCs). Monotone/bounded at high cell-Pe where the central scheme
   overshoots (validated cell-Pe≈40: central overshoots ±0.36 / blows up under FE; TVD
   stays in [0,1] to 1e-12 and conserves the mean). Face velocity is 0 at every wall, so
   the limiter never touches a wall flux. End-to-end duct+Koch+TVD at Sc=16: max|div|=0,
   c∈[0,1], mean conserved to 4e-16.

Configs: `configs/duct_koch.yaml` (plain duct, Sc=1, central — the seam-free baseline)
and `configs/duct_koch_highSc.yaml` (duct, Sc=16, TVD). Tests:
`tests/test_scalar_tvd.py`.

### Phase 5 RESULT — duct N-sweep (Sc=1 AND Sc=16): r^{-Df N} REFUTED, Sc-independently
Plain seam-free duct (no folding flow), single-interface Koch IC, N=0/1/2. The duct
laminar flow is unidirectional (v=w=0) and the IC is x-homogeneous, so the scalar is
EXACTLY cross-plane diffusion — the proposal's diffusive limit. Two independent drivers
agree: `scripts/run_mixing.py` (full coupled solver, Sc=1) and
`scripts/run_duct_diffusion.py` (velocity frozen=0, exercises the TVD scalar path; its
Sc=1 reproduces the full solver to 3 sig figs — a method cross-check). `run_mixing.py`
now also records chi(t)=<|grad c|^2>.
- **L_mix(N)/L_mix(0) = 1.000 / 0.982 / 0.981 at BOTH Sc=1 and Sc=16** — identical to
  FIVE significant figures (0.98234 / 0.98078 either Sc). Proposal predicts
  r^{-Df N} = 1.000 / 0.250 / 0.062 (4x cut/generation). Measured ~2%. => Eq. 4 REFUTED,
  and the refutation is **Sc-INDEPENDENT**: high Sc does NOT rescue it.
- **t_mix scales exactly with Sc** (14.64 -> 234.3 = 16.0x), i.e. L_mix ∝ Pe ∝ Sc — the
  textbook diffusive-limit signature, and why the ratio is Sc-invariant.
- **The area-sensitive chi(N)/chi(0) DOES grow with N, and MORE at high Sc**
  (Sc=1: 1.00/1.21/1.19; Sc=16: 1.00/1.25/1.30): the IC genuinely carries the extra
  fractal interface (the null is PHYSICAL, not a resolution artifact) — but the global
  mixing length is set by the gravest diffusive mode (duct width), which is N-blind.
- TVD verified in anger: overshoot ~1e-14 at Sc=16 (central would wiggle at this cell-Pe).
- Mechanism: a straight duct has no transverse stirring, so the fractal's fine scales just
  diffuse before they matter. SAME negative as CaNS, now in torChannel's seam-free duct,
  with the right observable (chi), across Sc=1->16, and with a monotone high-Sc scheme.
- Figure: `results/figures/duct_koch_Nsweep.png` (`scripts/plot_duct_nsweep.py`).

### Phase 5 DECISIVE — herringbone DUCT at high Sc: STRONG GENERAL NEGATIVE
The remaining fair test: give the fractal an active CHAOTIC folding flow at high Sc.
`configs/herringbone_duct.yaml` = Phase-4 staggered-herringbone floor (penalization) +
seam-free duct (`bc_y: wall`) + TVD scalar, Sc=16, N=0/1/2, full coupled solver on GPU
(dt=0.003, AB2-stable). Smoke: max|div|=1.5e-13, f_perp=7.2e-3 (real secondary flow,
4x the Phase-1 oblique groove), TVD bounded. cell-Pe=52 (TVD overshoot ~4e-14; central
would wiggle).
- **L_mix(N)/L_mix(0) = 1.000 / 1.005 / 1.001 — FLAT** (vs proposal 0.25/0.062). No
  fractal benefit even WITH chaotic folding at high Sc.
- The flow IS stirring: N=0 L_mix~49 vs the plain duct's diffusive ~130 (M=0.2) — the
  herringbone rolls roughly halve the mixing length. But they help every N EQUALLY.
- chi(N)/chi(0) = 1.00/1.29/1.31 — the fractal area is present and resolved; it just
  doesn't shorten L_mix.
- Why: the steady herringbone rolls (f_perp~7e-3) stir the GROSS interface but are not
  strongly chaotic (no fast exponential stretching), so the fractal's fine scales diffuse
  before the weak folding can cascade them. Figure: `results/figures/all_regimes_Nsweep.png`.
- Caveat (common-mode, doesn't affect the ratio): the scalar leaks into the immersed
  solid by diffusion (no no-flux at the immersed surface, only at domain walls) — same as
  all prior immersed runs; fluid-mean drifts ~1-2% over a run, identical across N.

### Robustness checks (asked "to be 100% sure")
- **No hidden early-time benefit (threshold-resolved).** Resolving t_mix at M=0.9..0.05:
  the fractal DOES accelerate the FIRST stage (plain duct Sc=16, L_mix(N2)/L(0): M=0.9 ->
  0.51, M=0.7 -> 0.81, M=0.5 -> 0.91, ... M=0.05 -> 0.98), but the benefit decays
  monotonically and never reaches r^{-Df N}=0.06 at ANY threshold. Initial M is
  near-identical across N (0.984/0.979/0.973) => the fractal folds carry ~no extra
  VARIANCE (variance set by the gross 0->1 split), so the enhanced flux drains a
  negligible reservoir. "Max early lead" M0-M_N peaks at only +0.03..0.046 (t~3-17) then
  is erased. So the null is not an artifact of measuring at homogenisation: the early
  effect is real, small, and gone before M~0.3.
- **BC sensitivity.** The RATIO is BC-robust (<4% across periodic channel, plain duct
  Sc=1/16/100, herringbone duct) because L_mix(N)/L(0) is set by the gravest decay mode,
  which is the SAME domain-scale (N-independent) mode for all N. BCs shift ABSOLUTE L_mix
  (and the gravest eigenvalue, Neumann vs Dirichlet ~4x) but not the ratio.
- **Sc reach.** Proposal regime is Sc~1000 (D~1e-9, water; Pe up to ~1e4). Measured ratio
  is identical at Sc=1, 16, 100 (0.982/0.981 to 3-4 sig figs; t_mix ∝ Sc exactly),
  confirming the analytic Sc-independence of the diffusive limit over a 100x span. My
  Sc=16 flow-Pe is already ~800 (in the proposal's range). UNTESTED door: high Sc combined
  with STRONG chaotic stretching (Batchelor) — my herringbone is weak (only ~2.6x mixing
  speedup vs the log(Pe)~100x of a true SHM), so a vigorously chaotic flow at high Sc is
  the one case not closed.

### STRONG laminar chaotic mixer — DONE (the open door, now closed): still flat
`configs/herringbone_duct_strong.yaml`: deep staggered-herringbone grooves (h0=0.18, A=0.28,
thin Lz=0.7 gap, 4 chevron cycles), Re=100, Sc=16 TVD, 64^3. Key finding from a config probe:
the chaotic strength is GEOMETRY-driven, not Re-driven (tripling Re 50->200 barely moved
f_perp 0.022->0.036; deepening grooves moved it 7x). So a vigorous laminar mixer is reachable
WELL WITHIN the proposal's Re in [1,200] — no turbulence, no edge-of-range Re needed.
- f_perp=0.11 (15x the earlier weak herringbone); it mixes hard: L_mix(N=0)=29.6 vs the
  diffusive 234 (8x faster) — genuinely strong, approaching but not at the log(Pe) floor.
- **L_mix(N)/L_mix(0) = 1.000 / 0.977 / 0.977 — still FLAT** (2.3% dip that SATURATES; N=2
  does not beat N=1). chi(N)/chi(0) = 1.00/1.43/1.49 (strongest fractal-area signal yet, since
  the strong mixer + high cell-Pe preserves more interface). Predicted r^{-Df N}=0.25/0.062.
- Snapshots (`results/figures/snapshots_herringbone.png`): by t=4 the chaotic flow imposes its
  OWN folding pattern and N=0 vs N=2 are visually indistinguishable through homogenisation —
  the fractal IC is folded away, not amplified. Matches the a-priori log-law bound
  L_mix(N)/L_mix(0) ~ 1-(Df-1)ln(r)N/lnPe (a few % dip, never the power law).
- TVD verified at cell-Pe=78 (overshoot ~9e-14). Snapshot/IC figures via
  `scripts/snapshot_fields.py`; combined plot `scripts/plot_all_nsweep.py`.

A standalone rebuttal write-up (setup + TikZ sketch, all tables/figures, mechanism,
limitations) lives in `rebuttal/` (git-EXCLUDED via .git/info/exclude, not tracked).

### FRACTAL INLET SURFACE (proxy) — DONE: a PENALTY, not a benefit
The proposal's OTHER implementation is the fractal inlet SURFACE (Koch-corrugated orifice
wall acting via near-wall secondary flow), distinct from the baffle (interface) we tested
above. Proxy: `immersed.py` kind='koch_herringbone' — the staggered-herringbone ridge gets
a generation-N area-balanced Koch zigzag (same generator/Df as the scalar), so the wall
carries multi-scale (fractal) corrugation; N=0 == smooth herringbone. Swept the WALL
generation with a FLAT two-stream interface (the wall does the folding), Sc=16 TVD, Re=100
(`configs` via `scripts/run_wall_sweep.py`). div~1e-12, stable.
- **L_mix(N)/L_mix(0) = 1.000 / 1.162 / 1.169 — a ~17% PENALTY** (vs proposal 0.25/0.062).
  The fractal wall mixes WORSE, saturating. f_perp ~0.045-0.049 across N (the corrugation
  adds no transverse energy).
- Mechanism (matches chaotic-mixing theory, t_mix~log(Pe)/lambda): the smooth chevron drives
  COHERENT counter-rotating rolls (high stretching rate lambda); the fractal corrugation
  FRAGMENTS the rolls into weaker multi-scale motion -> LOWER lambda -> slower mixing. The
  fractal wall degrades the coherent folding rather than folding at finer scales.
- So BOTH proposal implementations fail r^{-Df N}: baffle (interface) flat ~0.98; surface
  (wall) a ~1.17 penalty. The only fully-faithful untested variant is the surface in a TRUE
  developing junction (inflow/outflow), but the periodic-box proxy already shows a penalty.
  Wall-shape figure: `results/figures/fractal_wall.png`.

Lit context (web search): existing Koch/Cantor fractal micromixers enhance mixing via
DOWNSTREAM obstacles at a documented PRESSURE-DROP cost (Tian/Xiong/Chen; Cantor-GA Sci Rep
2022; Murray's-law baffles) — the Falk-Commenge coupling the proposal claims to break. The
inlet-only fractal is novel, but chaotic-mixing theory (t_mix~log(Pe)/lambda, striations
~sigma^-n from initial length ~w) says the initial interface gives at most a LOGARITHMIC,
not power-law, benefit — i.e. r^{-Df N} is inconsistent with standard mixing physics, which
our DNS confirms.

## Phase 5 — FAITHFUL test: developing duct (inflow/outflow) + fractal WALL surface

The earlier negatives were temporal periodic boxes with the fractal only in the scalar
IC. Phase 5 closes both gaps the proposal could complain about:

1. **Real developing flow.** Added inflow/outflow streamwise BCs (`domain.bc_x: inout`,
   `bc_y: wall`): prescribed inlet profile, convective/mass-corrected outflow, all-Neumann
   Poisson with a gauge pin. Validated against literature — square duct u_max/u_bulk=2.096
   (0.4%), circular pipe (volume-penalization immersed disc) u_max/u_bulk=1.983 vs
   Hagen-Poiseuille 2.000 (0.8%). Fluid divergence ~1e-12 throughout.
2. **The actual fractal SURFACE, not a proxy.** New immersed kind `pipe_koch`: a circular
   orifice whose WALL is a symmetric Koch-snowflake ring — the area-balanced Koch generator
   (4 sub-segments each length 1/r, Df=log4/log3=1.262, verified: arclength grows as (4/3)^N)
   tiled `n_lobes=6` times around the azimuth, unit-peak normalised so the envelope amplitude
   is N-independent ("more scales, same envelope"), localised at the INLET by a streamwise
   half-cosine envelope; downstream the pipe is the smooth disc. NO obstacles — the wall
   itself is folded, as in the proposal's Fig. 1. Combined with the Koch BAFFLE (scalar
   interface, gen N). Both scale together with N. (`scripts/check_fractal_wall.py` verifies
   the border dimension; an earlier n_lobes=1 wrap gave a too-gentle asymmetric border and
   was replaced by the 6-lobe symmetric ring.)

**Run** (`scripts/run_pipe_koch_mixing.py`, Re=40, Sc=0.5, round duct Lx=10, R=0.42,
corrugation 13% of R, n_lobes=6, inlet_len=1.0, grid 96x48x48, N=0,1,2; fluid div verified
1.15e-12, the larger all-cell div=0.18 is solid-confined Brinkman slip and does NOT touch
the fluid):

  The streamwise segregation M(x) curves for N=0, 1, 2 **collapse onto a single line**
  (mixing length identical to within line width; M→0.5 at x≈0.7 and M→0 by x≈4 for all N).
  L_mix(N)/L_mix(0) ≈ 1, NOT the predicted r^{-Df N} ≈ 0.25, 0.06.
  Figure: results/figures/pipe_koch_mixing.png.

This is the most faithful realisation achievable here — the real device geometry (round
orifice, folded wall at the junction, developing flow) — and it reproduces the same NULL.
The fractal inlet surface confers no mixing-length benefit.

## Phase 6 — Sc=10 campaign + RECALIBRATED interpretation

Important framing correction (per discussion with the proposal author): the earlier
"decisive null" language oversold the result. There IS a real, monotonic N-signal even at
Sc=0.5 — the inlet segregation drops with N (M(in) = 0.811, 0.796, 0.775 for N=0,1,2),
i.e. the fractal's extra interfacial area genuinely speeds up EARLY mixing. What collapses
is only the ASYMPTOTIC mixing length L_mix (set by the gravest cross-channel mode, which is
N-independent). So the honest statement is: the fractal enhances mixing transiently; whether
that shortens the full L_mix depends on regime, and the proposal's regime (high Sc; the
surface's Re-driven near-wall secondary flow) is exactly where the transient window is longer.
The proposal author works in a different flow regime — do not over-read these laminar,
low-Pe results as a refutation of that regime.

Compliance check vs the proposal reference (draft_BS.pdf): Re=40 ∈ [1,200] ✓, geometry
(circular fractal inlet surface at constant area + Koch baffle, r=3, Df=1.262, N=0..4) ✓,
but **Sc**: proposal ~1000 (water; Pe=Re·Sc up to ~2e5), ours was 0.5 (Pe=20). The IB
(volume penalization) is treated IMPLICITLY so it imposes NO dt restriction; the dt limit
is the explicit AB2 in-plane advection-diffusion (CFL), which tightens with grid refinement.
Cost to reach high Sc scales ~Pe² (temporal box) / ~Pe³ (developing duct), because steps∝Pe
(mix time) and cross-section grid∝√Pe (Batchelor scale, else false numerical diffusion fakes
a null). Sc=10 is hours/run; Sc=100 days; true Sc=1000 is years/run (infeasible by direct DNS
— same reason the proposal routes high-Sc to finite-volume OpenFOAM).

Campaign (`scripts/mixing_campaign.py`, fixed verified dt, convergence early-stop,
incremental history+snapshot+final checkpointing; SLURM in `slurm/`):
- **baffle** (temporal box, smooth square duct, Koch baffle IC): pure cross-sectional
  diffusion (v=w=0 in a straight duct) = the direct Eq.4 diffusive-limit test. dt=4e-4.
- **surface_baffle** (short developing duct Lx=6, circular fractal inlet wall + Koch baffle):
  steady M(x), resolves the near-inlet N-dependence. dt=5e-4.
Both at Sc=10, N=0..4. Results pending (jobs 277/278).

## SPEEDUP — parabolic (space-marching) steady scalar solver (2026-06-29)

The campaign and `run_frozen_scalar.py` compute a STEADY developing-duct field by marching
the UNSTEADY scalar to steady state (~Pe pseudo-time steps; cost grows linearly with Sc, the
wrong way for the high-Sc/high-N door). For a developing duct (`bc_x:'inout'`) with a steady,
forward (u>0) frozen velocity, the steady scalar can instead be obtained in a SINGLE downstream
sweep by dropping the (negligible at high Pe) streamwise-diffusion term: the balance
`u dc/dx = D(d2_yy+d2_zz)c - (v d_y+w d_z)c` is then PARABOLIC in x and marches plane-by-plane.

Implemented as an ADDITIVE option (reuses the batched tridiagonal solver and the same
no-flux z-stencil; replaces nothing):
- `scalar.march_scalar_steady(...)` — streamwise 2nd-order upwind (BDF2, low false diffusion);
  cross-plane (y,z) diffusion by ALTERNATING-DIRECTION line relaxation (ADI: z-implicit then
  y-implicit per sweep); transverse advection (central, weak) lagged. Returns the steady field.
- `ChannelFlow.solve_scalar_parabolic(...)` — thin method alongside `advance_scalar` /
  `advance_scalar_ssprk3`. Requires `bc_x='inout'`, `wall_bc='neumann'`.
- `scripts/march_scalar.py` — driver (solve velocity to steady, freeze, ONE parabolic sweep);
  writes a `{tag}_final.npz` in the campaign layout (Mx/x/scalar/u/chi_c) so the
  plot_campaign_* scripts work; `--compare <final.npz>` validates Mx against an existing run.
  (Also fixed `mixing_campaign.base_config` to accept `nx/ny/nz` — the ported
  `run_frozen_scalar.py` already assumed this.)

Validation (baffle, smooth pipe, v=w=0, frozen campaign velocity):
- Sc=10, N=2: converged Mx matches the campaign to ~1% on L_mix (M=0.8: 1.13 vs 1.13; M=0.7:
  3.04 vs 3.00). Mean conserved to 1e-6, c in [0,1] exactly.
- Sc=100, N=2: converges in ~8 ADI sweeps (vs ~150 at Sc=10 — convergence is FAST at high Sc,
  where the streamwise term dominates the diagonal; that is the regime the marcher is for).
- The near-inlet / far-field differences vs the campaign are the campaign's OWN streamwise
  numerical diffusion (1st-order-upwind-in-x at nx=192 gives false diffusion ~0.016 >> physical
  D=2.5e-4); the parabolic marcher drops streamwise transport error by construction, so it is
  the more faithful high-Pe solution. NOTE: low Sc needs many ADI sweeps (use the existing
  time-march there — it is already cheap at low Pe); the marcher's win is the high-Sc/high-N door.

This is the tool for the open N>=4 high-Sc test (needs a resolved cross-section ny=nz>=256):
`python scripts/march_scalar.py --mode baffle --Sc 100 --N 4 --nx 256 --ny 256 --nz 256`.

## OVERALL CONCLUSION (torChannel, all accessible laminar regimes)
L_mix(N)/L_mix(0) is FLAT (~1.0, never the proposed r^{-Df N}) across: plain duct Sc=1/16/100
(0.98, Sc-independent to 5 sig figs), weak herringbone duct (1.00), AND a STRONG laminar
chaotic mixer (0.977, saturating). The area-sensitive chi DOES see the fractal everywhere
(1.2-1.5x) — the null is physical, not a resolution/observable artifact. **Eq. 4 (the
proposal's only sharp falsifiable claim) is refuted in every laminar regime reachable here**,
including its intended high-Sc regime AND a vigorous deterministic chaotic-advection flow
(geometry-driven, fully within Re in [1,200]; NOT turbulence). The remaining untested door is
now essentially just the headline unchanged-pressure-drop (Falk-Commenge) claim, which needs
machinery beyond the present periodic-box temporal framing.

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
