# Rigid canopy flows (RKPM immersed boundary)

Simulates turbulent open-channel flow over a submerged rigid filamentous
canopy, following Monti, Nicholas, Omidyeganeh, Pinelli & Rosti (2022),
*"On the solidity parameter in canopy flows"* (JFM / arXiv:2205.08050).
Reference config: `config_canopy_monti.yaml` (θ=0°, λ=0.35, Re_b=6000).

**Height convention**: H = 1 is the *outer* region height (canopy tip to free
surface), h = 0.25 the canopy height → total channel height Lz = H + h = 1.25.
All Reynolds numbers use H: Re_b = U_b·H/ν, Re_τ,in/out = u_τ,in/out·H/ν.

## Method

Direct-forcing immersed boundary with RKPM transfer (Pinelli et al., JCP 2010):

- Each filament = a stack of cross-section **rings of 4 surface markers**
  (one ring per wall-normal grid cell crossed; `markers_per_ring: 1` gives a
  centerline-only debug mode). Effective diameter `canopy.diameter` ≈ 2.2 Δx.
- **Filaments are placed randomly within their ΔS² lattice tile** (seeded).
  This is physically required: regular spacing locks the canopy/outer-layer
  exchange (channeling, meandering of ejections/sweeps).
- Interpolation weights: Roma (1999) kernel windows corrected by linear
  reproducing conditions on the non-uniform staggered grid → exact linear
  interpolation, machine-precision partition of unity, one-sided supports at
  the wall. Spreading = transpose × volume ratio (conservative, no global
  solve), rescaled by the Pinelli ε self-response normalization.
- Forcing gain: `alpha: auto` estimates the largest eigenvalue of the
  marker-coupling operator (power iteration at init) and picks a
  stability-safe gain; `forcing.n_iter` multidirect iterations tighten the
  no-slip at the markers.
- The force is applied to u,v,w directly after the implicit z-diffusion and
  before the projection — it never enters the AB2 history, and it is fully
  compatible with `TORCHANNEL_COMPILE=1` and `TORCHANNEL_POISSON_CUDAGRAPH=1`
  (static gather/`index_add_` only; spreading uses atomics, so results differ
  in the last bits between runs).

## Grid

`stretching_type: double` (utils.generate_double_stretched_grid): tanh
clustering at **both** the canopy bed (z=0) and the filament tips
(z = `z_transition` = `canopy.h`), one-sided stretching above.
`gamma_outer: auto` solves for exact C1 continuity at the tips.

## Workflow

1. **Initialize from an existing turbulent field** (any grid — full-channel
   fields use their lower half): `initialization.type: interpolate` +
   `field_file`. The field is trilinearly interpolated, rescaled to `U_bulk`,
   re-projected; time restarts at 0.
2. Run with **constant flow rate** (the built-in PI controller; do not switch
   to a fixed pressure gradient for canopy flows).
3. **Wait for the canopy transient to die out before statistics**: watch
   `canopy_drag_x`, `forcing` and `u_tau_tip` in `timeseries.npz` for a
   plateau (several flow-through times), then set `statistics.t_stats` past it.
4. Post-process with the paper's Reynolds-number conventions:
   `python plot_statistics.py results_canopy_monti/turbulence_stats.npz
   --config config_canopy_monti.yaml --canopy-height 0.25`
   → reports **Re_τ,in** (bed shear) and **Re_τ,out** (total stress at z=h,
   target ≈ 1157 for λ=0.35), normalizes profiles by u_τ,out, marks z=h.

Runtime diagnostics: `canopy_Fx` (total streamwise force on the fluid) and
`u_tau_tip` = sqrt(forcing·(Lz−h)) (equilibrium momentum-balance estimate)
are printed every `n_out` steps and stored in `timeseries.npz`. Filament
positions/markers are saved to `canopy_geometry.npz` in the results folder.

## Tests

```
python tests/test_double_grid.py
python tests/test_canopy_geometry.py
python tests/test_rkpm_reproduction.py
python tests/test_rkpm_convergence.py
python tests/test_ibm_forcing.py
python tests/test_ibm_single_filament.py
python tests/test_ibm_canopy_array.py
python tests/test_field_interpolation.py
python tests/bench_canopy_overhead.py   # production-size, needs the GPU
```
