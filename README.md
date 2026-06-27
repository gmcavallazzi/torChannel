# TorChannel — private results (MERGE fractal-mixing study)

**Private companion** to the public [`torChannel`](https://github.com/gmcavallazzi/torChannel)
DNS solver. This repository holds the **simulation results and working branch** for the
MERGE fractal-interface mixing study — data that is not (yet) for public release. The base
solver code is identical to the public repo; only the `results/` data and study-specific
analysis scripts are private here.

> Branch: `feature/passive-scalar` (single branch — code + results).
> The public repo carries the same branch **without** the results.

---

## What is being tested

The **MERGE** proposal (de Oliveira & Scheid, *"Fractal Boundary Conditioning for Passive
Mixing Enhancement in Microfluidic Channels"*) makes one sharp, falsifiable claim:

```
L_mix(N) / L_mix(0)  ~  r^{-D_f N}        (proposal Eq. 4)
```

Imposing a generation-`N` Koch fold (contraction ratio `r=3`, dimension
`D_f = log4/log3 ≈ 1.262`) on a fluid–fluid interface should shorten the mixing length
by a factor ~4 per generation. We probe this with passive-scalar DNS in the accessible
laminar regimes (see `docs/MERGE_CONTEXT.md` for the full history and overall conclusion).

---

## The baffle Sc=10 campaign (what's in `results/`)

A **short developing pipe** (round orifice, smooth wall, inflow/outflow), with a
generation-`N` Koch interface injected at the inlet (the "baffle"). The steady streamwise
profile `M(x)` of the segregation intensity carries the near-inlet `N`-dependence, and the
mixing length is where `M(x)` crosses a threshold.

- Geometry: `Lx=6`, inscribed disc `R=0.42` in a unit square, grid `64×96×96`.
- `Re=40`, `Sc=10`, `dt=5e-4`, run to steady state (drift `< 2e-5`) or `120k` steps.
- Generations `N = 0 … 4`.

### Results layout

```
results/campaign/baffle_Sc10_N{N}_history.npz   # M(x), drift, |div| timeseries
results/campaign/baffle_Sc10_N{N}_snaps.npz     # downstream (y,z) & (x,y) snapshots
results/campaign/baffle_Sc10_N{N}_final.npz     # full scalar/u field + steady Mx(x)
```

Each `_final.npz` holds `scalar (nx,ny,nz)`, `u`, the steady `Mx (nx,)` profile, `x`, and
the solid mask `chi_c`.

### Status

| N | state |
|---|-------|
| 0, 1, 2, 3 | ✅ present (clean, point-in-polygon IC) |
| 4 | not run — benefit saturates by N=2/3 (see result below) |

**Result (L_mix at M=0.70): ratio 1.00 / 0.90 / 0.76 / 0.75 for N=0/1/2/3** — a real but
**saturating** enhancement (N=3 ≈ N=2, no further gain), nowhere near Eq.4's
1.00 / 0.25 / 0.06. The area-sensitive signal is present (the inlet interface folds with N),
but the asymptotic mixing length is set by the gravest cross-channel diffusive mode, which is
N-blind. Figures in `results/figures/campaign_Sc10_partial/`.

`surface_baffle` (fractal **wall** surface variant) results to be added later.

---

## Koch-interface sign fix (important)

The earlier N=3/N=4 inlets carried a **sign artifact**: `koch_interface_yz` signed the
cross-section by the nearest-segment side of an *open* polyline, which is ambiguous near the
domain corners. For `N ≥ 3` the higher-frequency folds flipped the sign there, planting
wrong-stream patches that diffused into spurious ~0.5 blobs and intruded into the fluid by
`N=4` — inflating the interfacial area and biasing those cases to look better-mixed.

**Fix:** keep the smooth distance magnitude, but take the *sign* from a robust
point-in-polygon test (interface closed along the top wall). Verified: corners saturated for
all `N`, interface-cell count monotonic, mean exactly 0.5, `N ≤ 2` unchanged, scalar tests
pass. `N=0,1,2` here were unaffected; `N=3,4` are being regenerated.

---

## Regenerating the figures

The two **canonical** figures (`results/figures/campaign_Sc10_partial/`):

```bash
module load texlive            # figures use LaTeX rendering
export TORCHANNEL_USETEX=1 PYTORCH_JIT=0

# steady M(x) per N (left panel only)
python scripts/plot_campaign_temp.py      --mode baffle --Sc 10 --Ns 0 1 2 3 --thr 0.7 --left-only
# developing (y,z) cross-sections down the pipe, clipped to the wall, M(x) labelled
python scripts/plot_campaign_xsections.py --mode baffle --Sc 10 --Ns 0 1 2 3
```

`scripts/plot_campaign_snaps.py` (inlet circles) and `scripts/plot_campaign_circumference.py`
(wall-circumference sampling) are **diagnostics/tests**, not canonical outputs.

Campaign driver: `scripts/mixing_campaign.py`; SLURM: `slurm/baffle_sc10*.sh`.

---

## See also

- Public code + base docs: https://github.com/gmcavallazzi/torChannel
- Full study context, phase history, and conclusion: `docs/MERGE_CONTEXT.md`

---

*Private repository — results not for public distribution. Base solver: MIT licensed (see `LICENSE`).*
