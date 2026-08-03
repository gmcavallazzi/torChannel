# Re_τ = 180 open channel — validation case

Open channel (no-slip bottom, free-slip/symmetry top), δ = L_z = 1, driven at
constant flow rate U_b = 1. This is the reference case for torChannel's
validation figures and the base flow for the control examples.

| | |
|---|---|
| box | 4π × 2π × 1 |
| grid | 192 × 192 × 128, tanh clustering at the bottom wall (`gamma: 1.6`) |
| resolution | Δx⁺ = 11.8, Δy⁺ = 5.9, Δz⁺_min = 0.37, Δz⁺_max = 2.44 |
| ν | 3.4843e-4 (`flow.Re: 2870` — the solver reads this as ν = 1/Re) |
| averaging | 400 t.u. after a 100 t.u. transient ≈ 25 eddy turnovers (δ/u_τ = 15.9) |
| cost | ~1 h on one GB10 |

## Regenerating the seed field

`seed180_cans.npz` (207 MB) is **not** committed. Regenerate it from the CaNS
checkpoint with the bundled converter:

```bash
python scripts/cans_to_npz.py \
    /home/giorgio/CaNS_DRL/run0_theory_big/data \
    examples/re180_open/seed180_cans.npz \
    --fld ../theory_dump/no_actuation_lumley/fld_0120.bin \
    --Lx 10.68 --Ly 3.2 --nu 3.4843e-4
```

Expected output: `u_bulk = 1.000000  u_tau = 0.06295  Re_tau = 180.7`.

Note `grid.out` lives in `run0_theory_big/data/`, not alongside the field dumps —
hence the `../` in `--fld`.

**Pick a late snapshot.** The `no_actuation_lumley` series is a spin-up followed
by a long plateau: `fld_0001` sits at Re_τ = 154, and u_τ only settles into its
179–187 band (mean ≈ 183) from roughly `fld_0020` onward, holding there through
`fld_0120`. Seeding from an early file starts the flow ~15% low in drag. The `t`
resets visible at `fld_0011/0026/0041/0061` are DRL episode boundaries; u_τ runs
continuously across them.

Any equilibrated turbulent field works as a seed — this one is used because it is
closest to the target flow.

## Caveats worth stating in any figure

- **The seed gets stretched.** The CaNS box is 10.68 × 3.2 and this box is
  4π × 2π, so the `interpolate` initializer maps x/y proportionally: ~18% stretch
  in x, ~2× in y. Spanwise streak spacing therefore arrives wrong and needs the
  100 t.u. transient to re-equilibrate. It also means comparison against the CaNS
  baseline is **statistical, not like-for-like**.
- **MKM is a closed channel.** Expect agreement in the near-wall region and a
  genuine, explainable difference toward the centreline. Label it rather than
  hide it.
- **Do not use `statistics.z_plus_target` here.** That legacy path hard-codes
  δ = L_z/2 and averages a "bottom wall" plane with a "top wall" plane — both
  wrong for an open channel, where δ = L_z and there is no top wall. With
  `z_plus_target: 15.0` it actually selects z⁺ = 7.4 and pairs it with a plane at
  the free surface. This config uses the multi-plane `spectra_z` mode, which
  takes physical heights, instead.

## Running

```bash
sbatch slurm/re180_open_gb10.sh
# or locally:
PYTORCH_JIT=0 TORCHANNEL_COMPILE=1 TORCHANNEL_POISSON_CUDAGRAPH=1 \
    python main.py examples/re180_open/config.yaml
```
