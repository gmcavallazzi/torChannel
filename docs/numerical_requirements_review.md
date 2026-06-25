# Numerical requirements for microfluidic mixing simulation — literature review

Distilled from a multi-source, adversarially-verified literature search (17 sources
fetched, 79 claims extracted, 25 verified by 3-vote refutation — 24 confirmed, 1
killed). Confidence tags reflect that verification. Purpose: inform the design of our
DNS-style passive-scalar mixing simulations (this `torChannel` code and the `merge`
prototype) for the MERGE fractal-mixing project. See `MERGE_CONTEXT.md` for project
context.

## TL;DR

At low Re (1-200) and high Schmidt number (Sc ~ 1000, Pe ~ 1e3-1e6), the dominant
numerical pitfall is **FALSE (numerical) diffusion**. It masks the true molecular
diffusivity, smears steep concentration gradients, and **systematically
overestimates mixing efficiency**. Controlling it — not passing a grid-independence
test on the mixing index — is what makes a high-Pe mixing simulation trustworthy.

## 1. The false-diffusion problem (core finding, high confidence)

- **False diffusion is THE central pitfall in high-Pe micromixer CFD.** It gives an
  erroneous estimate of molecular diffusion and **overestimates mixing**; evaluating
  mixing without analysing the numerical-diffusion contribution yields overestimated
  performance. At high cell-Pe "the molecular diffusivity of the fluid is completely
  masked by false diffusion errors." [high] — Bayareh 2021 review; Hadjigeorgiou &
  Kokkoris 2019.
- **Its magnitude is governed by three things: cell (grid) Péclet number, grid type,
  and discretization scheme.** High Sc/Pe → advection-dominated transport → steep
  gradients that low-order schemes mis-approximate → false diffusion. With a fine
  grid or high-order scheme it need not arise. [high]

## 2. The methodological lesson (high confidence)

- **Do NOT use the mixing index as a grid-convergence criterion.** Most published work
  picks resolution just to pass a grid-independence test on the mixing index — but the
  mixing index is itself corrupted by false diffusion, so this is circular. "Use of the
  mixing efficiency parameter for grid convergence studies may not be appropriate
  unless cell-Pe is quite low." [high] — Bayareh 2021; Okuducu & Aral 2018.
- **Rigorous practice = error-driven adaptive meshes + systematic mesh-independence +
  explicit false-diffusion control.** [high] — Hadjigeorgiou et al. 2021.

## 3. Concrete numbers / criteria (high confidence unless noted)

- **Cell-Péclet target:** false diffusion is negligible at **cell-Pe ≤ ~2** (textbook).
  Device-specific studies found numerical-diffusion effects only "tolerable" at
  **cell-Pe ~50 (Re=240) and ~100 (Re=120)** for swirl mixers — and even then ~10%
  mixing error persisted on 10-16M-element meshes. The ~50-100 figure is
  device-specific, NOT universal; the safe target remains cell-Pe ≤ 2. [high]
- **Grid geometry is decisive.** In 3D T-mixers, **flow-aligned hexahedral meshes give
  false diffusion 4-5 orders of magnitude lower than tetrahedral/prism**. At Re=100:
  hex false diffusivity ~1e-13 m^2/s (0.5% false mixing) vs ~1e-9 (prism) and ~1e-8
  (tet). Matching hex accuracy on unstructured tets would need ~1e9 to >1e19-1e22
  cells (infeasible). [high] — Okuducu & Aral 2018.
- **Diagnostic to isolate false diffusion:** run the scalar transport twice — once with
  molecular **D = 0** (gives D_effective ≈ D_numerical) and once with physical D — so
  **D_effective = D_molecular + D_numerical**. A scheme-independent method exists to
  compute an average false diffusivity from any solution and its effect on scalar
  decay rate. [high] — Okuducu & Aral 2018; Liu & Lin 2011.
- **Usable window:** there is a range of molecular diffusivity in which average false
  diffusion < molecular diffusion, so decay/mixing rates can be computed accurately —
  and it "covers most liquid solutions in chemical/biochemical engineering." Below it
  (high enough Pe), false diffusion dominates. [high] — Liu & Lin 2011.

## 4. Do people reduce Sc, or resolve real Sc? (high confidence)

- **Practitioners often use the REAL high-Sc diffusivity, accepting very high cell-Pe.**
  Example T-junction: crystal violet D=3e-10 m^2/s (no Sc reduction), **Pe 3.3e2-3.3e5,
  cell-Pe ~5000 on the finest hex mesh at Re=100, Δx = 2.0-6.6 μm**. A FEM study used
  D=5.75e-10 (dye in water), Sc ~1740, Re ~1, Pe ~1.4e3. [high]
- **Staggered herringbone (SHM)** sims hit **Pe up to ~1e6** (biomolecule D) and handle
  the resulting instabilities with **stabilization methods** — but the
  stabilization-induced artificial diffusion must itself be controlled via adaptive
  meshes + mesh-independence. [high] — Hadjigeorgiou et al. 2021.

## 5. Methods & tools (high confidence)

- **Finite-volume (OpenFOAM-style)** and **finite-element (COMSOL "Laminar flow" +
  "Transport of diluted species", stationary study)** are the standard workflows over
  Re ~ 0.1-100. [high]
- Some studies use only **first-order/linear scalar discretization** with a
  Courant-limited step — a false-diffusion liability at high Pe (coarse meshes
  overstated mixing index 86-94% vs ~63% converged — the classic false-diffusion
  signature). [high]
- Lattice-Boltzmann and spectral/DNS appear less commonly in this specific
  literature; FVM/FEM dominate.

## 6. Validation & mixing indices (high confidence)

- **Villermaux-Dushman** iodide/iodate reaction is the standard micromixing benchmark.
  **But the segregation index is strongly concentration-set dependent** — not
  comparable across studies unless concentrations match; a **concentration-free mixing
  time** is preferred for comparison. [high] — Falk & Commenge 2011.
- Mixing quantified by a mixing index / intensity of segregation / coefficient of
  variation of concentration (consistent with our `M = std(c)/std_max`).

## 7. Device-specific results (Koch/fractal baffles) (high confidence on trend)

- Koch/fractal-baffle micromixers are an active passive/chaotic-advection class.
  Reported: secondary Koch fractal baffle (SKFB) > primary; best staggered SKFB
  (30°, 0.1 mm) exceeds **95% at Re=0.05 AND Re=100**; Koch snowflake baffle reaches
  **99.70% at Re=100**, rising with fractal-iteration order, baffle count, spacing.
  **Caveat:** these are single-group CFD mixing-index outputs, **not experimentally
  validated**, and are themselves potentially subject to the false-diffusion
  overestimation this review warns about. Treat absolute percentages as
  paper-specific. — Tian et al. 2019; Xiong/Chen-type studies.

## Open questions the literature did NOT settle (directly relevant to us)

1. **No clear cells-per-striation / Batchelor-scale criterion for resolving interfacial
   FRACTAL structure** (as opposed to a bulk mixing index). This is exactly the
   resolution requirement we need to measure D_f(x) of the scalar interface — an
   apparent gap.
2. **No quantified full-resolution (DNS) cost for real Sc~1000 in a 3D chaotic mixer.**
   Real-D runs exist for T-junctions (Pe up to 3.3e5) and stabilized SHM (Pe~1e6), but
   nobody reports a true DNS-style fully-resolved scalar at Sc~1000 with cell counts /
   CPU-hours. Potential niche for our approach.
3. No head-to-head false-diffusion comparison of high-resolution schemes (QUICK, MUSCL,
   TVD, WENO) with cells-per-striation guidance for micromixers.
4. Sparse quantitative CFD-vs-experiment (Villermaux-Dushman / PLIF) error bars for
   these specific geometries.

## Implications for OUR code (torChannel + merge)

- **Our methodology already matches the literature's central demand.** The recommended
  false-diffusion diagnostic — run with D=0, check D_effective = D_molecular +
  D_numerical — is exactly the spirit of our verification: the merge cross-section
  marcher has **zero streamwise numerical diffusion by construction**, and the
  torChannel scalar passed an erf-diffusion test with **D_effective/D − 1 = −0.004%**.
  That is a quantitative false-diffusion certificate of the kind most micromixer papers
  omit.
- **Structured grids are the right call.** Hex/structured beats tet by 4-5 orders of
  magnitude on false diffusion — our staggered-FD (torChannel) and structured-FV
  (merge) choices are well-aligned; avoid unstructured tets for the scalar.
- **Report cell-Pe.** For every production run we should report the cell-Péclet number
  (U·Δx/D) and keep it low (≤ ~2 ideal), or otherwise quantify D_numerical, rather than
  trusting a mixing-index grid-independence check.
- **Sc~1000 is genuinely the cost driver, and full 3D DNS resolution at real Sc is an
  open niche** — consistent with our earlier analysis. Plan to either resolve it on GPU/
  HPC or reduce Sc deliberately and document it.
- The torChannel scalar advection is currently **2nd-order central** (verified
  non-dissipative on smooth fields). At high Pe with sharp interfaces, consider a TVD/
  flux-limited option for the scalar to suppress dispersive wiggles without adding
  false diffusion.

## Sources (all primary; peer-reviewed unless noted)

- Bayareh 2021, *Proc IMechE Part C* — review of artificial diffusion in liquid
  micromixing. doi:10.1177/0954406220982028
- Okuducu & Aral 2018, *Micromachines* — false diffusion vs mesh type in T-mixers.
  PMC6187341
- Hadjigeorgiou, Boudouvis & Kokkoris 2021, *Chem Eng J* 414:128775 — SHM, high-Pe
  stabilization. doi:10.1016/j.cej.2021.128775
- Hadjigeorgiou & Kokkoris 2019, *Processes* 7:121 — false diffusion / cell-Pe
  thresholds. doi:10.3390/pr7030121
- Liu & Lin 2011, *Chem Eng Sci* 66:2211 — quantifying false diffusion on scalar decay.
- Falk & Commenge 2011, *Chem Eng Process* 50:979 — Villermaux-Dushman, mixing-time vs
  segregation index.
- Tian et al. 2019, *Microgravity Sci Technol* 31:833 — bilateral Koch fractal baffles.
- Koch-snowflake-baffle study, *ScienceDirect* S2590123025046250 (2025).
- COMSOL micromixer study, *Micromachines* 17(5):525.
- arXiv:2402.07854 (FEM micromixer; PREPRINT, not peer-reviewed — treat with care).

### Lower-confidence / refuted (do not over-rely)
- The exact "tolerable" cell-Pe ~50-100 is device-specific (swirl mixers, Re 120-240),
  not universal.
- A claimed monotonic mixing-index drop 94.09%→62.79% across one arXiv mesh sweep
  FAILED verification (1-2) — the false-diffusion trend is real but that specific
  trajectory is not reliably supported; do not cite it.
- Absolute Koch-baffle efficiencies (95%, 99.70%) are un-validated single-group CFD.
