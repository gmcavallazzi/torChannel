# Re_τ = 180 open channel — validation result

Produced by `examples/re180_open/config_continue.yaml` (see that directory for
how to reproduce, including the CaNS seed).

| | |
|---|---|
| Re_b | 2792.8 — **matched to Moser, Kim & Mansour (1999) chan180** |
| Re_τ achieved | **179.69 ± 1.55** (MKM: 178.12) |
| samples | 2201, over 200 t.u. (~13 eddy turnovers) after a 50 t.u. transient |
| precision | `float64` |
| grid | 192 × 192 × 128 on 4π × 2π × 1 |
| resolution | Δz⁺ = 0.37, Δx⁺ = 11.76, Δy⁺ = 5.88 (at the achieved Re_τ) |

## Agreement with MKM chan180

| quantity | torChannel | MKM | diff |
|---|---|---|---|
| Re_τ | 179.69 | 178.12 | +0.9 % |
| peak ⟨u'u'⟩⁺ | 6.801 (z⁺=14.0) | 7.066 (z⁺=15.3) | −3.7 % |
| peak −⟨u'w'⟩⁺ | 0.729 | 0.723 | +0.9 % |
| U⁺, 30 < z⁺ < 100 | — | — | 0.87 % mean, 1.19 % max |

**Total-stress balance** (independent of any reference — an exact consequence of
the mean momentum equation): −⟨u'w'⟩⁺ + ν dU⁺/dz⁺ = 1 − z/δ to **max 0.0089,
RMS 0.0056**.

## Reading the comparison

- Re_b is matched; Re_τ is an *outcome*, and lands within 0.9 %.
- MKM is a **closed** channel, this is an **open** channel. Near-wall agreement
  is the meaningful test; the departure above z⁺ ≈ 130 is physical — a free-slip
  top suppresses the motions that cross a closed channel's centreline.
- Each dataset is scaled by **its own** u_τ, as is standard. They therefore sit
  at slightly different z/δ for a given z⁺.
- The spanwise box differs (2π vs 4π/3). Immaterial below Re_τ ≈ 1000, where
  there is no scale separation; resolution is the criterion that matters, and it
  is met.
- The −3.7 % on the u'u' peak is the largest discrepancy and the noisiest
  quantity here; it read 6.74–6.95 at intermediate sample counts.
