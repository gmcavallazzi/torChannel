"""
Test the double-stretched canopy grid (utils.generate_double_stretched_grid).

Checks:
1. Return contract matches generate_grid/generate_hybrid_grid (lengths, ghosts)
2. Monotonicity, positivity, exact span [0, Lz], face exactly at z_transition
3. Clustering at bed AND canopy tip (dz minima at both shear locations)
4. C1 continuity across the transition (< 1%) for the tuned gamma pair
5. Wall-unit spacings at the Monti et al. (2022) target Re_tau,out ~ 1157:
   dz_tip+ in the paper's range (0.24-0.44), dz_bed+ well below 1
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from utils import generate_double_stretched_grid

torch.set_default_dtype(torch.float64)

# Monti-like parameters (tuned pair: gamma_canopy=2.2, gamma_outer=2.5)
nz_canopy = 100
nz_outer = 200
h = 0.25
Lz = 1.0
gamma_canopy = 2.2
gamma_outer = 'auto'  # solve for C1 continuity at the transition
Re_tau_out = 1157.0  # z+ = z * Re_tau_out for Lz = 1

z_f, z_c, dz_f, dz_c = generate_double_stretched_grid(
    nz_canopy, nz_outer, h, Lz, gamma_canopy, gamma_outer)

nz = nz_canopy + nz_outer
failures = []

def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name} {detail}")
    if not cond:
        failures.append(name)

print("\n1. Return contract")
check("z_f length", len(z_f) == nz + 1, f"({len(z_f)} == {nz+1})")
check("z_c length (with ghosts)", len(z_c) == nz + 2, f"({len(z_c)} == {nz+2})")
check("dz_f length", len(dz_f) == nz, f"({len(dz_f)} == {nz})")
check("dz_c length", len(dz_c) == nz + 1, f"({len(dz_c)} == {nz+1})")
check("ghost center below wall", z_c[0].item() == -z_c[1].item(),
      f"(z_c[0]={z_c[0].item():.3e})")

print("\n2. Span and monotonicity")
check("z_f[0] == 0", abs(z_f[0].item()) < 1e-14)
check("z_f[-1] == Lz", abs(z_f[-1].item() - Lz) < 1e-12)
check("face at transition", abs(z_f[nz_canopy].item() - h) < 1e-12,
      f"(z_f[{nz_canopy}]={z_f[nz_canopy].item():.15f})")
check("strictly monotonic", bool(torch.all(z_f[1:] > z_f[:-1])))
check("positive spacings", bool(torch.all(dz_f > 0)))

print("\n3. Clustering at bed and tip")
dz_bed = dz_f[0].item()
dz_tip_below = dz_f[nz_canopy - 1].item()
dz_tip_above = dz_f[nz_canopy].item()
dz_mid_canopy = dz_f[nz_canopy // 2].item()
dz_top = dz_f[-1].item()
check("dz_bed < dz_mid_canopy / 3", dz_bed < dz_mid_canopy / 3,
      f"(bed {dz_bed:.3e} vs mid {dz_mid_canopy:.3e})")
check("dz_tip < dz_mid_canopy / 3", dz_tip_below < dz_mid_canopy / 3,
      f"(tip {dz_tip_below:.3e})")
check("dz_tip < dz_top (outer one-sided)", dz_tip_above < dz_top,
      f"(tip {dz_tip_above:.3e} vs top {dz_top:.3e})")

print("\n4. C1 continuity at transition")
jump = abs(dz_tip_above - dz_tip_below) / dz_tip_below
check("C1 jump < 1%", jump < 0.01, f"({jump*100:.3f}%)")

print("\n5. Wall units at Re_tau,out = 1157")
dz_bed_plus = dz_bed * Re_tau_out
dz_tip_plus = dz_tip_below * Re_tau_out
dz_max_plus = dz_f.max().item() * Re_tau_out
print(f"  dz_bed+ = {dz_bed_plus:.3f}, dz_tip+ = {dz_tip_plus:.3f}, dz_max+ = {dz_max_plus:.2f}")
check("dz_tip+ in [0.2, 0.5] (paper: 0.24-0.44)", 0.2 <= dz_tip_plus <= 0.5)
check("dz_bed+ < 1", dz_bed_plus < 1.0)
check("dz_max+ < 15", dz_max_plus < 15.0)

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("All double-stretched grid checks passed.")
