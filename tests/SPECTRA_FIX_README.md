# 2D Premultiplied Spectra Fix

## Problem Description

When averaging velocity fluctuations at z⁺=15 from both the bottom and top walls, the current implementation produces **spurious double peaks** in the 2D premultiplied spectra. This is incorrect - there should only be **one peak**.

## Root Cause

In channel flow, the top and bottom walls are **mirror images** of each other with respect to the channel centerline. When extracting statistics from both walls:

- **Bottom wall (z=0)**: Flow structures oriented normally
- **Top wall (z=Lz)**: Flow structures are **mirrored** in the spanwise (y) direction

The current code directly averages the two planes without accounting for this reflection symmetry:

```python
# WRONG: Direct averaging causes destructive interference
u_plane = 0.5 * (u_bot + u_top)  # ❌
```

This creates two symmetric peaks in wavenumber space that cancel out the actual flow structures.

## Solution

Before averaging, we must **flip the top wall data** in the spanwise (y) direction to align it with the bottom wall:

```python
# CORRECT: Flip top wall data to account for mirror symmetry
u_top = torch.flip(u_top, dims=[1])      # Flip u in y
v_top = -torch.flip(v_top, dims=[1])     # Flip v in y AND change sign
w_top = torch.flip(w_top, dims=[1])      # Flip w in y

u_plane = 0.5 * (u_bot + u_top)  # ✓ Now properly aligned
```

**Why different treatment for v?**
- `u` (streamwise): Symmetric under y-flip
- `v` (spanwise): **Anti-symmetric** under y-flip (reverses direction)
- `w` (wall-normal): Symmetric under y-flip

## Testing Instructions

### 1. Run the test script

```bash
cd /path/to/DNS_homemade
python test_spectra_fix.py results/fields.npz --output spectra_comparison.png
```

This will:
- Load your field file
- Compute 2D spectra using **both methods** (original and fixed)
- Generate a comparison plot showing 8 panels (4 components × 2 methods)

### 2. Expected Results

**Original method (top row):**
- Should show **TWO symmetric peaks** (one on each side)
- This is the spurious artifact

**Fixed method (bottom row):**
- Should show **ONE central peak**
- This is the physically correct result

### 3. Visual Inspection

Look for:
- ✅ **Single peak** in fixed version → Fix is working correctly
- ❌ **Double peaks** in fixed version → Something is wrong

## Applying the Fix to statistics.py

If the test confirms the fix works, apply it to `statistics.py` by replacing lines 251-259:

**Before:**
```python
# Top wall plane
u_top = u_fluct[:, :, self.k_top - 1]
v_top = v_fluct[:, :, self.k_top - 1]
w_top = w_fluct[:, :, self.k_top - 1]

# Average between walls
u_plane = 0.5 * (u_bot + u_top)
v_plane = 0.5 * (v_bot + v_top)
w_plane = 0.5 * (w_bot + w_top)
```

**After:**
```python
# Top wall plane
u_top = u_fluct[:, :, self.k_top - 1]
v_top = v_fluct[:, :, self.k_top - 1]
w_top = w_fluct[:, :, self.k_top - 1]

# Flip top wall data in y-direction to account for channel flow symmetry
# (top and bottom walls are mirror images about the centerline)
u_top = torch.flip(u_top, dims=[1])      # u: symmetric under y-flip
v_top = -torch.flip(v_top, dims=[1])     # v: anti-symmetric (changes sign)
w_top = torch.flip(w_top, dims=[1])      # w: symmetric under y-flip

# Average between walls (now properly aligned)
u_plane = 0.5 * (u_bot + u_top)
v_plane = 0.5 * (v_bot + v_top)
w_plane = 0.5 * (w_bot + w_top)
```

## Physical Interpretation

The fix ensures that:
1. Coherent structures (streaks, vortices) near both walls are properly aligned in the average
2. The resulting spectrum represents the **true** energy distribution at z⁺=15
3. Results are comparable to published DNS data (e.g., Moser, Kim & Mansour 1999)

## References

- Moser, Kim & Mansour (1999), "Direct numerical simulation of turbulent channel flow up to Re_τ=590"
- Del Álamo & Jiménez (2003), "Spectra of the very large anisotropic scales in turbulent channels"

The correct treatment of wall symmetry is standard in channel flow DNS literature.
