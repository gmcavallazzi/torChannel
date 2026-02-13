"""
Test script for hybrid grid generation.

Verifies:
1. Grid monotonicity
2. C1 continuity at transition
3. Grid visualization
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from utils import generate_hybrid_grid

print("=" * 80)
print("HYBRID GRID GENERATION TEST")
print("=" * 80)

# Test parameters (from config_canopy_monti.yaml)
nz_uniform = 75
nz_stretched = 75
z_transition = 0.25
Lz = 1.25
gamma = 1.8

print(f"\nParameters:")
print(f"  Uniform region: nz={nz_uniform}, z ∈ [0, {z_transition}]")
print(f"  Stretched region: nz={nz_stretched}, z ∈ [{z_transition}, {Lz}], gamma={gamma}")

# Generate grid
print(f"\nGenerating hybrid grid...")
z_f, z_c, dz_f, dz_c = generate_hybrid_grid(
    nz_uniform, nz_stretched, z_transition, Lz, gamma, device='cpu'
)

# Convert to numpy for analysis
z_f_np = z_f.cpu().numpy()
z_c_np = z_c.cpu().numpy()
dz_f_np = dz_f.cpu().numpy()
dz_c_np = dz_c.cpu().numpy()

print(f"\nGrid statistics:")
print(f"  Total cells: {len(dz_f)}")
print(f"  Face coordinates: {len(z_f)} points")
print(f"  Cell centers: {len(z_c)} points (includes 2 ghost cells)")

# Check monotonicity
is_monotonic = np.all(z_f_np[1:] > z_f_np[:-1])
print(f"\nMonotonicity check: {'PASS' if is_monotonic else 'FAIL'}")

# Check transition continuity
idx_transition = nz_uniform
dz_before = dz_f_np[idx_transition - 1]
dz_after = dz_f_np[idx_transition]
discontinuity = abs(dz_after - dz_before) / dz_before

print(f"\nC1 continuity at z={z_transition}:")
print(f"  Cell spacing before transition: {dz_before:.6e}")
print(f"  Cell spacing after transition:  {dz_after:.6e}")
print(f"  Relative discontinuity: {discontinuity*100:.3f}%")
print(f"  Status: {'PASS' if discontinuity < 0.01 else 'WARNING (>1%)'}")

# Grid spacing statistics
print(f"\nCell spacing statistics:")
print(f"  Uniform region:")
print(f"    dz_min = {dz_f_np[:nz_uniform].min():.6e}")
print(f"    dz_max = {dz_f_np[:nz_uniform].max():.6e}")
print(f"    dz_avg = {dz_f_np[:nz_uniform].mean():.6e}")
print(f"  Stretched region:")
print(f"    dz_min = {dz_f_np[nz_uniform:].min():.6e}")
print(f"    dz_max = {dz_f_np[nz_uniform:].max():.6e}")
print(f"    dz_avg = {dz_f_np[nz_uniform:].mean():.6e}")
print(f"  Stretching ratio (max/min): {dz_f_np.max() / dz_f_np.min():.2f}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Grid coordinates
ax = axes[0, 0]
ax.plot(range(len(z_f)), z_f_np, 'o-', markersize=2, label='Face coordinates')
ax.axhline(z_transition, color='r', linestyle='--', linewidth=2, label=f'Transition (z={z_transition})')
ax.axvline(nz_uniform, color='orange', linestyle=':', linewidth=2, label=f'Index={nz_uniform}')
ax.set_xlabel('Grid index', fontsize=11)
ax.set_ylabel('z coordinate', fontsize=11)
ax.set_title('Grid Face Coordinates', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Cell spacing
ax = axes[0, 1]
ax.plot(range(len(dz_f)), dz_f_np, 'o-', markersize=3, color='blue')
ax.axvline(nz_uniform, color='r', linestyle='--', linewidth=2, label=f'Transition (cell {nz_uniform})')
ax.set_xlabel('Cell index', fontsize=11)
ax.set_ylabel('Cell height (dz)', fontsize=11)
ax.set_title('Cell Spacing Distribution', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Add annotation for discontinuity
ax.annotate(f'Δ = {discontinuity*100:.2f}%',
           xy=(nz_uniform, dz_after),
           xytext=(nz_uniform + 10, dz_after + 0.002),
           arrowprops=dict(arrowstyle='->', color='red'),
           fontsize=10, color='red')

# Plot 3: Zoomed view around transition
ax = axes[1, 0]
zoom_range = 10  # cells on each side
idx_start = max(0, nz_uniform - zoom_range)
idx_end = min(len(dz_f), nz_uniform + zoom_range)
indices = range(idx_start, idx_end)
ax.plot(indices, dz_f_np[idx_start:idx_end], 'o-', markersize=5, linewidth=2)
ax.axvline(nz_uniform, color='r', linestyle='--', linewidth=2, label='Transition')
ax.set_xlabel('Cell index', fontsize=11)
ax.set_ylabel('Cell height (dz)', fontsize=11)
ax.set_title(f'Zoomed View: Transition Zone (±{zoom_range} cells)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Height vs spacing
ax = axes[1, 1]
z_cell_centers = 0.5 * (z_f_np[:-1] + z_f_np[1:])
ax.plot(dz_f_np, z_cell_centers, 'o-', markersize=2)
ax.axhline(z_transition, color='r', linestyle='--', linewidth=2, label=f'z={z_transition}')
ax.set_xlabel('Cell height (dz)', fontsize=11)
ax.set_ylabel('z coordinate', fontsize=11)
ax.set_title('Spacing vs Height', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hybrid_grid_test.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved: hybrid_grid_test.png")

plt.show()

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
