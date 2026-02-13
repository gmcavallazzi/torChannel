"""
Find optimal gamma for hybrid grid with minimal C1 discontinuity.

Searches for gamma value that minimizes spacing jump at transition.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import generate_hybrid_grid

print("=" * 80)
print("FINDING OPTIMAL GAMMA FOR HYBRID GRID")
print("=" * 80)

# Grid parameters
nz_uniform = 75
nz_stretched = 75
z_transition = 0.25
Lz = 1.25

# Search range for gamma
gamma_values = np.linspace(0.5, 3.5, 31)
discontinuities = []
dz_before_list = []
dz_after_list = []

print(f"\nSearching gamma ∈ [{gamma_values[0]:.2f}, {gamma_values[-1]:.2f}]...")
print(f"Testing {len(gamma_values)} values...\n")

for gamma in gamma_values:
    try:
        z_f, z_c, dz_f, dz_c = generate_hybrid_grid(
            nz_uniform, nz_stretched, z_transition, Lz, gamma, device='cpu'
        )

        # Check continuity at transition
        idx_transition = nz_uniform
        dz_before = dz_f[idx_transition - 1].item()
        dz_after = dz_f[idx_transition].item()
        discontinuity = abs(dz_after - dz_before) / dz_before

        discontinuities.append(discontinuity)
        dz_before_list.append(dz_before)
        dz_after_list.append(dz_after)

    except Exception as e:
        print(f"  γ = {gamma:.3f}: ERROR - {e}")
        discontinuities.append(np.nan)
        dz_before_list.append(np.nan)
        dz_after_list.append(np.nan)

# Find optimal gamma
discontinuities = np.array(discontinuities)
valid_mask = ~np.isnan(discontinuities)
valid_gammas = gamma_values[valid_mask]
valid_disc = discontinuities[valid_mask]

if len(valid_disc) > 0:
    idx_best = np.argmin(valid_disc)
    gamma_optimal = valid_gammas[idx_best]
    disc_optimal = valid_disc[idx_best]

    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nOptimal gamma: {gamma_optimal:.4f}")
    print(f"Discontinuity: {disc_optimal*100:.3f}%")
    print(f"dz before: {dz_before_list[np.where(gamma_values == gamma_optimal)[0][0]]:.6e}")
    print(f"dz after:  {dz_after_list[np.where(gamma_values == gamma_optimal)[0][0]]:.6e}")

    # Show top 5 candidates
    print(f"\nTop 5 candidates:")
    sorted_indices = np.argsort(valid_disc)[:5]
    for rank, idx in enumerate(sorted_indices, 1):
        g = valid_gammas[idx]
        d = valid_disc[idx]
        print(f"  {rank}. γ = {g:.4f}, discontinuity = {d*100:.3f}%")

    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Discontinuity vs gamma
    ax = axes[0]
    ax.plot(valid_gammas, valid_disc * 100, 'o-', linewidth=2, markersize=6)
    ax.axvline(gamma_optimal, color='r', linestyle='--', linewidth=2,
               label=f'Optimal: γ={gamma_optimal:.4f}')
    ax.axhline(1.0, color='orange', linestyle=':', linewidth=1.5,
               label='1% threshold')
    ax.set_xlabel('Stretching parameter γ', fontsize=12)
    ax.set_ylabel('Relative discontinuity (%)', fontsize=12)
    ax.set_title('C1 Continuity vs Gamma', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 2: Cell spacings at transition
    ax = axes[1]
    ax.plot(valid_gammas, np.array(dz_before_list)[valid_mask] * 1000, 'o-',
            label='dz before transition', linewidth=2, markersize=5)
    ax.plot(valid_gammas, np.array(dz_after_list)[valid_mask] * 1000, 's-',
            label='dz after transition', linewidth=2, markersize=5)
    ax.axvline(gamma_optimal, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Stretching parameter γ', fontsize=12)
    ax.set_ylabel('Cell spacing (mm)', fontsize=12)
    ax.set_title('Cell Spacing at Transition vs Gamma', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('optimal_gamma_search.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: optimal_gamma_search.png")

    # Test optimal gamma
    print(f"\nTesting optimal gamma...")
    z_f, z_c, dz_f, dz_c = generate_hybrid_grid(
        nz_uniform, nz_stretched, z_transition, Lz, gamma_optimal, device='cpu'
    )

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print(f"\nUpdate config_canopy_monti.yaml:")
    print(f"  gamma_stretched: {gamma_optimal:.4f}  # Optimized for C1 continuity")

else:
    print("ERROR: No valid gamma values found!")

print("\n" + "=" * 80)
