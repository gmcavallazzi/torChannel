#!/usr/bin/env python3
"""
Script to compute gamma parameter from grid.csv file.

The grid is generated with symmetric stretching from both walls.
This script analyzes the grid spacings to back-compute gamma.
"""

import numpy as np
import sys

def compute_gamma_from_grid(csv_file='results/grid.csv'):
    """
    Read grid.csv and compute the gamma stretching parameter.

    The stretching formula (from generate_grid) is:
    - Symmetric stretching from both walls
    - dz increases from wall to center following gamma
    """

    # Read CSV file
    # Format: z_f, z_c, dz_f, dz_c (with header)
    # NaNs are used for padding (different array sizes)
    # z_f has nz+1 points (face positions), last value is NaN
    # z_c, dz_f, dz_c have nz points, then NaN padding
    try:
        data = np.genfromtxt(csv_file, delimiter=',', skip_header=1)
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found")
        sys.exit(1)

    z_f_raw = data[:, 0]
    z_c_raw = data[:, 1]
    dz_f_raw = data[:, 2]
    dz_c_raw = data[:, 3]

    # Remove NaN values
    # z_f: has nz+1 values, remove all NaNs
    z_f = z_f_raw[~np.isnan(z_f_raw)]

    # dz_f: has nz values, remove all NaNs
    dz_f = dz_f_raw[~np.isnan(dz_f_raw)]

    nz = len(dz_f)
    # Lz is the last face position (z_f has nz+1 points: 0 to nz)
    Lz = z_f[-1] if len(z_f) > 0 else np.nan

    print(f"Grid info:")
    print(f"  nz = {nz}")
    print(f"  Lz = {Lz:.6f}")
    print(f"  dz_min = {dz_f.min():.6e}")
    print(f"  dz_max = {dz_f.max():.6e}")
    print(f"  Ratio (max/min) = {dz_f.max()/dz_f.min():.3f}")

    # EXACT INVERSE OF GENERATION FORMULA
    # The grid is generated using:
    #   k = linspace(0, nz, nz+1)
    #   xi = (2*k/nz) - 1
    #   z_f = 0.5 * Lz * (1 + tanh(gamma*xi) / tanh(gamma))
    #
    # To invert this:
    #   z_f / (0.5*Lz) = 1 + tanh(gamma*xi) / tanh(gamma)
    #   tanh(gamma*xi) = tanh(gamma) * (2*z_f/Lz - 1)
    #   gamma*xi = atanh(tanh(gamma) * (2*z_f/Lz - 1))
    #
    # We know xi at each point, so we can solve for gamma

    # Select a few points away from boundaries for robust fitting
    # (avoid boundary effects and numerical issues near tanh asymptotes)
    n_sample = min(10, nz // 4)  # Use first 10 points (or nz/4)

    gamma_estimates = []

    for i in range(1, n_sample + 1):
        # Compute xi for this grid point
        # k[i] = i, so xi[i] = (2*i/nz) - 1
        xi_i = (2.0 * i / nz) - 1.0

        # z_f[i] is known
        z_i = z_f[i]

        # Normalize: (2*z_i/Lz - 1) = tanh(gamma*xi_i) / tanh(gamma)
        z_norm = 2.0 * z_i / Lz - 1.0

        # For small |z_norm|, we have:
        # z_norm ≈ tanh(gamma*xi_i) / tanh(gamma)
        #
        # Use numerical solver to find gamma
        # Define: f(gamma) = tanh(gamma*xi_i)/tanh(gamma) - z_norm = 0

        from scipy.optimize import brentq

        def func(gamma_test):
            if abs(gamma_test) < 1e-10:
                return xi_i - z_norm  # Limit as gamma->0
            return np.tanh(gamma_test * xi_i) / np.tanh(gamma_test) - z_norm

        try:
            # Search for gamma in range [0.01, 5.0]
            gamma_i = brentq(func, 0.01, 5.0)
            gamma_estimates.append(gamma_i)
        except ValueError:
            # If brentq fails (no sign change), skip this point
            pass

    gamma_estimates = np.array(gamma_estimates)

    # Best estimate: mean of all point estimates
    gamma_best = np.mean(gamma_estimates)
    gamma_std = np.std(gamma_estimates)

    print(f"\nEXACT INVERSE CALCULATION:")
    print(f"  Using {len(gamma_estimates)} grid points near wall")
    print(f"  Gamma estimates range: [{gamma_estimates.min():.6f}, {gamma_estimates.max():.6f}]")
    print(f"  Mean: {gamma_best:.6f}")
    print(f"  Std dev: {gamma_std:.6e}")

    print(f"\n{'='*60}")
    print(f"EXACT GAMMA: {gamma_best:.6f} ± {gamma_std:.6e}")
    print(f"{'='*60}")

    # Verification: regenerate grid with computed gamma and compare
    k_verify = np.linspace(0, nz, nz+1)
    xi_verify = (2 * k_verify / nz) - 1
    z_f_reconstructed = 0.5 * Lz * (1 + np.tanh(gamma_best * xi_verify) / np.tanh(gamma_best))

    # Compare to original
    max_error = np.max(np.abs(z_f - z_f_reconstructed))
    print(f"\nVerification:")
    print(f"  Max reconstruction error: {max_error:.6e}")
    print(f"  (This should be very small if gamma is correct)")

    return gamma_best


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compute gamma from grid.csv')
    parser.add_argument('csv_file', type=str,
                        help='Path to grid.csv file')

    args = parser.parse_args()

    gamma = compute_gamma_from_grid(args.csv_file)
