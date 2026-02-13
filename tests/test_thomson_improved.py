#!/usr/bin/env python3
"""
Improved Thomson Problem solver with:
- Better optimization (FIRE algorithm)
- Tangential projection (forces act only on sphere surface)
- Multiple restarts to avoid local minima
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform


def random_sphere_points(N, radius=1.0):
    """Generate N random points uniformly distributed on sphere surface."""
    theta = np.random.uniform(0, 2*np.pi, N)
    phi = np.arccos(np.random.uniform(-1, 1, N))

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    return np.column_stack([x, y, z])


def project_to_sphere(points, radius=1.0):
    """Project points onto sphere surface."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return radius * points / norms


def compute_forces_tangential(points, radius=1.0, power=2):
    """
    Compute repulsive forces projected onto tangent plane of sphere.
    This is more efficient than full 3D forces + reprojection.

    Parameters:
    -----------
    power : float
        Force law exponent. F ~ 1/r^power
        power=1: Coulomb in 2D (on sphere surface)
        power=2: Coulomb in 3D (default)
    """
    N = len(points)
    forces = np.zeros_like(points)

    for i in range(N):
        # Vectors from point i to all others
        diff = points - points[i]
        dist = np.linalg.norm(diff, axis=1, keepdims=True)

        # Avoid self-interaction
        dist[i] = np.inf

        # Repulsive force magnitude
        force_mag = 1.0 / (dist**power + 1e-12)
        force_vec = force_mag * (-diff / (dist + 1e-12))

        # Sum forces
        total_force = np.sum(force_vec, axis=0)

        # Project force to tangent plane (perpendicular to radial direction)
        radial = points[i] / radius
        tangential_force = total_force - np.dot(total_force, radial) * radial

        forces[i] = tangential_force

    return forces


def thomson_FIRE(N, radius=1.0, n_iterations=5000, dt_start=0.01,
                 tolerance=1e-8, verbose=True, power=2):
    """
    Thomson problem using FIRE (Fast Inertial Relaxation Engine) algorithm.

    FIRE is a molecular dynamics optimizer that adapts the time step and
    includes inertia for faster convergence.

    Parameters:
    -----------
    power : float
        Exponent in force law (1/r^power). Default=2 for Coulomb 3D.
    """
    # Initialize
    points = random_sphere_points(N, radius)
    velocity = np.zeros_like(points)

    # FIRE parameters
    dt = dt_start
    dt_max = 10 * dt_start
    N_min = 5  # minimum steps before increasing dt
    f_inc = 1.1  # factor to increase dt
    f_dec = 0.5  # factor to decrease dt
    alpha_start = 0.1
    f_alpha = 0.99

    alpha = alpha_start
    N_pos = 0  # number of steps with P > 0

    history = {
        'max_force': [],
        'mean_force': [],
        'energy': [],
        'min_distance': [],
        'cv_nn': []
    }

    for iteration in range(n_iterations):
        # Compute forces
        forces = compute_forces_tangential(points, radius, power)

        # FIRE algorithm
        P = np.sum(forces * velocity)  # Power

        if P > 0:
            N_pos += 1
            if N_pos > N_min:
                dt = min(dt * f_inc, dt_max)
                alpha = alpha * f_alpha
        else:
            N_pos = 0
            dt = dt * f_dec
            velocity = np.zeros_like(velocity)
            alpha = alpha_start

        # Update velocity
        force_norm = np.linalg.norm(forces)
        vel_norm = np.linalg.norm(velocity)

        if force_norm > 0 and vel_norm > 0:
            velocity = (1 - alpha) * velocity + alpha * (vel_norm / force_norm) * forces
        else:
            velocity = forces * alpha

        # Update positions
        velocity = velocity + dt * forces
        points = points + dt * velocity

        # Project to sphere
        points = project_to_sphere(points, radius)

        # Metrics
        force_magnitudes = np.linalg.norm(forces, axis=1)
        max_force = np.max(force_magnitudes)
        mean_force = np.mean(force_magnitudes)

        dists = pdist(points)
        energy = np.sum(1.0 / (dists**power + 1e-12))

        dist_matrix = squareform(dists)
        np.fill_diagonal(dist_matrix, np.inf)
        nn_dists = np.min(dist_matrix, axis=1)
        cv_nn = np.std(nn_dists) / np.mean(nn_dists)

        history['max_force'].append(max_force)
        history['mean_force'].append(mean_force)
        history['energy'].append(energy)
        history['min_distance'].append(np.min(nn_dists))
        history['cv_nn'].append(cv_nn)

        if verbose and (iteration % 100 == 0 or iteration < 10):
            print(f"Iter {iteration:4d}: max_F={max_force:.2e}, "
                  f"E={energy:.2f}, CV_nn={cv_nn:.4f}, "
                  f"dt={dt:.2e}, α={alpha:.3f}")

        if max_force < tolerance:
            if verbose:
                print(f"\n✓ Converged at iteration {iteration}")
            break

    return points, history


def multi_restart_thomson(N, radius=1.0, n_restarts=3, **kwargs):
    """
    Run Thomson optimization multiple times and return best result.
    Helps avoid local minima.
    """
    print(f"\nRunning {n_restarts} independent optimizations...\n")

    best_points = None
    best_cv = np.inf
    best_energy = np.inf

    all_results = []

    for restart in range(n_restarts):
        print(f"{'='*60}")
        print(f"RESTART {restart + 1}/{n_restarts}")
        print(f"{'='*60}")

        points, history = thomson_FIRE(N, radius, verbose=True, **kwargs)

        # Evaluate quality
        dists = pdist(points)
        dist_matrix = squareform(dists)
        np.fill_diagonal(dist_matrix, np.inf)
        nn_dists = np.min(dist_matrix, axis=1)
        cv_nn = np.std(nn_dists) / np.mean(nn_dists)
        energy = history['energy'][-1]

        all_results.append({
            'points': points,
            'history': history,
            'cv_nn': cv_nn,
            'energy': energy
        })

        print(f"\nResult: CV_nn={cv_nn:.6f}, Energy={energy:.4f}")

        if cv_nn < best_cv:
            best_cv = cv_nn
            best_energy = energy
            best_points = points
            best_history = history
            print("★ NEW BEST ★")

    print(f"\n{'='*60}")
    print(f"BEST RESULT: CV_nn={best_cv:.6f}, Energy={best_energy:.4f}")
    print(f"{'='*60}\n")

    return best_points, best_history, all_results


def analyze_distribution(points, radius=1.0):
    """Analyze point distribution quality."""
    N = len(points)
    dists = pdist(points)
    theoretical_spacing = np.sqrt(4 * np.pi * radius**2 / N)

    dist_matrix = squareform(dists)
    np.fill_diagonal(dist_matrix, np.inf)
    nearest_neighbor = np.min(dist_matrix, axis=1)

    print("\n" + "="*60)
    print("DISTRIBUTION ANALYSIS")
    print("="*60)
    print(f"Number of points: {N}")
    print(f"Sphere radius: {radius:.4f}")
    print(f"\nTheoretical optimal spacing: {theoretical_spacing:.6f}")
    print(f"\nNearest neighbor distances:")
    print(f"  Min:  {np.min(nearest_neighbor):.6f}")
    print(f"  Mean: {np.mean(nearest_neighbor):.6f}")
    print(f"  Max:  {np.max(nearest_neighbor):.6f}")
    print(f"  Std:  {np.std(nearest_neighbor):.6f}")
    print(f"  CV (coefficient of variation): {np.std(nearest_neighbor)/np.mean(nearest_neighbor):.6f}")
    print(f"\nUniformity metrics:")
    print(f"  Mean NN / theoretical: {np.mean(nearest_neighbor)/theoretical_spacing:.6f}")
    print(f"  (Min NN) / (Max NN): {np.min(nearest_neighbor)/np.max(nearest_neighbor):.6f}")
    print(f"  CV (lower is better): {np.std(nearest_neighbor)/np.mean(nearest_neighbor):.6f}")
    print("="*60 + "\n")

    return {
        'theoretical_spacing': theoretical_spacing,
        'nearest_neighbor': nearest_neighbor,
        'cv_nn': np.std(nearest_neighbor)/np.mean(nearest_neighbor)
    }


def plot_results(points, history, stats, radius=1.0):
    """Visualize results."""
    fig = plt.figure(figsize=(18, 5))

    # 1. Points on sphere (colored by height/z-coordinate)
    ax1 = fig.add_subplot(131, projection='3d')
    colors = points[:, 2]  # Use z-coordinate for coloring
    scatter = ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                c=colors, s=50, alpha=0.8, edgecolors='black', linewidth=0.5,
                cmap='viridis', vmin=-radius, vmax=radius)
    plt.colorbar(scatter, ax=ax1, label='Height (z)', shrink=0.5, pad=0.1)

    # Wireframe
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x_sphere = radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_wireframe(x_sphere, y_sphere, z_sphere,
                       color='lightblue', alpha=0.15, linewidth=0.5)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'Thomson Distribution (N={len(points)}, CV={stats["cv_nn"]:.4f})')
    ax1.set_box_aspect([1,1,1])

    # 2. Convergence
    ax2 = fig.add_subplot(132)
    iters = range(len(history['cv_nn']))
    ax2.plot(iters, history['cv_nn'], 'b-', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('CV of Nearest Neighbor Distances')
    ax2.set_title('Uniformity Convergence (lower = more uniform)')
    ax2.grid(True, alpha=0.3)

    # 3. Distance histogram
    ax3 = fig.add_subplot(133)
    nn_distances = stats['nearest_neighbor']
    ax3.hist(nn_distances, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
    ax3.axvline(stats['theoretical_spacing'], color='red', linestyle='--',
                linewidth=2, label=f'Theoretical: {stats["theoretical_spacing"]:.4f}')
    ax3.axvline(np.mean(nn_distances), color='green', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(nn_distances):.4f}')
    ax3.set_xlabel('Nearest Neighbor Distance')
    ax3.set_ylabel('Count')
    ax3.set_title(f'NN Distance Distribution (CV={stats["cv_nn"]:.4f})')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('thomson_improved.png', dpi=150, bbox_inches='tight')
    print(f"Plot saved: thomson_improved.png")
    plt.show()


def main():
    """Main test function."""

    # CONFIGURATION
    N = 100  # Number of points
    radius = 1.0
    n_iterations = 3000
    n_restarts = 3  # Try multiple random starts
    power = 2  # Force law: 1/r^power (2=Coulomb 3D, 1=Coulomb 2D)

    print("="*60)
    print("IMPROVED THOMSON PROBLEM - FIRE ALGORITHM")
    print("="*60)
    print(f"N points: {N}")
    print(f"Radius: {radius}")
    print(f"Force law: 1/r^{power}")
    print(f"Restarts: {n_restarts}")
    print("="*60)

    # Run with multiple restarts
    points, history, all_results = multi_restart_thomson(
        N=N,
        radius=radius,
        n_iterations=n_iterations,
        n_restarts=n_restarts,
        power=power,
        dt_start=0.01,
        tolerance=1e-8
    )

    # Analyze
    stats = analyze_distribution(points, radius)

    # Visualize
    plot_results(points, history, stats, radius)

    # Save
    np.savetxt('thomson_points_improved.txt', points,
               header=f'Thomson (FIRE): N={N}, radius={radius}, CV={stats["cv_nn"]:.6f}',
               fmt='%.10f')
    print(f"Points saved: thomson_points_improved.txt\n")

    # Print comparison of all restarts
    print("All restart results:")
    for i, result in enumerate(all_results):
        print(f"  Restart {i+1}: CV={result['cv_nn']:.6f}, E={result['energy']:.4f}")

    return points, history, stats


if __name__ == "__main__":
    points, history, stats = main()
