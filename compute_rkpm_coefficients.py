#!/usr/bin/env python
"""
Precompute RKPM coefficients for IBM simulations using MPI parallelization.

This script computes the expensive RKPM interpolation/spreading coefficients
offline, saving them to NPZ files for fast loading during the main simulation.

Usage:
    # Serial (single core)
    python compute_rkpm_coefficients.py config_canopy_monti.yaml

    # Parallel (MPI, 8 cores)
    mpirun -np 8 python compute_rkpm_coefficients.py config_canopy_monti.yaml

Output:
    - rkpm_coefficients.npz: Support structures (indices, weights, volumes)
    - rkpm_epsilon.npz: Epsilon values for partition of unity

Author: Claude Code
Date: 2025
"""

import numpy as np
import yaml
import sys
import os
import math
import hashlib
from mpi4py import MPI

def load_config(config_file):
    """Load YAML configuration file."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def compute_config_hash(config, n_lag):
    """
    Compute hash of configuration parameters that affect RKPM coefficients.
    """
    hash_params = {
        'nx': config['grid']['nx'],
        'ny': config['grid']['ny'],
        'nz': config['grid']['nz'],
        'Lx': config['domain']['Lx'],
        'Ly': config['domain']['Ly'],
        'Lz': config['domain']['Lz'],
        'n_lag': n_lag,
        'rkpm_basis': config.get('ibm', {}).get('rkpm_basis', 'linear'),
        'rkpm_search_range': config.get('ibm', {}).get('rkpm_search_range', 4),
        'obstacle_type': config['ibm']['obstacle_type'],
    }

    # Add grid-specific params
    if config['domain'].get('stretching_type') == 'hybrid':
        hash_params['nz_uniform'] = config['grid'].get('nz_uniform', 75)
        hash_params['nz_stretched'] = config['grid'].get('nz_stretched', 75)
        hash_params['z_transition'] = config['domain'].get('z_transition', 0.25)
        hash_params['gamma_stretched'] = config['domain'].get('gamma_stretched', 1.8)
    else:
        hash_params['gamma'] = config['flow'].get('gamma', 2.4)
        hash_params['stretching_type'] = config['domain'].get('stretching_type', 'symmetric')

    # Add obstacle-specific params
    if config['ibm']['obstacle_type'] == 'canopy':
        canopy_cfg = config['ibm']['canopy']
        hash_params['spacing'] = canopy_cfg['spacing']
        hash_params['height'] = canopy_cfg['height']
        hash_params['diameter'] = canopy_cfg['diameter']
        hash_params['seed'] = canopy_cfg.get('seed', 42)

    # Convert to string and hash
    hash_str = str(sorted(hash_params.items()))
    return hashlib.md5(hash_str.encode()).hexdigest()

def scalar_roma_kernel(r, h):
    """Roma et al. (1999) Kernel (scalar version)."""
    ar = abs(r)
    q = ar / h
    if q <= 0.5:
        return (1.0/3.0) * (1.0 + math.sqrt(max(0.0, 1.0 - 3.0 * q**2))) / h
    elif q <= 1.5:
        return (1.0/6.0) * (5.0 - 3.0 * q - math.sqrt(max(0.0, 1.0 - 3.0 * (1.0 - q)**2))) / h
    return 0.0

def compute_rkpm_for_point(i, x_lag, y_lag, z_lag, x_grid, y_grid, z_grid, dz_grid,
                           use_linear, search_range, dx_grid, dy_grid):
    """
    Compute RKPM coefficients for a single Lagrangian point.

    Returns:
        Dictionary with 'ix', 'iy', 'iz', 'wdt', 'vol' arrays
    """
    xl, yl, zl = x_lag[i], y_lag[i], z_lag[i]

    nx_g, ny_g, nz_g = len(x_grid), len(y_grid), len(z_grid)
    n_basis = 4 if use_linear else 10

    # 1. Find nearest grid point
    idx_x = np.abs(x_grid - xl).argmin()
    idx_y = np.abs(y_grid - yl).argmin()
    idx_z = np.abs(z_grid - zl).argmin()

    # 2. Determine support size (adaptive)
    def get_h(grid, idx, dist):
        spacings = []
        if idx > 0: spacings.append(abs(grid[idx] - grid[idx-1]))
        if idx < len(grid)-1: spacings.append(abs(grid[idx+1] - grid[idx]))
        if not spacings: return 1.0
        R_min, R_max = min(spacings), max(spacings)
        return ((5.0 * R_max + R_min) / 6.0 + dist / 9.0) * 1.5

    hx = get_h(x_grid, idx_x, abs(x_grid[idx_x] - xl))
    hy = get_h(y_grid, idx_y, abs(y_grid[idx_y] - yl))
    hz = get_h(z_grid, idx_z, abs(z_grid[idx_z] - zl))

    # 3. Find neighbors in support cage
    support_radius = 1.5

    ix_min = max(0, idx_x - search_range)
    ix_max = min(nx_g, idx_x + search_range + 1)
    iy_min = max(0, idx_y - search_range)
    iy_max = min(ny_g, idx_y + search_range + 1)
    iz_min = max(0, idx_z - search_range)
    iz_max = min(nz_g, idx_z + search_range + 1)

    neighbors = []
    M = np.zeros((n_basis, n_basis))

    # Scaling matrix H_inv
    if use_linear:
        H_inv = np.array([1.0, 1.0/hx, 1.0/hy, 1.0/hz])
    else:
        H_inv = np.array([
            1.0,
            1.0/hx, 1.0/hy, 1.0/hz,
            1.0/(hx*hy), 1.0/(hx*hz), 1.0/(hy*hz),
            1.0/(hx**2), 1.0/(hy**2), 1.0/(hz**2)
        ])

    for iz in range(iz_min, iz_max):
        dz_val = z_grid[iz] - zl
        if abs(dz_val) > support_radius * hz: continue
        val_z = scalar_roma_kernel(dz_val, hz)
        if val_z == 0: continue

        # Cell height
        if dz_grid is not None:
            dz_cell = dz_grid[iz]
        elif iz < nz_g - 1:
            dz_cell = z_grid[iz+1] - z_grid[iz]
        else:
            dz_cell = z_grid[iz] - z_grid[iz-1]

        for iy in range(iy_min, iy_max):
            dy_val = y_grid[iy] - yl
            if abs(dy_val) > support_radius * hy: continue
            val_y = scalar_roma_kernel(dy_val, hy)
            if val_y == 0: continue

            for ix in range(ix_min, ix_max):
                dx_val = x_grid[ix] - xl
                if abs(dx_val) > support_radius * hx: continue
                val_x = scalar_roma_kernel(dx_val, hx)
                if val_x == 0: continue

                # Kernel value
                phi = val_x * val_y * val_z

                # Volume
                vol = dx_grid * dy_grid * dz_cell

                # Scaled Basis
                sx, sy, sz = dx_val/hx, dy_val/hy, dz_val/hz
                if use_linear:
                    P_scaled = np.array([1.0, sx, sy, sz])
                else:
                    P_scaled = np.array([
                        1.0,
                        sx, sy, sz,
                        sx*sy, sx*sz, sy*sz,
                        sx**2, sy**2, sz**2
                    ])

                # Accumulate M
                M += np.outer(P_scaled, P_scaled) * phi * vol

                neighbors.append({
                    'ix': ix, 'iy': iy, 'iz': iz,
                    'dx': dx_val, 'dy': dy_val, 'dz': dz_val,
                    'vol': vol,
                    'phi': phi
                })

    # Solve M * c = [1, 0, ...]^T
    rhs = np.zeros(n_basis)
    rhs[0] = 1.0

    # Regularize if needed
    if np.linalg.cond(M) > 1e12:
        M += np.eye(n_basis) * 1e-8

    try:
        c = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        c = np.zeros(n_basis)
        c[0] = 1.0

    # Transform to b
    b = c * H_inv

    # Compute final weights
    ix_arr = np.array([n['ix'] for n in neighbors], dtype=np.int32)
    iy_arr = np.array([n['iy'] for n in neighbors], dtype=np.int32)
    iz_arr = np.array([n['iz'] for n in neighbors], dtype=np.int32)
    vol_arr = np.array([n['vol'] for n in neighbors], dtype=np.float64)

    # Recompute weights with unscaled basis
    wdt_list = []
    for n in neighbors:
        dx, dy, dz = n['dx'], n['dy'], n['dz']
        if use_linear:
            P = np.array([1.0, dx, dy, dz])
        else:
            P = np.array([
                1.0,
                dx, dy, dz,
                dx*dy, dx*dz, dy*dz,
                dx**2, dy**2, dz**2
            ])
        correction = np.dot(P, b)
        wdt = correction * n['phi']
        wdt_list.append(wdt)

    wdt_arr = np.array(wdt_list, dtype=np.float64)

    return {
        'ix': ix_arr,
        'iy': iy_arr,
        'iz': iz_arr,
        'wdt': wdt_arr,
        'vol': vol_arr
    }

def compute_rkpm_parallel(config, x_lag, y_lag, z_lag, x_grid, y_grid, z_grid,
                          dz_grid, component, comm, rank, size):
    """
    Compute RKPM coefficients in parallel using MPI.

    Each rank processes a subset of Lagrangian points.
    """
    use_linear = config.get('ibm', {}).get('rkpm_basis', 'linear') == 'linear'
    search_range = config.get('ibm', {}).get('rkpm_search_range', 4)

    Lx = config['domain']['Lx']
    Ly = config['domain']['Ly']
    nx = config['grid']['nx']
    ny = config['grid']['ny']
    dx_grid = Lx / nx
    dy_grid = Ly / ny

    n_lag = len(x_lag)

    # Distribute work
    points_per_rank = n_lag // size
    remainder = n_lag % size

    if rank < remainder:
        start = rank * (points_per_rank + 1)
        end = start + points_per_rank + 1
    else:
        start = rank * points_per_rank + remainder
        end = start + points_per_rank

    my_points = list(range(start, end))

    if rank == 0:
        print(f"\n[Rank 0] Computing RKPM for {component}-velocity...", flush=True)
        print(f"  Basis: {'linear' if use_linear else 'quadratic'} ({'4' if use_linear else '10'} terms)", flush=True)
        print(f"  Search range: {search_range}", flush=True)
        print(f"  Total Lagrangian points: {n_lag}", flush=True)
        print(f"  MPI ranks: {size}", flush=True)
        print(f"  Points per rank: {points_per_rank} (+ {remainder} to first ranks)", flush=True)

    # Compute coefficients for local points
    local_support = []
    progress_interval = max(1, len(my_points) // 10)

    for local_idx, i in enumerate(my_points):
        support = compute_rkpm_for_point(
            i, x_lag, y_lag, z_lag, x_grid, y_grid, z_grid, dz_grid,
            use_linear, search_range, dx_grid, dy_grid
        )
        local_support.append((i, support))

        if (local_idx + 1) % progress_interval == 0 or local_idx == len(my_points) - 1:
            print(f"[Rank {rank}] Progress: {local_idx+1}/{len(my_points)} "
                  f"({(local_idx+1)/len(my_points)*100:.1f}%)", flush=True)

    # Gather all results to rank 0
    if rank == 0:
        print(f"\n[Rank 0] Gathering results from all ranks...", flush=True)

    all_support = comm.gather(local_support, root=0)

    if rank == 0:
        # Flatten gathered results
        support_list = [None] * n_lag
        for rank_data in all_support:
            for i, support in rank_data:
                support_list[i] = support

        print(f"[Rank 0] ✓ Completed RKPM for {component}-velocity", flush=True)
        return support_list
    else:
        return None

def compute_epsilon_direct(support_list, n_lag):
    """
    Direct solver for epsilon: A * eps = 1
    A_ij = sum_k (w_ik * w_jk * vol_k)

    Optimized version using grid-based sparse construction.
    """
    if not support_list:
        return np.ones(n_lag, dtype=np.float64)

    from scipy.sparse import lil_matrix, csr_matrix
    from scipy.sparse.linalg import spsolve
    import time

    print("  Building sparse matrix A (optimized)...", flush=True)
    start_time = time.time()

    # Use LIL format for efficient construction
    A = lil_matrix((n_lag, n_lag), dtype=np.float64)

    # Build grid point → Lagrangian point mapping for faster lookup
    print("    Creating reverse lookup table...", flush=True)
    grid_to_lag = {}  # grid_idx → list of (lag_idx, local_idx, weight, vol)

    for i in range(n_lag):
        s = support_list[i]
        if len(s['ix']) == 0:
            continue

        for local_idx in range(len(s['ix'])):
            grid_key = (int(s['ix'][local_idx]), int(s['iy'][local_idx]), int(s['iz'][local_idx]))
            if grid_key not in grid_to_lag:
                grid_to_lag[grid_key] = []
            grid_to_lag[grid_key].append((i, local_idx, s['wdt'][local_idx], s['vol'][local_idx]))

    n_grid_points = len(grid_to_lag)
    print(f"    Grid points with neighbors: {n_grid_points}", flush=True)

    # Compute matrix entries efficiently
    print("    Computing matrix entries...", flush=True)
    progress_interval = max(1, n_lag // 20)
    nonzeros = 0

    for i in range(n_lag):
        if (i + 1) % progress_interval == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (n_lag - i - 1) / rate if rate > 0 else 0
            print(f"      Progress: {i+1}/{n_lag} ({(i+1)/n_lag*100:.1f}%) - "
                  f"ETA: {remaining/60:.1f} min", flush=True)

        s_i = support_list[i]
        if len(s_i['ix']) == 0:
            A[i, i] = 1.0  # Isolated point
            continue

        # For each grid point in i's support
        for local_i in range(len(s_i['ix'])):
            grid_key = (int(s_i['ix'][local_i]), int(s_i['iy'][local_i]), int(s_i['iz'][local_i]))
            w_i = s_i['wdt'][local_i]
            vol_i = s_i['vol'][local_i]

            # Find all Lagrangian points sharing this grid point
            if grid_key in grid_to_lag:
                for j, local_j, w_j, vol_j in grid_to_lag[grid_key]:
                    # A_ij += w_i * w_j * vol
                    A[i, j] += w_i * w_j * vol_i

        nonzeros += A.rows[i].__len__()

    build_time = time.time() - start_time
    print(f"    ✓ Matrix built in {build_time/60:.1f} minutes", flush=True)
    print(f"    Matrix A: {n_lag}×{n_lag}, {nonzeros} nonzeros", flush=True)
    print(f"    Sparsity: {100 * (1 - nonzeros/(n_lag**2)):.3f}%", flush=True)

    # Convert to CSR for efficient solving
    print("    Converting to CSR format...", flush=True)
    A_csr = A.tocsr()

    # Solve A * eps = 1
    print("  Solving A * eps = 1...", flush=True)
    solve_start = time.time()

    rhs = np.ones(n_lag, dtype=np.float64)
    epsilon = spsolve(A_csr, rhs)

    solve_time = time.time() - solve_start
    print(f"  ✓ Solved in {solve_time:.1f} seconds", flush=True)
    print(f"  ✓ Epsilon computed (range: [{epsilon.min():.3e}, {epsilon.max():.3e}])", flush=True)

    total_time = time.time() - start_time
    print(f"  Total epsilon computation time: {total_time/60:.1f} minutes", flush=True)

    return epsilon

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if len(sys.argv) < 2:
        if rank == 0:
            print("Usage: python compute_rkpm_coefficients.py <config.yaml>")
            print("   or: mpirun -np N python compute_rkpm_coefficients.py <config.yaml>")
        sys.exit(1)

    config_file = sys.argv[1]

    if rank == 0:
        print("=" * 80)
        print("RKPM COEFFICIENT PRECOMPUTATION (MPI)")
        print("=" * 80)
        print(f"\nConfig file: {config_file}")
        print(f"MPI ranks: {size}")

    # Load config on all ranks
    config = load_config(config_file)

    # Generate grid and Lagrangian points (all ranks need this data)
    if rank == 0:
        print("\nGenerating grid and Lagrangian points...", flush=True)

    # Import grid generation
    import torch
    from utils import generate_grid, generate_hybrid_grid

    nx = config['grid']['nx']
    ny = config['grid']['ny']
    nz = config['grid']['nz']
    Lx = config['domain']['Lx']
    Ly = config['domain']['Ly']
    Lz = config['domain']['Lz']

    # Generate grids (on CPU for all ranks)
    x_f = torch.linspace(0, Lx, nx+1, dtype=torch.float64).numpy()
    x_c = 0.5 * (x_f[:-1] + x_f[1:])

    y_f = torch.linspace(0, Ly, ny+1, dtype=torch.float64).numpy()
    y_c = 0.5 * (y_f[:-1] + y_f[1:])

    stretching_type = config['domain'].get('stretching_type', 'symmetric')
    if stretching_type == 'hybrid':
        nz_uniform = config['grid'].get('nz_uniform', 75)
        nz_stretched = config['grid'].get('nz_stretched', 75)
        z_transition = config['domain'].get('z_transition', 0.25)
        gamma_stretched = config['domain'].get('gamma_stretched', 1.8)

        z_f, z_c, dz_f, dz_c = generate_hybrid_grid(
            nz_uniform, nz_stretched, z_transition, Lz, gamma_stretched, device='cpu'
        )
    else:
        gamma = config['flow'].get('gamma', 2.4)
        z_f, z_c, dz_f, dz_c = generate_grid(gamma, nz, Lz, device='cpu', stretching_type=stretching_type)

    z_f = z_f.numpy()
    z_c = z_c.numpy()
    dz_f = dz_f.numpy()
    dz_c = dz_c.numpy()

    # Generate Lagrangian points
    obstacle_type = config['ibm']['obstacle_type']

    if obstacle_type == 'canopy':
        from cylinder_lagrangian_generator import generate_canopy_lagrangian_points
        if rank == 0:
            print("\nGenerating canopy Lagrangian points...", flush=True)
        x_lag, y_lag, z_lag, dS_lag, centers = generate_canopy_lagrangian_points(config)
    else:
        if rank == 0:
            print(f"Error: Obstacle type '{obstacle_type}' not supported in this script")
        sys.exit(1)

    n_lag = len(x_lag)

    if rank == 0:
        print(f"✓ Grid: {nx}×{ny}×{len(z_f)-1}")
        print(f"✓ Lagrangian points: {n_lag}")

    # Compute config hash
    config_hash = compute_config_hash(config, n_lag)
    if rank == 0:
        print(f"✓ Configuration hash: {config_hash}")

    # Compute RKPM coefficients in parallel
    support_u = compute_rkpm_parallel(config, x_lag, y_lag, z_lag, x_f, y_c, z_c, dz_c, 'u', comm, rank, size)
    support_v = compute_rkpm_parallel(config, x_lag, y_lag, z_lag, x_c, y_f, z_c, dz_c, 'v', comm, rank, size)
    support_w = compute_rkpm_parallel(config, x_lag, y_lag, z_lag, x_c, y_c, z_f, dz_f, 'w', comm, rank, size)

    # Compute epsilon (only on rank 0)
    if rank == 0:
        print("\n" + "=" * 80)
        print("COMPUTING EPSILON VALUES")
        print("=" * 80)

        print("\nComputing epsilon for u-velocity...", flush=True)
        epsilon_u = compute_epsilon_direct(support_u, n_lag)

        print("\nComputing epsilon for v-velocity...", flush=True)
        epsilon_v = compute_epsilon_direct(support_v, n_lag)

        print("\nComputing epsilon for w-velocity...", flush=True)
        epsilon_w = compute_epsilon_direct(support_w, n_lag)

        # Save to files
        print("\n" + "=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)

        output_dir = config.get('output', {}).get('results_folder', 'results')
        os.makedirs(output_dir, exist_ok=True)

        coeff_file = os.path.join(output_dir, 'rkpm_coefficients.npz')
        epsilon_file = os.path.join(output_dir, 'rkpm_epsilon.npz')

        # Flatten support lists
        def flatten_support(support_list):
            ix_list, iy_list, iz_list, wdt_list, vol_list = [], [], [], [], []
            n_neighbors = []

            for s in support_list:
                ix_list.append(s['ix'])
                iy_list.append(s['iy'])
                iz_list.append(s['iz'])
                wdt_list.append(s['wdt'])
                vol_list.append(s['vol'])
                n_neighbors.append(len(s['ix']))

            return {
                'ix': np.concatenate(ix_list),
                'iy': np.concatenate(iy_list),
                'iz': np.concatenate(iz_list),
                'wdt': np.concatenate(wdt_list),
                'vol': np.concatenate(vol_list),
                'n_neighbors': np.array(n_neighbors, dtype=np.int32)
            }

        print(f"\nFlattening support structures...", flush=True)
        u_flat = flatten_support(support_u)
        v_flat = flatten_support(support_v)
        w_flat = flatten_support(support_w)

        print(f"Saving coefficients to {coeff_file}...", flush=True)
        np.savez_compressed(
            coeff_file,
            config_hash=config_hash,
            n_lag=n_lag,
            # U-velocity support
            u_ix=u_flat['ix'], u_iy=u_flat['iy'], u_iz=u_flat['iz'],
            u_wdt=u_flat['wdt'], u_vol=u_flat['vol'], u_n_neighbors=u_flat['n_neighbors'],
            # V-velocity support
            v_ix=v_flat['ix'], v_iy=v_flat['iy'], v_iz=v_flat['iz'],
            v_wdt=v_flat['wdt'], v_vol=v_flat['vol'], v_n_neighbors=v_flat['n_neighbors'],
            # W-velocity support
            w_ix=w_flat['ix'], w_iy=w_flat['iy'], w_iz=w_flat['iz'],
            w_wdt=w_flat['wdt'], w_vol=w_flat['vol'], w_n_neighbors=w_flat['n_neighbors']
        )

        coeff_size = os.path.getsize(coeff_file) / 1e6
        print(f"✓ Saved coefficients ({coeff_size:.1f} MB)", flush=True)

        print(f"\nSaving epsilon values to {epsilon_file}...", flush=True)
        np.savez_compressed(
            epsilon_file,
            config_hash=config_hash,
            n_lag=n_lag,
            epsilon_u=epsilon_u,
            epsilon_v=epsilon_v,
            epsilon_w=epsilon_w
        )

        eps_size = os.path.getsize(epsilon_file) / 1e6
        print(f"✓ Saved epsilon values ({eps_size:.1f} MB)", flush=True)

        print("\n" + "=" * 80)
        print("PRECOMPUTATION COMPLETE")
        print("=" * 80)
        print(f"\nGenerated files:")
        print(f"  1. {coeff_file} ({coeff_size:.1f} MB)")
        print(f"  2. {epsilon_file} ({eps_size:.1f} MB)")
        print(f"\nConfiguration hash: {config_hash}")
        print(f"\nTo use in simulation, update config file:")
        print(f"  ibm:")
        print(f"    rkpm_coefficients_file: \"{coeff_file}\"")
        print(f"    rkpm_epsilon_file: \"{epsilon_file}\"")
        print("")

if __name__ == '__main__':
    main()
