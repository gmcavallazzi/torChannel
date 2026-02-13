import numpy as np
import matplotlib.pyplot as plt

def regular_delta(r, h):
    """
    Regularized delta function (3-point kernel).
    r: distance
    h: support size
    """
    ar = abs(r)
    val = 0.0
    
    # Condition 1: h/2 < |r| <= 3h/2
    if (ar > 0.5 * h) and (ar <= 1.5 * h):
        val = (5.0 - 3.0 * ar / h - np.sqrt(-3.0 * (1.0 - ar/h)**2 + 1.0)) / (6.0 * h)
        
    # Condition 2: |r| <= h/2
    elif ar <= 0.5 * h:
        val = (1.0 + np.sqrt(1.0 - 3.0 * (ar/h)**2)) / (3.0 * h)
        
    return val

def test_rkpm_1d():
    print("========================================")
    print("1D RKPM VERIFICATION (QUADRATIC BASIS)")
    print("========================================")
    
    # 1. Setup Non-Uniform Grid
    # Create a grid that gets finer in the middle
    N = 20
    # x = tanh stretched grid from 0 to 1
    xi = np.linspace(-1, 1, N)
    x_grid = 0.5 * (1.0 + np.tanh(2.0 * xi) / np.tanh(2.0))
    
    print(f"Grid size: {N}")
    print(f"Grid range: [{x_grid[0]:.4f}, {x_grid[-1]:.4f}]")
    
    # 2. Define Lagrangian Point (Off-grid)
    # Pick a point in a stretched region
    X_l = 0.333333 
    print(f"Lagrangian Point: {X_l:.6f}")
    
    # Find nearest grid node
    idx = (np.abs(x_grid - X_l)).argmin()
    print(f"Nearest Node: {idx} ({x_grid[idx]:.6f})")
    
    # 3. Determine Support Size (h)
    # Adaptive h based on local spacing
    def get_R_stats(grid, i):
        spacings = []
        if i > 0: spacings.append(abs(grid[i] - grid[i-1]))
        if i < len(grid)-1: spacings.append(abs(grid[i+1] - grid[i]))
        return min(spacings), max(spacings)
        
    R1, R11 = get_R_stats(x_grid, idx)
    dist = abs(x_grid[idx] - X_l)
    
    # Pinelli formula
    h = (5.0 * R11 + R1) / 6.0 + dist / 9.0
    print(f"Computed h: {h:.6f}")
    
    # 4. Identify Neighbors
    # Support radius = 1.5 * h (standard) or 2.0 * h (extended)
    # Let's try to reproduce the issue: Standard support might be too small for quadratic?
    support_radius_factor = 2.0 
    support_radius = support_radius_factor * h
    
    neighbors = []
    for i, x in enumerate(x_grid):
        if abs(x - X_l) <= support_radius:
            neighbors.append(i)
            
    print(f"Neighbors in support ({support_radius_factor}*h): {len(neighbors)}")
    print(f"Neighbor indices: {neighbors}")
    
    if len(neighbors) < 3:
        print("ERROR: Not enough neighbors for Quadratic basis (need >= 3)")
        return

    # 5. Build Moment Matrix (Quadratic Basis: 1, x, x^2)
    # M_ab = sum( P_a * P_b * phi * dV ) ?? 
    # WAIT: rkpm_explained.md says:
    # "M = sum phi * [1, dx, dx^2; ...]"
    # AND "For FD: M = sum phi * p * p^T" (NO VOLUME)
    # Let's stick to the FD formulation in ibm.py which assumes no volume in M.
    
    n_basis = 3 # {1, x, x^2}
    M = np.zeros((n_basis, n_basis))
    
    # Scaling matrix H_inv
    # To improve conditioning, we scale x by h
    # P_scaled = [1, x/h, (x/h)^2]
    
    for i in neighbors:
        dx = x_grid[i] - X_l
        phi = regular_delta(dx, h)
        
        # Scaled basis vector
        s = dx / h
        P = np.array([1.0, s, s**2])
        
        M += np.outer(P, P) * phi
        
    print("\nMoment Matrix M:")
    print(M)
    print(f"Condition Number: {np.linalg.cond(M):.4e}")
    
    # 6. Solve for Coefficients b
    rhs = np.zeros(n_basis)
    rhs[0] = 1.0
    
    try:
        c = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        print("ERROR: Matrix singular!")
        return
        
    print(f"Coefficients c: {c}")
    
    # 7. Compute Corrected Weights
    # w_i = phi_i * (P_scaled_i . c)
    weights = []
    total_weight = 0.0
    first_moment = 0.0
    second_moment = 0.0
    
    print("\nWeights:")
    for i in neighbors:
        dx = x_grid[i] - X_l
        phi = regular_delta(dx, h)
        s = dx / h
        P = np.array([1.0, s, s**2])
        
        correction = np.dot(P, c)
        w = phi * correction
        weights.append(w)
        
        total_weight += w
        first_moment += w * dx
        second_moment += w * (dx**2) # Should be 0 for consistency? No, sum w * (x-X)^2 = 0?
        # Wait, moment condition is sum w * (x-X)^n = delta_n0
        # n=0: sum w = 1
        # n=1: sum w * (x-X) = 0  => sum w*x - X*sum w = 0 => sum w*x = X
        # n=2: sum w * (x-X)^2 = 0
        
        print(f"  Node {i}: x={x_grid[i]:.4f}, dx={dx:.4f}, phi={phi:.4f}, corr={correction:.4f}, w={w:.6f}")
        
    print("\nVerification:")
    print(f"  Sum(w) [Expect 1.0]: {total_weight:.10f}")
    print(f"  Sum(w * dx) [Expect 0.0]: {first_moment:.10f}")
    print(f"  Sum(w * dx^2) [Expect 0.0]: {second_moment:.10f}")
    
    # Check errors
    err_0 = abs(total_weight - 1.0)
    err_1 = abs(first_moment)
    err_2 = abs(second_moment)
    
    if max(err_0, err_1, err_2) < 1e-10:
        print("\nSUCCESS: Moments reproduced correctly!")
    else:
        print("\nFAILURE: Moments not reproduced.")

if __name__ == "__main__":
    test_rkpm_1d()
