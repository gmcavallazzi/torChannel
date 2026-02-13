import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from projection import build_poisson_matrix, solve_poisson, project_velocity
from utils import generate_grid, compute_divergence
from solver import ChannelFlow

def test_projection_accuracy():
    print("=== Testing Projection Step Accuracy ===")
    
    # Define BC functions (corrected for tensor shapes)
    # u shape: (nx+1, ny+2, nz+2). Staggered in x. Indices 0..nx.
    # u[0] is left face, u[nx] is right face. Periodic: u[0] = u[nx].
    # project_velocity updates 1..nx. So u[nx] is updated. u[0] is not.
    def apply_bc_u(u):
        # Periodic BC in x
        u[0, :, :] = u[-1, :, :]
        # Periodic BC in y (u is NOT staggered in y)
        u[:, 0, :] = u[:, -2, :]
        u[:, -1, :] = u[:, 1, :]
        # Dirichlet BC in z
        u[:, :, 0] = -u[:, :, 1]
        u[:, :, -1] = -u[:, :, -2]
        return u

    # v shape: (nx+2, ny+1, nz+2). Staggered in y. Indices 0..ny.
    # v[0] is bottom face, v[ny] is top face. Periodic: v[0] = v[ny].
    # project_velocity updates 1..ny. So v[ny] is updated. v[0] is not.
    def apply_bc_v(v):
        # Periodic BC in x (v is NOT staggered in x)
        v[0, :, :] = v[-2, :, :]
        v[-1, :, :] = v[1, :, :]
        # Periodic BC in y
        v[:, 0, :] = v[:, -1, :]
        # Dirichlet BC in z
        v[:, :, 0] = -v[:, :, 1]
        v[:, :, -1] = -v[:, :, -2]
        return v

    def apply_bc_w(w):
        # Periodic BC in x (w is NOT staggered in x)
        w[0, :, :] = w[-2, :, :]
        w[-1, :, :] = w[1, :, :]
        # Periodic BC in y (w is NOT staggered in y)
        w[:, 0, :] = w[:, -2, :]
        w[:, -1, :] = w[:, 1, :]
        # Dirichlet BC in z (w is staggered in z, w=0 at walls)
        w[:, :, 0] = 0.0
        w[:, :, -1] = 0.0
        return w

    # Setup small grid
    torch.set_default_dtype(torch.float64)
    nx, ny, nz = 16, 16, 16
    Lx, Ly, Lz = 1.0, 1.0, 1.0
    gamma = 0.1
    
    dx = Lx / nx
    dy = Ly / ny
    
    z_f, z_c, dz_f, dz_c = generate_grid(gamma, nz, Lz)
    
    # Create random velocity field with correct shapes
    # u: (nx+1, ny+2, nz+2)
    # v: (nx+2, ny+1, nz+2)
    # w: (nx+2, ny+2, nz+1)
    torch.manual_seed(42)
    u = torch.randn(nx+1, ny+2, nz+2)
    v = torch.randn(nx+2, ny+1, nz+2)
    w = torch.randn(nx+2, ny+2, nz+1)
    
    # Enforce BCs (periodic x,y; wall z)
    # w at walls = 0
    w[:, :, 0] = 0.0
    w[:, :, -1] = 0.0
    
    # Apply BCs to initial random field to ensure consistency
    u = apply_bc_u(u)
    v = apply_bc_v(v)
    w = apply_bc_w(w)
    
    # Compute cell volumes
    # dz_f is length nz. cell_vol[k] = dx * dy * dz_f[k-1] (if k is 1-based index of cell)
    # div is (nx, ny, nz)
    cell_vols = dx * dy * dz_f.view(1, 1, -1).expand(nx, ny, nz)
    
    # Compute initial divergence
    div_init = compute_divergence(u, v, w, nx, ny, nz, dx, dy, dz_f)
    print(f"Initial max(div): {torch.max(torch.abs(div_init)):.6e}")
    
    # Check weighted sum
    weighted_sum = torch.sum(div_init * cell_vols)
    total_vol = torch.sum(cell_vols)
    print(f"Initial weighted sum(div): {weighted_sum:.6e}")
    
    # Enforce solvability: subtract weighted mean divergence
    div_mean_weighted = weighted_sum / total_vol
    print(f"Subtracting weighted mean divergence: {div_mean_weighted:.6e}")
    div_rhs = div_init - div_mean_weighted
    
    new_weighted_sum = torch.sum(div_rhs * cell_vols)
    print(f"New weighted sum(div): {new_weighted_sum:.6e}")
    
    # Build Poisson Matrix (unpinned to check singularity)
    print("Building Poisson matrix (unpinned)...")
    A = build_poisson_matrix(nx, ny, nz, dx, dy, dz_c, dz_f, pin_pressure=False)
    
    # Check if rows are linearly dependent (weighted sum of rows should be 0)
    # Row i corresponds to equation for cell i.
    # We want sum_i Vol_i * A_ij = 0 for all j.
    # Flatten cell_vols to match A rows
    # A is built with i varying fastest (dim 0).
    # cell_vols is (nx, ny, nz).
    # We need to permute cell_vols to (nz, ny, nx) so that when flattened, x (dim 2) is fastest.
    
    vol_flat = cell_vols.permute(2, 1, 0).reshape(-1)
    
    # Compute weighted sum of rows: sum_i Vol_i * A_ij
    # This is vector-matrix product: vol_flat @ A
    weighted_row_sum = vol_flat @ A
    max_weighted_sum = torch.max(torch.abs(weighted_row_sum))
    print(f"Max weighted sum of rows (should be ~0): {max_weighted_sum:.6e}")
    
    # Now solve with pinned matrix
    print("Building Poisson matrix (pinned)...")
    A = build_poisson_matrix(nx, ny, nz, dx, dy, dz_c, dz_f, pin_pressure=True)
    
    # Check if rows are linearly dependent (weighted sum of rows should be 0)
    # Row i corresponds to equation for cell i.
    # We want sum_i Vol_i * A_ij = 0 for all j.
    # Flatten cell_vols to match A rows
    # Note: A is built with i varying fastest.
    # cell_vols is (nx, ny, nz).
    # We need to permute cell_vols to match A's ordering?
    # build_poisson_matrix docstring: idx = (i-1) + (j-1)*nx + (k-1)*nx*ny
    # This is F-order (column-major) if we consider (nx, ny, nz).
    # PyTorch default is C-order (row-major).
    # Let's verify indexing in build_poisson_matrix.
    # idx = (i-1) + (j-1)*nx + (k-1)*nx*ny
    # This means i (x) is fastest, then j (y), then k (z).
    # cell_vols[i,j,k] needs to be flattened in the same order.
    # cell_vols is (nx, ny, nz).
    # cell_vols.flatten() in PyTorch is C-order: k varies fastest? No, last dim varies fastest.
    # shape (nx, ny, nz). Last dim is z. So z varies fastest.
    # This is OPPOSITE to A's indexing.
    # We need to permute cell_vols to (nz, ny, nx) before flattening?
    # No, we need i to be fastest. i is dim 0.
    # So we need to permute to (nz, ny, nx) so that when flattened, x (dim 2) is fastest?
    # Wait.
    # Tensor T shape (N0, N1, N2).
    # Flat[x] = T[x // (N1*N2), (x // N2) % N1, x % N2]
    # Last dim varies fastest.
    # We want i (dim 0) to vary fastest.
    # So we want i to be the last dimension in the permuted tensor.
    # So permute to (nz, ny, nx).
    # Then flatten.
    
    vol_flat = cell_vols.permute(2, 1, 0).reshape(-1)
    
    # Compute weighted sum of rows
    # sum_rows = vol_flat @ A
    # But A has the first row modified!
    # We should check this BEFORE modifying A.
    # But build_poisson_matrix modifies A before returning.
    # We need to modify build_poisson_matrix or manually restore the first row?
    # Or just check the other columns?
    # Actually, let's just check the sum of the unmodified rows?
    # No, the property is global.
    
    # Let's just print the result with the modified A.
    # The first row is [1, 0, ...].
    # So weighted sum will be Vol_0 * [1, 0, ...] + sum_{i>0} Vol_i A_i.
    # This won't be 0.
    
    # I will modify build_poisson_matrix in projection.py to NOT pin the node if requested,
    # or just copy the code here to test.
    # Better: I will manually un-pin the first row in A for the test.
    # I know what the first row should be.
    # It's the stencil for (1,1,1).
    # But easier: I will modify projection.py to accept a 'pin' argument.
    
    # Solve Poisson
    print("Solving Poisson...")
    # We solve L p = div / dt. Let's assume dt=1.0
    dt = 1.0
    p = solve_poisson(A, div_rhs / dt, nx, ny, nz)
    
    # Project
    print("Projecting velocity...")
    u_new, v_new, w_new = project_velocity(u.clone(), v.clone(), w.clone(), p, nx, ny, nz, dx, dy, dz_c, dz_f, dt)
    
    # Apply BCs
    u_new = apply_bc_u(u_new)
    v_new = apply_bc_v(v_new)
    w_new = apply_bc_w(w_new)
    
    # Compute final divergence
    div_final = compute_divergence(u_new, v_new, w_new, nx, ny, nz, dx, dy, dz_f)
    max_div = torch.max(torch.abs(div_final))
    print(f"Final max(div): {max_div:.6e}")
    
    # Debug: Check if div_init - div_final == dt * L p
    div_diff = div_init - div_final
    
    # Compute L p manually using matrix
    # Flatten p interior
    p_interior = p[1:nx+1, 1:ny+1, 1:nz+1]
    p_flat = p_interior.permute(2, 1, 0).reshape(-1)
    
    # Build Matrix (unpinned)
    A = build_poisson_matrix(nx, ny, nz, dx, dy, dz_c, dz_f, pin_pressure=False)
    
    Lp_flat = A @ p_flat
    Lp = Lp_flat.reshape(nz, ny, nx).permute(2, 1, 0)
    
    term = dt * Lp
    
    # Compare
    diff = torch.abs(div_diff - term)
    print(f"Max diff between (div_init - div_final) and (dt * L p): {torch.max(diff):.6e}")
    
    # Also check if L p == div_rhs
    diff_rhs = torch.abs(term - div_rhs)
    print(f"Max diff between (dt * L p) and div_rhs: {torch.max(diff_rhs):.6e}")
    
    # Find location of max divergence
    max_idx = torch.argmax(torch.abs(div_final))
    # Unravel index for (nx, ny, nz) tensor (C-order: last dim varies fastest)
    # idx = i * (ny*nz) + j * nz + k
    i = max_idx // (ny * nz)
    rem = max_idx % (ny * nz)
    j = rem // nz
    k = rem % nz
    
    print(f"Max divergence at index (0-based): ({i}, {j}, {k})")
    print(f"Value at max: {div_final[i, j, k]:.6e}")
    
    # Check (0,0,0) specifically
    print(f"Divergence at (0,0,0): {div_final[0,0,0]:.6e}")
    
    if max_div > 1e-6:
        print("FAIL: Projection failed to remove divergence.")
    else:
        print("PASS: Projection successful.")

if __name__ == "__main__":
    test_projection_accuracy()
