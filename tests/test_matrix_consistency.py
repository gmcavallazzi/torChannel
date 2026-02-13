import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from projection import build_poisson_matrix, solve_poisson, project_velocity
from utils import generate_grid, compute_divergence
from solver import ChannelFlow

def test_matrix_consistency():
    print("=== Testing Matrix-Operator Consistency ===")
    
    torch.set_default_dtype(torch.float64)
    nx, ny, nz = 16, 16, 16
    Lx, Ly, Lz = 1.0, 1.0, 1.0
    gamma = 0.1
    
    dx = Lx / nx
    dy = Ly / ny
    
    z_f, z_c, dz_f, dz_c = generate_grid(gamma, nz, Lz)
    
    # Create random pressure field
    torch.manual_seed(42)
    p = torch.randn(nx+2, ny+2, nz+2)
    
    # Enforce BCs on p for consistency test
    # Periodic x, y
    p[0, :, :] = p[nx, :, :]
    p[nx+1, :, :] = p[1, :, :]
    p[:, 0, :] = p[:, ny, :]
    p[:, ny+1, :] = p[:, 1, :]
    # Neumann z
    p[:, :, 0] = p[:, :, 1]
    p[:, :, nz+1] = p[:, :, nz]
    
    # Compute Laplacian via operators: div(grad(p))
    # 1. Grad p
    # u_grad = dp/dx (at faces)
    # v_grad = dp/dy (at faces)
    # w_grad = dp/dz (at faces)
    
    u_grad = torch.zeros(nx+1, ny+2, nz+2)
    v_grad = torch.zeros(nx+2, ny+1, nz+2)
    w_grad = torch.zeros(nx+2, ny+2, nz+1)
    
    # dp/dx
    for i in range(1, nx+1):
        for j in range(1, ny+1):
            for k in range(1, nz+1):
                u_grad[i, j, k] = (p[i+1, j, k] - p[i, j, k]) / dx
                
    # dp/dy
    for i in range(1, nx+1):
        for j in range(1, ny+1):
            for k in range(1, nz+1):
                v_grad[i, j, k] = (p[i, j+1, k] - p[i, j, k]) / dy
                
    # dp/dz
    for i in range(1, nx+1):
        for j in range(1, ny+1):
            for k in range(1, nz): # Interior faces 1..nz-1
                w_grad[i, j, k] = (p[i, j, k+1] - p[i, j, k]) / dz_c[k]
                
    # 2. Div(grad)
    # Re-use compute_divergence, but we need to pass fields with correct shapes/ghosts?
    # compute_divergence expects u(nx+1, ny+2, nz+2) etc.
    # We need to fill ghosts for u_grad, v_grad, w_grad?
    # Actually, compute_divergence only uses interior indices.
    # u indices: i=1..nx. u[i] and u[i-1].
    # u_grad has 1..nx. u_grad[0] is not set.
    # But loop is i=1..nx. u[i]-u[i-1].
    # For i=1, need u[0].
    # u[0] corresponds to face 0.
    # For periodic BC, u[0] = u[nx].
    # Let's fill ghosts for gradients.
    
    # u_grad ghosts (periodic)
    u_grad[0, :, :] = u_grad[nx, :, :]
    
    # v_grad ghosts (periodic)
    v_grad[:, 0, :] = v_grad[:, ny, :]
    
    # w_grad ghosts (wall)
    # w_grad is dp/dz. At walls dp/dz = 0.
    # w_grad[..., 0] is bottom wall. w_grad[..., nz] is top wall.
    # We initialized w_grad to zeros, so this is already set.
    
    # We need to reshape u_grad to match compute_divergence expectation
    # u_grad is (nx+1, ny+2, nz+2). compute_divergence takes u of this shape.
    # But we need to be careful about indices.
    # compute_divergence:
    # du_dx = (u[i, j, k] - u[i-1, j, k]) / dx
    # This matches.
    
    lap_op = compute_divergence(u_grad, v_grad, w_grad, nx, ny, nz, dx, dy, dz_f)
    
    # Compute Laplacian via Matrix
    # Flatten p interior
    p_interior = p[1:nx+1, 1:ny+1, 1:nz+1]
    # Permute to match matrix (i fastest)
    p_flat = p_interior.permute(2, 1, 0).reshape(-1)
    
    # Build Matrix (unpinned)
    A = build_poisson_matrix(nx, ny, nz, dx, dy, dz_c, dz_f, pin_pressure=False)
    
    lap_matrix_flat = A @ p_flat
    
    # Reshape back
    lap_matrix = lap_matrix_flat.reshape(nz, ny, nx).permute(2, 1, 0)
    
    # Compare
    diff = torch.abs(lap_op - lap_matrix)
    max_diff = torch.max(diff)
    print(f"Max difference between Matrix and Operator Laplacian: {max_diff:.6e}")
    
    if max_diff < 1e-6:
        print("PASS: Matrix is consistent with Operator.")
    else:
        print("FAIL: Matrix is inconsistent.")
        # Print some details
        max_idx = torch.argmax(diff)
        print(f"Max diff at index {max_idx}")
        # Unravel
        k = max_idx // (nx * ny)
        rem = max_idx % (nx * ny)
        j = rem // nx
        i = rem % nx
        print(f"Index (i,j,k): ({i}, {j}, {k})")
        print(f"Operator: {lap_op[i,j,k]:.6e}")
        print(f"Matrix: {lap_matrix[i,j,k]:.6e}")

if __name__ == "__main__":
    test_matrix_consistency()
