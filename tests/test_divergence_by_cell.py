import torch
import sys
sys.path.append('..')

from projection_fft import initialize_fft_solver, solve_poisson_fft
from utils import generate_grid, compute_divergence
from projection import project_velocity

# Define apply_bc_all inline to avoid import issues
def apply_bc_all(u, v, w, top_wall_bc_type='dirichlet'):
    """Apply boundary conditions to velocity fields"""
    # U-velocity
    u[0, :, :] = u[-1, :, :]
    u[:, 0, :] = u[:, -2, :]
    u[:, -1, :] = u[:, 1, :]
    u[:, :, 0] = -u[:, :, 1]
    u[:, :, -1] = u[:, :, -2] if top_wall_bc_type == 'neumann' else -u[:, :, -2]
    
    # V-velocity
    v[0, :, :] = v[-2, :, :]
    v[-1, :, :] = v[1, :, :]
    v[:, 0, :] = v[:, -1, :]
    v[:, :, 0] = -v[:, :, 1]
    v[:, :, -1] = v[:, :, -2] if top_wall_bc_type == 'neumann' else -v[:, :, -2]
    
    # W-velocity
    w[0, :, :] = w[-2, :, :]
    w[-1, :, :] = w[1, :, :]
    w[:, 0, :] = w[:, -2, :]
    w[:, -1, :] = w[:, 1, :]
    w[:, :, 0] = 0.0
    w[:, :, -1] = 0.0

# Test parameters
nx, ny, nz = 8, 8, 16
Lx, Ly, Lz = 1.0, 1.0, 1.0
dx, dy = Lx/nx, Ly/ny

# Generate grid
gamma = 1.5
z_f, z_c, dz_f, dz_c = generate_grid(gamma, nz, Lz, device='cpu', stretching_type='bottom')

print("="*60)
print("TEST: Check divergence at each cell for Dirichlet pressure BC")
print("="*60)

# Initialize FFT solver with Neumann top wall BC (free-slip velocity → Dirichlet pressure)
fft_data = initialize_fft_solver(nx, ny, nz, dx, dy, dz_c, dz_f, top_wall_bc_type='neumann')

# Create velocity field with known divergence
u = torch.zeros(nx+1, ny+2, nz+2)
v = torch.zeros(nx+2, ny+1, nz+2)
w = torch.zeros(nx+2, ny+2, nz+1)

# Simple test: constant divergence field
u[1:nx+1, 1:ny+1, 1:nz+1] = torch.randn(nx, ny, nz) * 0.01
v[1:nx+1, 1:ny+1, 1:nz+1] = torch.randn(nx, ny, nz) * 0.01
w[1:nx+1, 1:ny+1, 1:nz] = torch.randn(nx, ny, nz-1) * 0.01

# Apply BCs
apply_bc_all(u, v, w, top_wall_bc_type='neumann')

# Compute initial divergence
div_initial = compute_divergence(u, v, w, nx, ny, nz, dx, dy, dz_f)
print(f"\nInitial divergence:")
print(f"  Overall: min={div_initial.min():.6e}, max={div_initial.max():.6e}")
print(f"  Top cell (k={nz}): {div_initial[0, 0, -1]:.6e}")
print(f"  Second from top (k={nz-1}): {div_initial[0, 0, -2]:.6e}")

# Solve Poisson
dt = 0.001
p = solve_poisson_fft(div_initial / dt, fft_data)

print(f"\nPressure field:")
print(f"  Top cell (k={nz}): {p[1, 1, nz]:.6e}")
print(f"  Second from top (k={nz-1}): {p[1, 1, nz-1]:.6e}")
print(f"  Ghost (k={nz+1}): {p[1, 1, nz+1]:.6e}")

# Project velocity
u_proj, v_proj, w_proj = project_velocity(u, v, w, p, nx, ny, nz, dx, dy, dz_c, dz_f, dt)

# Apply BCs again
apply_bc_all(u_proj, v_proj, w_proj, top_wall_bc_type='neumann')

# Compute final divergence
div_final = compute_divergence(u_proj, v_proj, w_proj, nx, ny, nz, dx, dy, dz_f)
print(f"\nFinal divergence:")
print(f"  Overall: min={div_final.min():.6e}, max={div_final.max():.6e}")
print(f"  Top cell (k={nz}): {div_final[0, 0, -1]:.6e}")
print(f"  Second from top (k={nz-1}): {div_final[0, 0, -2]:.6e}")

print(f"\nDivergence reduction:")
print(f"  Top cell: {div_initial[0, 0, -1]:.6e} → {div_final[0, 0, -1]:.6e}")
print(f"  Second from top: {div_initial[0, 0, -2]:.6e} → {div_final[0, 0, -2]:.6e}")

if div_final.abs().max() < 1e-12:
    print("\n✓ SUCCESS")
else:
    print(f"\n✗ FAILURE: max divergence = {div_final.abs().max():.6e}")
    # Find which cell has max divergence
    max_idx = torch.argmax(div_final.abs())
    i, j, k = torch.unravel_index(max_idx, div_final.shape)
    print(f"  Max divergence at cell ({i}, {j}, {k})")
