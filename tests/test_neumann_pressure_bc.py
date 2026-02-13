import torch
import sys
sys.path.append('..')

from projection_fft import initialize_fft_solver, solve_poisson_fft
from utils import generate_grid, compute_divergence
from projection import project_velocity

# Test parameters
nx, ny, nz = 32, 32, 32
Lx, Ly, Lz = 1.0, 1.0, 1.0
dx, dy = Lx/nx, Ly/ny

# Generate grid
gamma = 1.5
z_f, z_c, dz_f, dz_c = generate_grid(gamma, nz, Lz, device='cpu', stretching_type='bottom')

print("="*60)
print("TEST: Neumann Pressure BC (No-Slip Top Wall)")
print("="*60)

# Initialize FFT solver with Dirichlet top wall BC (no-slip velocity → Neumann pressure)
fft_data = initialize_fft_solver(nx, ny, nz, dx, dy, dz_c, dz_f, top_wall_bc_type='dirichlet')

# Create velocity field
u = torch.zeros(nx+1, ny+2, nz+2)
v = torch.zeros(nx+2, ny+1, nz+2)
w = torch.zeros(nx+2, ny+2, nz+1)

# Add some divergent flow
u[1:nx+1, 1:ny+1, 1:nz+1] = torch.randn(nx, ny, nz) * 0.1
v[1:nx+1, 1:ny+1, 1:nz+1] = torch.randn(nx, ny, nz) * 0.1
w[1:nx+1, 1:ny+1, 1:nz] = torch.randn(nx, ny, nz-1) * 0.1

# Compute initial divergence
div_initial = compute_divergence(u, v, w, nx, ny, nz, dx, dy, dz_f)
print(f"\nInitial divergence: max={div_initial.abs().max():.6e}")

# Solve Poisson
dt = 0.001
p = solve_poisson_fft(div_initial / dt, fft_data)

# Project velocity
u_proj, v_proj, w_proj = project_velocity(u, v, w, p, nx, ny, nz, dx, dy, dz_c, dz_f, dt)

# Compute final divergence
div_final = compute_divergence(u_proj, v_proj, w_proj, nx, ny, nz, dx, dy, dz_f)
print(f"Final divergence:   max={div_final.abs().max():.6e}")
print(f"Reduction factor: {(div_initial.abs().max() / div_final.abs().max()).item():.2e}")

if div_final.abs().max() < 1e-12:
    print("\n✓ SUCCESS: Divergence reduced to machine precision!")
else:
    print(f"\n✗ FAILURE: Divergence still {div_final.abs().max():.6e}")
