import torch
import sys
sys.path.append('..')

from projection import build_poisson_matrix, solve_poisson, project_velocity
from utils import compute_divergence, generate_grid

# Test parameters
nx, ny, nz = 4, 4, 8
Lx, Ly, Lz = 0.1, 0.1, 2.0
dx, dy = Lx/nx, Ly/ny

# Generate grid
gamma = 1.5
z_f, z_c, dz_f, dz_c = generate_grid(gamma, nz, Lz)

print("="*60)
print("TEST: Poisson Solver and Projection")
print("="*60)
print(f"Grid: nx={nx}, ny={ny}, nz={nz}")
print(f"Domain: Lx={Lx}, Ly={Ly}, Lz={Lz}")
print(f"Spacing: dx={dx:.6f}, dy={dy:.6f}, dz_min={torch.min(dz_f):.6f}")

# Build Poisson matrix
print("\nBuilding Poisson matrix...")
A = build_poisson_matrix(nx, ny, nz, dx, dy, dz_c, dz_f)
print(f"Matrix size: {A.shape}")

# Create a simple test velocity field with known divergence
# Initialize with zeros
u = torch.zeros(nx+1, ny+2, nz+2)
v = torch.zeros(nx+2, ny+1, nz+2)
w = torch.zeros(nx+2, ny+2, nz+1)

# Add a sinusoidal divergent field: u(x) = sin(2*pi*x/Lx)
# This gives div = du/dx = (2*pi/Lx) * cos(2*pi*x/Lx)
# This satisfies periodic BCs and has zero mean divergence (compatibility condition)
for i in range(nx+1):
    x = i * dx
    u[i, :, :] = torch.sin(torch.tensor(2 * torch.pi * x / Lx))

# Apply periodic BC in x
# u[0] and u[nx] are the periodic faces (x=0 and x=Lx). 
# The sin function already ensures u[0] = u[nx] = 0.
# No need to overwrite them with interior values.

print("\nTest case: u(x) = sin(2*pi*x/Lx), v=0, w=0")
print(f"Expected divergence: du/dx = (2*pi/Lx) * cos(2*pi*x/Lx)")
print(f"Max expected divergence: {2 * torch.pi / Lx:.6f}")

# Compute initial divergence
div_initial = compute_divergence(u, v, w, nx, ny, nz, dx, dy, dz_f)
print(f"Initial divergence: min={torch.min(div_initial):.6e}, max={torch.max(div_initial):.6e}, mean={torch.mean(div_initial):.6e}")

# Solve Poisson equation
dt = 0.0002
print(f"\nSolving Poisson equation with dt={dt}...")
p = solve_poisson(A, div_initial / dt, nx, ny, nz)
print(f"Pressure: min={torch.min(p):.6e}, max={torch.max(p):.6e}, mean={torch.mean(p):.6e}")

# Apply projection
print("\nApplying projection...")
u_proj, v_proj, w_proj = project_velocity(u, v, w, p, nx, ny, nz, dx, dy, dz_c, dz_f, dt)

# Compute final divergence
div_final = compute_divergence(u_proj, v_proj, w_proj, nx, ny, nz, dx, dy, dz_f)
print(f"Final divergence: min={torch.min(div_final):.6e}, max={torch.max(div_final):.6e}, mean={torch.mean(div_final):.6e}")

# Check if divergence was reduced
initial_div_max = torch.max(torch.abs(div_initial))
final_div_max = torch.max(torch.abs(div_final))
reduction_factor = initial_div_max / final_div_max if final_div_max > 0 else float('inf')

print("\n" + "="*60)
print("RESULTS:")
print("="*60)
print(f"Initial max |div|: {initial_div_max:.6e}")
print(f"Final max |div|:   {final_div_max:.6e}")
print(f"Reduction factor:  {reduction_factor:.2e}")

if final_div_max < 1e-6:
    print("\n✓ SUCCESS: Divergence reduced to machine precision!")
elif final_div_max < initial_div_max * 0.01:
    print("\n✓ PASS: Divergence reduced by >99%")
elif final_div_max < initial_div_max:
    print(f"\n⚠ PARTIAL: Divergence reduced but only by {100*(1-final_div_max/initial_div_max):.1f}%")
else:
    print("\n✗ FAILURE: Divergence NOT reduced!")
    print("The projection step is not working correctly.")
