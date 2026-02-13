import torch
import sys
sys.path.append('/Users/giorgio.cavallazzi/Library/CloudStorage/OneDrive-City,UniversityofLondon/python_DNS_playground/DNS_homemade')

from utils import generate_grid, compute_bulk_velocity

torch.set_default_dtype(torch.float64)

print("="*90)
print("DIAGNOSTIC TEST: Forcing Consistency on Staggered Grid")
print("="*90)

nx, ny, nz = 8, 8, 8
Lx, Ly, Lz = 2.0, 1.0, 2.0
gamma = 1.0

dx = Lx / nx
dy = Ly / ny

z_f, z_c, dz_f, dz_c = generate_grid(gamma, nz, Lz)

cell_vol = (dx * dy * dz_f.view(1, 1, -1)).expand(nx, ny, nz)
total_volume = Lx * Ly * Lz

print("\nTest: Does forcing achieve target bulk velocity with new compute_bulk_velocity?")
print("="*90)

u_test = torch.randn(nx+1, ny+2, nz+2)

# Compute initial bulk velocity (using new cell-centered method)
u_bulk_before = compute_bulk_velocity(u_test, cell_vol, total_volume)

U_target = 1.5
dt = 0.01
forcing = (U_target - u_bulk_before) / dt

print(f"Initial u_bulk: {u_bulk_before.item():.15f}")
print(f"Target U_bulk:  {U_target:.15f}")
print(f"Forcing:        {forcing.item():.6e}")

# Apply forcing to u[1:nx+1] (interior x-faces, right faces of cells)
u_test[1:nx+1, 1:ny+1, 1:nz+1] += dt * forcing

# Apply periodic BC: u[0] = u[nx]
u_test[0, :, :] = u_test[nx, :, :]

# Compute final bulk velocity
u_bulk_after = compute_bulk_velocity(u_test, cell_vol, total_volume)

print(f"Final u_bulk:   {u_bulk_after.item():.15f}")
print(f"Error:          {abs(u_bulk_after.item() - U_target):.3e}")

if abs(u_bulk_after.item() - U_target) < 1e-10:
    print("\n✓ PASS: Forcing achieves target with new compute_bulk_velocity")
else:
    print("\n✗ FAIL: Forcing does NOT achieve target!")
    print("  This means forcing application is inconsistent with bulk velocity calculation")

# Detailed analysis
print("\n" + "="*90)
print("Detailed Analysis: How does forcing affect cell-centered values?")
print("="*90)

# Check specific cells
print("\nCell 0 (near left boundary):")
print(f"  Uses u[0] (periodic) and u[1]")
print(f"  Before: u[0] = {u_test[0, 0, 1].item() - dt*forcing:.6f}, u[1] = {u_test[1, 0, 1].item() - dt*forcing:.6f}")
print(f"  Forcing applied to u[1] only")
print(f"  After: u[0] = {u_test[0, 0, 1].item():.6f} (updated by BC), u[1] = {u_test[1, 0, 1].item():.6f}")
print(f"  Expected increment in u_cell[0]: {0.5 * dt * forcing:.6e}")

print("\nCell 4 (middle):")
print(f"  Uses u[4] and u[5]")
print(f"  Forcing applied to both")
print(f"  Expected increment in u_cell[4]: {dt * forcing:.6e}")

print("\n" + "="*90)
