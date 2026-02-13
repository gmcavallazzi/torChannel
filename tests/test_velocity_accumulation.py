import torch
import sys
sys.path.append('/Users/giorgio.cavallazzi/Library/CloudStorage/OneDrive-City,UniversityofLondon/python_DNS_playground/DNS_homemade')

from solver import ChannelFlow
from utils import compute_bulk_velocity, compute_u_tau

torch.set_default_dtype(torch.float64)

print("="*90)
print("DIAGNOSTIC TEST: Velocity Field Evolution")
print("="*90)

# Create solver
print("\nInitializing simulation...")
solver = ChannelFlow(config_file='config.yaml')

h = solver.Lz / 2.0

print(f"\n{'='*90}")
print(f"{'Step':>6} {'forcing':>12} {'utau²/h':>12} {'u_min':>12} {'u_max':>12} {'u_mean':>12} {'u[wall]':>12}")
print("="*90)

for step in range(20):
    # Apply timestep
    u_bulk, forcing = solver.step_adams_bashforth2(solver.dt)
    u_tau = compute_u_tau(solver.u, solver.z_c, solver.nu)
    forcing_expected = u_tau**2 / h

    # Get velocity statistics
    u_interior = solver.u[1:solver.nx+1, 1:solver.ny+1, 1:solver.nz+1]
    u_min = torch.min(u_interior)
    u_max = torch.max(u_interior)
    u_mean = torch.mean(u_interior)
    u_wall = torch.mean(solver.u[:, :, 1])  # First interior layer near wall

    print(f"{step:6d} {forcing:12.6e} {forcing_expected.item():12.6e} {u_min.item():12.6f} {u_max.item():12.6f} {u_mean.item():12.6f} {u_wall.item():12.6f}")

print("="*90)

print("\nObservations:")
print("  If u_mean is growing while u_wall stays constant:")
print("    → Velocity is accumulating in the interior, not reaching walls")
print("    → This would explain why forcing grows but utau doesn't")
print("  ")
print("  If u_max - u_min is growing:")
print("    → Profile is becoming more non-uniform")
print("  ")
print("  Possible causes:")
print("    1. Boundary conditions not being applied correctly after forcing")
print("    2. Projection step removing wall-normal momentum transfer")
print("    3. Viscous diffusion not strong enough to equilibrate profile")

print("="*90)
