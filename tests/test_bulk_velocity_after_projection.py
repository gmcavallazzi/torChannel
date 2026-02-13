import torch
import sys
sys.path.append('/Users/giorgio.cavallazzi/Library/CloudStorage/OneDrive-City,UniversityofLondon/python_DNS_playground/DNS_homemade')

from solver import ChannelFlow
from utils import compute_bulk_velocity, compute_u_tau

torch.set_default_dtype(torch.float64)

print("="*90)
print("DIAGNOSTIC TEST: Bulk Velocity Before/After Full Timestep")
print("="*90)

# Create solver
print("\nInitializing simulation...")
solver = ChannelFlow(config_file='config.yaml')

h = solver.Lz / 2.0

print(f"\n{'='*90}")
print("Checking if bulk velocity is actually conserved through timestep")
print("="*90)
print(f"{'Step':>6} {'u_bulk_start':>15} {'u_bulk_end':>15} {'deviation':>15} {'forcing':>12}")
print("="*90)

for step in range(20):
    # Compute bulk velocity before timestep
    u_bulk_before = compute_bulk_velocity(solver.u, solver.cell_vol_ratio, solver.total_volume)

    # Apply timestep
    u_bulk_returned, forcing = solver.step_adams_bashforth2(solver.dt)

    # Compute bulk velocity after timestep (including projection!)
    u_bulk_after = compute_bulk_velocity(solver.u, solver.cell_vol_ratio, solver.total_volume)

    # Check deviation from target
    deviation = u_bulk_after - solver.U_bulk

    print(f"{step:6d} {u_bulk_before.item():15.12f} {u_bulk_after.item():15.12f} {deviation.item():15.6e} {forcing:12.6e}")

print("="*90)

print("\nAnalysis:")
print("  The returned 'u_bulk' is computed BEFORE forcing is applied.")
print("  But we need to check the ACTUAL bulk velocity after forcing+projection.")
print("  ")
print("  If deviation != 0:")
print("    → Bulk velocity is NOT conserved through the timestep!")
print("    → The projection step is changing the bulk velocity")
print("    → This would explain why forcing must keep growing")
print("  ")
print("  Expected: deviation should be ~0 (machine precision)")

print("="*90)
