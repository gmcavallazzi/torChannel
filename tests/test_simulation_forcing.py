import torch
import sys
sys.path.append('/Users/giorgio.cavallazzi/Library/CloudStorage/OneDrive-City,UniversityofLondon/python_DNS_playground/DNS_homemade')

from solver import ChannelFlow
from utils import compute_bulk_velocity, compute_u_tau

torch.set_default_dtype(torch.float64)

print("="*90)
print("DIAGNOSTIC TEST: Forcing-utau Relation During Simulation")
print("="*90)

# Create a small test case for faster execution
print("\nInitializing small test simulation...")
solver = ChannelFlow(config_file='config.yaml')

n_steps = 100
print(f"\nRunning {n_steps} timesteps with detailed diagnostics...")
print("="*90)
print(f"{'Step':>6} {'u_bulk':>12} {'forcing':>12} {'u_tau':>12} {'utau²/h':>12} {'|forcing-utau²/h|':>15} {'rel_error%':>12}")
print("="*90)

h = solver.Lz / 2.0  # Half channel height

for step in range(n_steps):
    # Compute initial state
    u_bulk_before = compute_bulk_velocity(solver.u, solver.cell_vol_ratio, solver.total_volume)

    # Apply one timestep (this includes forcing application)
    u_bulk_after, forcing_applied = solver.step_adams_bashforth2(solver.dt)

    # Compute u_tau after the timestep
    u_tau = compute_u_tau(solver.u, solver.z_c, solver.nu)

    # Expected forcing from physics: forcing = utau² / h
    forcing_expected = u_tau**2 / h

    # Error
    error_abs = torch.abs(forcing_applied - forcing_expected)
    error_rel = 100.0 * error_abs / forcing_expected if forcing_expected > 1e-12 else 0.0

    # Print every 10 steps
    if step % 10 == 0 or step < 10:
        print(f"{step:6d} {u_bulk_after:12.6e} {forcing_applied:12.6e} {u_tau:12.6e} {forcing_expected.item():12.6e} {error_abs.item():15.6e} {error_rel.item():12.3f}")

print("="*90)

print("\nAnalysis:")
print("  If error is large and fluctuating, there's an inconsistency.")
print("  Possible causes:")
print("    1. u_bulk computed from u[1:nx+1, 1:ny+1, 1:nz+1] doesn't match what forcing acts on")
print("    2. u_tau computed incorrectly (but Test 1 showed it's correct for parabolic profile)")
print("    3. Projection step changes bulk velocity (but you said this shouldn't happen)")
print("    4. Ghost cell updates affect the computation inconsistently")
print("="*90)
