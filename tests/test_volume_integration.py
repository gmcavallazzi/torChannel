import torch
import sys
sys.path.append('/Users/giorgio.cavallazzi/Library/CloudStorage/OneDrive-City,UniversityofLondon/python_DNS_playground/DNS_homemade')

from utils import generate_grid, compute_bulk_velocity

torch.set_default_dtype(torch.float64)

print("="*90)
print("DIAGNOSTIC TEST: Volume Integration Consistency")
print("="*90)

# Test with multiple grid configurations
test_configs = [
    {"name": "Uniform grid", "nx": 8, "ny": 8, "nz": 8, "gamma": 0.0, "Lx": 2.0, "Ly": 1.0, "Lz": 2.0},
    {"name": "Stretched grid (gamma=1.0)", "nx": 8, "ny": 8, "nz": 16, "gamma": 1.0, "Lx": 2.67, "Ly": 0.8, "Lz": 2.0},
    {"name": "Stretched grid (gamma=2.5, actual config)", "nx": 64, "ny": 64, "nz": 64, "gamma": 2.5, "Lx": 2.67, "Ly": 0.8, "Lz": 2.0},
]

all_passed = True

for config in test_configs:
    print(f"\n{'='*90}")
    print(f"Test: {config['name']}")
    print(f"{'='*90}")
    print(f"Grid: nx={config['nx']}, ny={config['ny']}, nz={config['nz']}, gamma={config['gamma']}")
    print(f"Domain: Lx={config['Lx']}, Ly={config['Ly']}, Lz={config['Lz']}")

    nx, ny, nz = config['nx'], config['ny'], config['nz']
    Lx, Ly, Lz = config['Lx'], config['Ly'], config['Lz']
    gamma = config['gamma']

    # Generate grid
    z_f, z_c, dz_f, dz_c = generate_grid(gamma, nz, Lz)

    dx = Lx / nx
    dy = Ly / ny

    # Test 1: Check that sum(dz_f) == Lz
    print(f"\nTest 1: Grid spacing consistency")
    print(f"  dz_f shape: {dz_f.shape}")
    print(f"  sum(dz_f): {torch.sum(dz_f).item():.15f}")
    print(f"  Lz:        {Lz:.15f}")
    diff_dz = torch.abs(torch.sum(dz_f) - Lz)
    print(f"  Difference: {diff_dz:.3e}")

    if diff_dz < 1e-12:
        print(f"  ✓ PASS: sum(dz_f) == Lz")
    else:
        print(f"  ✗ FAIL: sum(dz_f) != Lz")
        all_passed = False

    # Test 2: Check cell volume sum
    print(f"\nTest 2: Cell volume integration")
    cell_vol = (dx * dy * dz_f.view(1, 1, -1)).expand(nx, ny, nz)
    total_volume = Lx * Ly * Lz

    print(f"  cell_vol shape: {cell_vol.shape}")
    print(f"  sum(cell_vol):  {torch.sum(cell_vol).item():.15f}")
    print(f"  total_volume:   {total_volume:.15f}")
    diff_vol = torch.abs(torch.sum(cell_vol) - total_volume)
    print(f"  Difference: {diff_vol:.3e}")

    if diff_vol < 1e-12:
        print(f"  ✓ PASS: sum(cell_vol) == total_volume")
    else:
        print(f"  ✗ FAIL: sum(cell_vol) != total_volume")
        all_passed = False

    # Test 3: Test bulk velocity calculation with uniform field
    print(f"\nTest 3: Bulk velocity with uniform field")
    u_uniform = torch.ones(nx+1, ny+2, nz+2) * 2.5  # Uniform u = 2.5
    u_bulk = compute_bulk_velocity(u_uniform, cell_vol, total_volume)

    print(f"  Expected u_bulk: 2.5")
    print(f"  Computed u_bulk: {u_bulk.item():.15f}")
    diff_bulk = torch.abs(u_bulk - 2.5)
    print(f"  Difference: {diff_bulk:.3e}")

    if diff_bulk < 1e-12:
        print(f"  ✓ PASS: Bulk velocity correct for uniform field")
    else:
        print(f"  ✗ FAIL: Bulk velocity incorrect for uniform field")
        all_passed = False

    # Test 4: Test forcing correction
    print(f"\nTest 4: Forcing correction consistency")
    u_test = torch.randn(nx+1, ny+2, nz+2)
    u_bulk_before = compute_bulk_velocity(u_test, cell_vol, total_volume)

    U_target = 1.0
    dt = 0.01
    forcing = (U_target - u_bulk_before) / dt

    # Apply uniform forcing to interior
    u_test[1:nx+1, 1:ny+1, 1:nz+1] += dt * forcing

    u_bulk_after = compute_bulk_velocity(u_test, cell_vol, total_volume)

    print(f"  Target U_bulk: {U_target:.15f}")
    print(f"  Initial u_bulk: {u_bulk_before.item():.15f}")
    print(f"  After forcing: {u_bulk_after.item():.15f}")
    diff_forcing = torch.abs(u_bulk_after - U_target)
    print(f"  Difference from target: {diff_forcing:.3e}")

    if diff_forcing < 1e-12:
        print(f"  ✓ PASS: Forcing correction achieves target bulk velocity")
    else:
        print(f"  ✗ FAIL: Forcing correction does not achieve target")
        print(f"    This indicates volume integration inconsistency!")
        all_passed = False

    # Test 5: Check indexing - no ghost cells included
    print(f"\nTest 5: Ghost cell exclusion")
    u_ghost = torch.zeros(nx+1, ny+2, nz+2)
    u_ghost[1:nx+1, 1:ny+1, 1:nz+1] = 1.0  # Interior = 1
    # Ghost cells remain 0

    u_bulk_ghost = compute_bulk_velocity(u_ghost, cell_vol, total_volume)
    print(f"  Expected u_bulk (interior only): 1.0")
    print(f"  Computed u_bulk: {u_bulk_ghost.item():.15f}")
    diff_ghost = torch.abs(u_bulk_ghost - 1.0)
    print(f"  Difference: {diff_ghost:.3e}")

    if diff_ghost < 1e-12:
        print(f"  ✓ PASS: Ghost cells correctly excluded")
    else:
        print(f"  ✗ FAIL: Ghost cells may be included or interior cells excluded")
        all_passed = False

print(f"\n{'='*90}")
if all_passed:
    print("ALL TESTS PASSED ✓")
else:
    print("SOME TESTS FAILED ✗")
    print("\nPossible issues:")
    print("  1. Grid spacing dz_f does not sum to Lz")
    print("  2. Cell volume computation includes/excludes wrong cells")
    print("  3. Indexing mismatch in compute_bulk_velocity")
print(f"{'='*90}\n")
