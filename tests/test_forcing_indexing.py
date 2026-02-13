import torch
import sys
sys.path.append('/Users/giorgio.cavallazzi/Library/CloudStorage/OneDrive-City,UniversityofLondon/python_DNS_playground/DNS_homemade')

from utils import generate_grid, compute_bulk_velocity

torch.set_default_dtype(torch.float64)

print("="*90)
print("DIAGNOSTIC TEST: Forcing Application and Bulk Velocity Indexing")
print("="*90)

# Use actual simulation parameters
nx, ny, nz = 64, 64, 64
Lx, Ly, Lz = 2.67, 0.8, 2.0
gamma = 2.5

dx = Lx / nx
dy = Ly / ny

# Generate grid
z_f, z_c, dz_f, dz_c = generate_grid(gamma, nz, Lz)

# Create cell volumes
cell_vol = (dx * dy * dz_f.view(1, 1, -1)).expand(nx, ny, nz)
total_volume = Lx * Ly * Lz

print(f"\nGrid configuration:")
print(f"  nx={nx}, ny={ny}, nz={nz}")
print(f"  u array shape: ({nx+1}, {ny+2}, {nz+2})")
print(f"  cell_vol shape: {cell_vol.shape}")
print(f"  Interior region for forcing: u[1:{nx+1}, 1:{ny+1}, 1:{nz+1}]")
print(f"  Interior region for bulk: u[1:{nx+1}, 1:{ny+1}, 1:{nz+1}]")

# Test 1: Check that forcing application and bulk velocity use same region
print(f"\n{'='*90}")
print("Test 1: Verify forcing application region matches bulk velocity region")
print("="*90)

u_test = torch.zeros(nx+1, ny+2, nz+2)
u_test[1:nx+1, 1:ny+1, 1:nz+1] = 1.0  # Set interior to 1.0

# Compute bulk velocity
u_bulk_1 = compute_bulk_velocity(u_test, cell_vol, total_volume)
print(f"  Interior u = 1.0 everywhere")
print(f"  u_bulk = {u_bulk_1.item():.15f}")
print(f"  Expected: 1.0")

if abs(u_bulk_1.item() - 1.0) < 1e-12:
    print(f"  ✓ PASS: Bulk velocity correct")
else:
    print(f"  ✗ FAIL: Bulk velocity incorrect!")

# Test 2: Apply uniform forcing and check bulk velocity change
print(f"\n{'='*90}")
print("Test 2: Apply forcing and verify bulk velocity change")
print("="*90)

u_test2 = torch.randn(nx+1, ny+2, nz+2)
u_bulk_before = compute_bulk_velocity(u_test2, cell_vol, total_volume)

U_target = 1.5
dt = 0.01
forcing = (U_target - u_bulk_before) / dt

print(f"  u_bulk before: {u_bulk_before.item():.15f}")
print(f"  Target U_bulk: {U_target:.15f}")
print(f"  Forcing: {forcing.item():.6e}")

# Apply forcing to interior
u_test2[1:nx+1, 1:ny+1, 1:nz+1] += dt * forcing

u_bulk_after = compute_bulk_velocity(u_test2, cell_vol, total_volume)

print(f"  u_bulk after: {u_bulk_after.item():.15f}")
print(f"  Error from target: {abs(u_bulk_after.item() - U_target):.3e}")

if abs(u_bulk_after.item() - U_target) < 1e-12:
    print(f"  ✓ PASS: Forcing correctly achieves target bulk velocity")
else:
    print(f"  ✗ FAIL: Forcing does not achieve target!")
    print(f"    This indicates indexing mismatch between forcing and bulk velocity")

# Test 3: Check if ghost cells affect bulk velocity
print(f"\n{'='*90}")
print("Test 3: Verify ghost cells don't affect bulk velocity")
print("="*90)

u_test3a = torch.zeros(nx+1, ny+2, nz+2)
u_test3a[1:nx+1, 1:ny+1, 1:nz+1] = 2.0

u_test3b = torch.zeros(nx+1, ny+2, nz+2)
u_test3b[1:nx+1, 1:ny+1, 1:nz+1] = 2.0
u_test3b[:, :, 0] = 999.0  # Set ghost cells to large values
u_test3b[:, :, -1] = 999.0
u_test3b[0, :, :] = 999.0
u_test3b[:, 0, :] = 999.0
u_test3b[:, -1, :] = 999.0

u_bulk_3a = compute_bulk_velocity(u_test3a, cell_vol, total_volume)
u_bulk_3b = compute_bulk_velocity(u_test3b, cell_vol, total_volume)

print(f"  u_bulk (clean): {u_bulk_3a.item():.15f}")
print(f"  u_bulk (with ghost=999): {u_bulk_3b.item():.15f}")
print(f"  Difference: {abs(u_bulk_3a.item() - u_bulk_3b.item()):.3e}")

if abs(u_bulk_3a.item() - u_bulk_3b.item()) < 1e-12:
    print(f"  ✓ PASS: Ghost cells correctly excluded from bulk velocity")
else:
    print(f"  ✗ FAIL: Ghost cells affecting bulk velocity calculation!")

# Test 4: Check x-direction staggering
print(f"\n{'='*90}")
print("Test 4: Check staggered grid x-direction indexing")
print("="*90)

print(f"  u is staggered in x: shape ({nx+1}, {ny+2}, {nz+2})")
print(f"  There are {nx+1} x-faces for {nx} cells")
print(f"  For bulk velocity, we use u[1:{nx+1}, 1:{ny+1}, 1:{nz+1}]")
print(f"  This gives {nx} values in x (one per cell) ✓")
print(f"  ")
print(f"  But for forcing, we apply to u[1:{nx+1}, 1:{ny+1}, 1:{nz+1}]")
print(f"  This applies forcing to indices 1 through {nx} (nx values)")
print(f"  ")
print(f"  WAIT - for staggered u:")
print(f"  Indices 0 to {nx} give {nx+1} faces")
print(f"  u[1:{nx+1}] selects indices 1,2,...,{nx} which is {nx} faces")
print(f"  But u[0] and u[{nx}] are periodic copies, so we're missing u[{nx}]!")

# Verify this
u_test4 = torch.zeros(nx+1, ny+2, nz+2)
u_test4[:, :, :] = 1.0

# What if we don't include the last x-face?
u_manual_sum = torch.sum(u_test4[1:nx+1, 1:ny+1, 1:nz+1] * cell_vol)
u_manual_avg = u_manual_sum / total_volume

print(f"\n  Manual calculation:")
print(f"    sum(u[1:{nx+1}] * cell_vol) / total_vol = {u_manual_avg.item():.6f}")
print(f"    Expected for uniform u=1: 1.0")

error_4 = abs(u_manual_avg.item() - 1.0)
if error_4 < 1e-12:
    print(f"  ✓ Indexing appears correct")
else:
    print(f"  ✗ Indexing issue detected! Error = {error_4:.3e}")

print(f"\n{'='*90}\n")
