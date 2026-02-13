import torch
import sys
sys.path.append('/Users/giorgio.cavallazzi/Library/CloudStorage/OneDrive-City,UniversityofLondon/python_DNS_playground/DNS_homemade')

from utils import generate_grid, compute_u_tau, compute_bulk_velocity

torch.set_default_dtype(torch.float64)

print("="*90)
print("DIAGNOSTIC TEST: Forcing = utau² Relation")
print("="*90)

# Use actual simulation parameters
nx, ny, nz = 64, 64, 64
Lx, Ly, Lz = 2.67, 0.8, 2.0
gamma = 2.5
Re = 2870.0
nu = 1.0 / Re

dx = Lx / nx
dy = Ly / ny

# Generate grid
z_f, z_c, dz_f, dz_c = generate_grid(gamma, nz, Lz)

print(f"\nPhysical parameters:")
print(f"  Re = {Re}")
print(f"  nu = {nu:.6e}")
print(f"  Lz = {Lz} (full channel height)")
print(f"  h = {Lz/2} (half-height)")

print(f"\nGrid near wall:")
print(f"  z_f[0] (wall) = {z_f[0].item():.6e}")
print(f"  z_c[1] (first interior) = {z_c[1].item():.6e}")
print(f"  dz_f[0] (first cell height) = {dz_f[0].item():.6e}")

# Test 1: Verify expected relation for parabolic profile
print(f"\n{'='*90}")
print("Test 1: Parabolic profile (analytical solution)")
print("="*90)

# For pressure-driven Poiseuille flow between z=0 and z=Lz:
# U(z) = (forcing / 2nu) * z * (Lz - z)
# U_max = (forcing / 2nu) * (Lz/2)² = forcing * Lz² / (8*nu)
# U_bulk = 2/3 * U_max = forcing * Lz² / (12*nu)
# At wall: du/dz|_wall = (forcing / 2nu) * Lz
# tau_wall = nu * du/dz|_wall = forcing * Lz / 2
# utau² = tau_wall = forcing * Lz / 2
# Therefore: forcing = 2 * utau² / Lz = utau² / h

U_bulk_target = 1.0
forcing_analytical = 12.0 * nu * U_bulk_target / Lz**2

print(f"\nFor U_bulk = {U_bulk_target}:")
print(f"  Analytical forcing = {forcing_analytical:.6e}")

# Create parabolic profile
u_parabolic = torch.zeros(nx+1, ny+2, nz+2)
for k in range(nz+2):
    z = z_c[k].item()
    u_val = (forcing_analytical / (2*nu)) * z * (Lz - z)
    u_parabolic[:, :, k] = u_val

# Apply no-slip BC (should already be satisfied, but let's enforce)
u_parabolic[:, :, 0] = -u_parabolic[:, :, 1]
u_parabolic[:, :, -1] = -u_parabolic[:, :, -2]

# Compute bulk velocity
cell_vol = (dx * dy * dz_f.view(1, 1, -1)).expand(nx, ny, nz)
total_volume = Lx * Ly * Lz
u_bulk_computed = compute_bulk_velocity(u_parabolic, cell_vol, total_volume)

print(f"  Computed U_bulk = {u_bulk_computed.item():.6e}")
print(f"  Error = {abs(u_bulk_computed.item() - U_bulk_target):.3e}")

# Compute utau from wall shear stress
u_tau_computed = compute_u_tau(u_parabolic, z_c, nu)

print(f"\nWall shear stress:")
print(f"  u_tau (computed) = {u_tau_computed.item():.6e}")

# Expected utau from forcing
h = Lz / 2
utau_expected = torch.sqrt(torch.tensor(forcing_analytical * h))
print(f"  u_tau (expected from forcing * h) = {utau_expected.item():.6e}")
print(f"  Error = {abs(u_tau_computed.item() - utau_expected.item()):.3e}")

# Check relation: forcing = utau² / h
forcing_from_utau = u_tau_computed**2 / h
print(f"\nRelation check:")
print(f"  forcing (analytical) = {forcing_analytical:.6e}")
print(f"  forcing (from u_tau² / h) = {forcing_from_utau.item():.6e}")
error_forcing = abs(forcing_analytical - forcing_from_utau.item())
print(f"  Error = {error_forcing:.3e}")

if error_forcing < 1e-6:
    print(f"  ✓ PASS: Relation forcing = utau² / h holds for parabolic profile")
else:
    print(f"  ✗ FAIL: Relation does not hold - issue with compute_u_tau!")

# Test 2: Check wall gradient calculation
print(f"\n{'='*90}")
print("Test 2: Wall gradient calculation")
print("="*90)

# For parabolic profile: du/dz|_wall = forcing * Lz / (2*nu)
dudz_wall_analytical = forcing_analytical * Lz / (2 * nu)
print(f"  Analytical du/dz|_wall = {dudz_wall_analytical:.6e}")

# Compute from first interior point
u_mean_bot = torch.mean(u_parabolic[:, :, 1])
dist = z_c[1].item()
dudz_wall_computed = u_mean_bot / dist
print(f"  Computed du/dz|_wall = {dudz_wall_computed.item():.6e}")
print(f"  (u[1] = {u_mean_bot.item():.6e}, dist = {dist:.6e})")
error_gradient = abs(dudz_wall_analytical - dudz_wall_computed.item())
print(f"  Error = {error_gradient:.3e}")

# This error comes from discretization - not exact for discrete profile
if error_gradient / dudz_wall_analytical < 0.05:  # 5% tolerance
    print(f"  ✓ PASS: Wall gradient computed within 5% (discretization error expected)")
else:
    print(f"  ✗ FAIL: Wall gradient computation has large error")

# Test 3: Check if issue is with indexing of u array
print(f"\n{'='*90}")
print("Test 3: u array indexing near wall")
print("="*90)

print(f"  u array shape: {u_parabolic.shape}")
print(f"  Interior region: u[1:{nx+1}, 1:{ny+1}, 1:{nz+1}]")
print(f"  u[:, :, 0] (ghost, below wall): mean = {torch.mean(u_parabolic[:, :, 0]):.6e}")
print(f"  u[:, :, 1] (first interior): mean = {torch.mean(u_parabolic[:, :, 1]):.6e}")
print(f"  u[:, :, -2] (last interior): mean = {torch.mean(u_parabolic[:, :, -2]):.6e}")
print(f"  u[:, :, -1] (ghost, above wall): mean = {torch.mean(u_parabolic[:, :, -1]):.6e}")

# Check no-slip BC
bc_error_bot = torch.mean(torch.abs(u_parabolic[:, :, 0] + u_parabolic[:, :, 1]))
bc_error_top = torch.mean(torch.abs(u_parabolic[:, :, -1] + u_parabolic[:, :, -2]))
print(f"\nNo-slip BC check:")
print(f"  |u[0] + u[1]| = {bc_error_bot:.3e}")
print(f"  |u[-1] + u[-2]| = {bc_error_top:.3e}")

if bc_error_bot < 1e-12 and bc_error_top < 1e-12:
    print(f"  ✓ PASS: No-slip BC correctly applied")
else:
    print(f"  ✗ FAIL: No-slip BC not correct")

print(f"\n{'='*90}\n")
