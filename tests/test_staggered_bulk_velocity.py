import torch
import sys
sys.path.append('/Users/giorgio.cavallazzi/Library/CloudStorage/OneDrive-City,UniversityofLondon/python_DNS_playground/DNS_homemade')

from utils import generate_grid

torch.set_default_dtype(torch.float64)

print("="*90)
print("DIAGNOSTIC TEST: Staggered Grid Bulk Velocity Calculation")
print("="*90)

nx, ny, nz = 8, 8, 8
Lx, Ly, Lz = 2.0, 1.0, 2.0
gamma = 1.0

dx = Lx / nx
dy = Ly / ny

z_f, z_c, dz_f, dz_c = generate_grid(gamma, nz, Lz)

cell_vol = (dx * dy * dz_f.view(1, 1, -1)).expand(nx, ny, nz)
total_volume = Lx * Ly * Lz

print(f"\nStaggered grid in x:")
print(f"  nx = {nx} cells")
print(f"  u has {nx+1} x-faces (indices 0 to {nx})")
print(f"  cell_vol has shape (nx, ny, nz) = ({nx}, {ny}, {nz})")
print(f"  ")
print(f"  For cell i (i=0 to {nx-1}):")
print(f"    Left face: x = i * dx,  u-index = i")
print(f"    Right face: x = (i+1) * dx, u-index = i+1")
print(f"    Cell center: x = (i+0.5) * dx")

# Create test with linear profile in x
print(f"\n{'='*90}")
print("Test: Linear profile in x-direction")
print("="*90)

u_test = torch.zeros(nx+1, ny+2, nz+2)

# Set u at x-faces to linear profile: u(x) = x
for i in range(nx+1):
    x_face = i * dx
    u_test[i, :, :] = x_face

print(f"u at x-faces: {[u_test[i, 0, 0].item() for i in range(nx+1)]}")

# Method 1: Current implementation (right face values)
u_bulk_right = torch.sum(u_test[1:nx+1, 1:ny+1, 1:nz+1] * cell_vol) / total_volume
print(f"\nMethod 1 (current): Use right face u[1:{nx+1}]")
print(f"  u-values used: {[u_test[i, 0, 1].item() for i in range(1, nx+1)]}")
print(f"  u_bulk = {u_bulk_right.item():.6f}")

# Method 2: Use left face values
u_bulk_left = torch.sum(u_test[0:nx, 1:ny+1, 1:nz+1] * cell_vol) / total_volume
print(f"\nMethod 2 (left face): Use left face u[0:{nx}]")
print(f"  u-values used: {[u_test[i, 0, 1].item() for i in range(0, nx)]}")
print(f"  u_bulk = {u_bulk_left.item():.6f}")

# Method 3: Average to cell centers (CORRECT for staggered grid!)
u_cell_centers = 0.5 * (u_test[0:nx, 1:ny+1, 1:nz+1] + u_test[1:nx+1, 1:ny+1, 1:nz+1])
u_bulk_centered = torch.sum(u_cell_centers * cell_vol) / total_volume
print(f"\nMethod 3 (cell-centered): Average left and right faces")
print(f"  u-values used: {[u_cell_centers[i, 0, 0].item() for i in range(nx)]}")
print(f"  u_bulk = {u_bulk_centered.item():.6f}")

# Analytical result for u(x) = x over domain [0, Lx]
# Integral: ∫₀^Lx x dx / Lx = [x²/2]₀^Lx / Lx = Lx²/(2*Lx) = Lx/2
u_bulk_analytical = Lx / 2
print(f"\nAnalytical result for u(x) = x:")
print(f"  u_bulk = {u_bulk_analytical:.6f}")

print(f"\nErrors:")
print(f"  Method 1 (right): {abs(u_bulk_right.item() - u_bulk_analytical):.6e}")
print(f"  Method 2 (left):  {abs(u_bulk_left.item() - u_bulk_analytical):.6e}")
print(f"  Method 3 (centered): {abs(u_bulk_centered.item() - u_bulk_analytical):.6e}")

if abs(u_bulk_centered.item() - u_bulk_analytical) < 1e-12:
    print(f"\n✓ Method 3 (cell-centered) is CORRECT")
    if abs(u_bulk_right.item() - u_bulk_analytical) > 1e-6:
        print(f"✗ Method 1 (current implementation) is INCORRECT!")
        print(f"  This could explain why forcing ≠ utau²!")
else:
    print(f"\nAll methods have errors - likely discretization")

print("="*90)
