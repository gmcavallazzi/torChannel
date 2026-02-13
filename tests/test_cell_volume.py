import torch
import sys
sys.path.append('/Users/giorgio.cavallazzi/Library/CloudStorage/OneDrive-City,UniversityofLondon/python_DNS_playground/DNS_homemade')

from utils import generate_grid

torch.set_default_dtype(torch.float64)

nx, ny, nz = 4, 4, 8
Lx, Ly, Lz = 2.0, 1.0, 2.0
dx = Lx / nx
dy = Ly / ny

z_f, z_c, dz_f, dz_c = generate_grid(gamma=1.0, nz=nz, Lz=Lz)

print("="*80)
print("Testing Cell Volume Computation")
print("="*80)
print(f"\nGrid: nx={nx}, ny={ny}, nz={nz}")
print(f"dz_f shape: {dz_f.shape}")
print(f"dz_f values: {dz_f}")

# Original loop version
cell_vol_loop = torch.zeros(nx, ny, nz)
for k in range(nz):
    cell_vol_loop[:, :, k] = dx * dy * dz_f[k]

print(f"\nLoop version:")
print(f"  cell_vol shape: {cell_vol_loop.shape}")
print(f"  cell_vol[0, 0, :]: {cell_vol_loop[0, 0, :]}")

# Vectorized version
cell_vol_vec = dx * dy * dz_f.view(1, 1, -1)

print(f"\nVectorized version:")
print(f"  cell_vol shape: {cell_vol_vec.shape}")
print(f"  cell_vol[0, 0, :]: {cell_vol_vec[0, 0, :]}")

# Check if they're equal
diff = torch.abs(cell_vol_loop - cell_vol_vec)
max_diff = torch.max(diff)

print(f"\nComparison:")
print(f"  Max difference: {max_diff:.2e}")

if max_diff < 1e-15:
    print("  ✓ PASS: Cell volume vectorization is correct")
else:
    print("  ✗ FAIL: Significant differences detected!")

# Now test if broadcasting works as expected when used in operations
u_test = torch.randn(nx+1, ny+2, nz+2)
interior_u = u_test[1:nx+1, 1:ny+1, 1:nz+1]

print(f"\nTesting usage in compute_bulk_velocity:")
print(f"  interior_u shape: {interior_u.shape}")
print(f"  cell_vol_loop shape: {cell_vol_loop.shape}")
print(f"  cell_vol_vec shape: {cell_vol_vec.shape}")

# Loop version
sum_loop = torch.sum(interior_u * cell_vol_loop)
print(f"  Sum with loop version: {sum_loop:.6e}")

# Vectorized version
sum_vec = torch.sum(interior_u * cell_vol_vec)
print(f"  Sum with vectorized version: {sum_vec:.6e}")

diff_sum = abs(sum_loop - sum_vec)
print(f"  Difference: {diff_sum:.2e}")

if diff_sum < 1e-12:
    print("  ✓ PASS: Broadcasting works correctly in operations")
else:
    print("  ✗ FAIL: Broadcasting issue detected!")

print("\n" + "="*80)
