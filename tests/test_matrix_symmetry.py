import torch
import sys
sys.path.append('..')

from projection import build_poisson_matrix
from utils import generate_grid

# Small test case to debug
nx, ny, nz = 8, 8, 64
Lx, Ly, Lz = 0.1, 0.1, 2.0
dx, dy = Lx/nx, Ly/ny

gamma = 1.5
z_f, z_c, dz_f, dz_c = generate_grid(gamma, nz, Lz)

print("Building Poisson matrix...")
A = build_poisson_matrix(nx, ny, nz, dx, dy, dz_c, dz_f)

print(f"Matrix size: {A.shape}")

# Check symmetry
diff = A - A.T
max_diff = torch.max(torch.abs(diff))
print(f"\nMax |A - A^T|: {max_diff:.6e}")

if max_diff > 1e-12:
    print("\n✗ Matrix is NOT symmetric!")
    
    # Find the worst asymmetric entries
    asymmetric_indices = torch.nonzero(torch.abs(diff) > 1e-12)
    num_asymmetric = len(asymmetric_indices)
    print(f"Number of asymmetric entries: {num_asymmetric}")
    
    if num_asymmetric > 0:
        # Show first few asymmetric entries
        print("\nFirst 10 asymmetric entries:")
        for idx in range(min(10, num_asymmetric)):
            i, j = asymmetric_indices[idx]
            print(f"  A[{i},{j}] = {A[i,j]:.6e}, A[{j},{i}] = {A[j,i]:.6e}, diff = {diff[i,j]:.6e}")
        
        # Convert flat indices to (i,j,k) grid indices
        print("\nConverting to grid indices...")
        i_flat, j_flat = asymmetric_indices[0]
        i_flat, j_flat = i_flat.item(), j_flat.item()
        
        # Decode using: idx = (i-1) + (j-1)*nx + (k-1)*nx*ny
        def flat_to_ijk(idx):
            k = idx // (nx * ny)
            remainder = idx % (nx * ny)
            j = remainder // nx
            i = remainder % nx
            return i+1, j+1, k+1
        
        i1, j1, k1 = flat_to_ijk(i_flat)
        i2, j2, k2 = flat_to_ijk(j_flat)
        
        print(f"Grid point 1: (i={i1}, j={j1}, k={k1})")
        print(f"Grid point 2: (i={i2}, j={j2}, k={k2})")
        print(f"Spatial relationship: di={i2-i1}, dj={j2-j1}, dk={k2-k1}")
else:
    print("\n✓ Matrix is symmetric!")
