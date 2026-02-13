import numpy as np
import sys
sys.path.append('..')

from utils import generate_grid

# Create a 1D non-uniform grid (z-direction only)
nz = 8
Lz = 2.0
gamma = 1.5

# Generate stretched grid
z_f, z_c, dz_f, dz_c = generate_grid(gamma, nz, Lz)

print("="*60)
print("1D POISSON MATRIX TEST (Z-direction, Non-uniform Grid)")
print("="*60)
print(f"\nGrid: nz = {nz}")
print(f"Domain: Lz = {Lz}")
print(f"Stretching: gamma = {gamma}")

print(f"\nCell centers z_c:")
for k in range(nz+2):
    print(f"  z_c[{k}] = {z_c[k]:.6f}")

print(f"\nCell faces z_f:")
for k in range(nz+1):
    print(f"  z_f[{k}] = {z_f[k]:.6f}")

print(f"\nCell center spacings dz_c:")
for k in range(nz+1):
    print(f"  dz_c[{k}] = {dz_c[k]:.6f}  (distance from center {k} to center {k+1})")

print(f"\nCell widths dz_f:")
for k in range(nz):
    print(f"  dz_f[{k}] = {dz_f[k]:.6f}  (width of cell {k+1})")

# Build 1D Laplacian matrix with Neumann BCs
# User's formula:
# d²p/dz² = 2/(dz[k]*(dz[k]+dz[k-1]))*p[k+1] 
#         - (2/(dz[k]*(dz[k]+dz[k-1])) + 2/(dz[k-1]*(dz[k]+dz[k-1])))*p[k]
#         + 2/(dz[k-1]*(dz[k]+dz[k-1]))*p[k-1]
# where dz[k] = dz_c[k] (distance from center k to center k+1)

N = nz
A = np.zeros((N, N))

for k in range(1, nz+1):  # k = 1, 2, ..., nz
    idx = k - 1  # Matrix index (0-based)
    
    diag = 0.0
    
    # Coefficient for p[k-1]
    if k > 1:
        idx_down = k - 2
        denom = dz_c[k-1] * (dz_c[k] + dz_c[k-1])
        coeff_down = 2.0 / denom
        A[idx, idx_down] = coeff_down
        diag -= coeff_down
    
    # Coefficient for p[k+1]
    if k < nz:
        idx_up = k
        denom = dz_c[k] * (dz_c[k] + dz_c[k-1])
        coeff_up = 2.0 / denom
        A[idx, idx_up] = coeff_up
        diag -= coeff_up
    
    # Diagonal
    A[idx, idx] = diag

print(f"\n{'='*60}")
print("MATRIX ANALYSIS")
print(f"{'='*60}")

print(f"\nMatrix:")
np.set_printoptions(precision=4, suppress=True, linewidth=120)
print(A)

# Check symmetry
is_symmetric = np.allclose(A, A.T, atol=1e-10)
max_asymmetry = np.max(np.abs(A - A.T))

print(f"\nSymmetry check:")
print(f"  Is symmetric: {is_symmetric}")
print(f"  Max |A - A^T|: {max_asymmetry:.6e}")

if not is_symmetric:
    print(f"\n  Asymmetric entries:")
    for i in range(N):
        for j in range(i+1, N):
            if abs(A[i,j] - A[j,i]) > 1e-10:
                print(f"    A[{i},{j}] = {A[i,j]:.6e}, A[{j},{i}] = {A[j,i]:.6e}, diff = {A[i,j]-A[j,i]:.6e}")

# Row sums (should be ~0 for Neumann BCs with null space)
row_sums = np.sum(A, axis=1)
print(f"\nRow sums (should be ~0):")
print(f"  Min: {np.min(row_sums):.6e}")
print(f"  Max: {np.max(row_sums):.6e}")
print(f"  Mean: {np.mean(row_sums):.6e}")

# Eigenvalues
eigenvalues = np.linalg.eigvalsh(A)
print(f"\nEigenvalues:")
print(f"  Smallest: {eigenvalues[0]:.6e}")
print(f"  Largest: {eigenvalues[-1]:.6e}")
print(f"  Near-zero (<1e-10): {np.sum(np.abs(eigenvalues) < 1e-10)}")

print(f"\n{'='*60}")
if is_symmetric and eigenvalues[-1] < 1e-10 and eigenvalues[0] < 0:
    print("✓ SUCCESS: Matrix is symmetric and negative semidefinite!")
else:
    print("✗ FAILURE: Matrix has issues")
    if not is_symmetric:
        print("  - Not symmetric")
    if eigenvalues[-1] > 1e-10:
        print(f"  - Not negative semidefinite (max eigenvalue = {eigenvalues[-1]:.2e})")
print(f"{'='*60}")
