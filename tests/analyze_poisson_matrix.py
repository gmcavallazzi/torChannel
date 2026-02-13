import numpy as np
import sys

# Load the Poisson matrix
print("Loading Poisson matrix from CSV...")
A = np.loadtxt('results/poisson_matrix.csv', delimiter=',')

print(f"\n{'='*60}")
print(f"POISSON MATRIX ANALYSIS")
print(f"{'='*60}")

# Basic properties
N = A.shape[0]
print(f"\nMatrix size: {N}×{N}")
print(f"Grid dimensions: nx×ny×nz = {int(N**(1/3))}³ (assuming cubic)")

# Symmetry check
is_symmetric = np.allclose(A, A.T, atol=1e-10)
max_asymmetry = np.max(np.abs(A - A.T))
print(f"\nSymmetry check:")
print(f"  Is symmetric: {is_symmetric}")
print(f"  Max |A - A^T|: {max_asymmetry:.6e}")

# Sparsity
nnz = np.count_nonzero(np.abs(A) > 1e-12)
sparsity = 1 - nnz / (N * N)
print(f"\nSparsity:")
print(f"  Non-zero entries: {nnz} / {N*N}")
print(f"  Sparsity: {sparsity*100:.2f}%")
print(f"  Average non-zeros per row: {nnz/N:.1f}")

# Diagonal dominance
diag = np.diag(A)
off_diag_sum = np.sum(np.abs(A), axis=1) - np.abs(diag)
is_diag_dominant = np.all(np.abs(diag) >= off_diag_sum)
print(f"\nDiagonal dominance:")
print(f"  Is diagonally dominant: {is_diag_dominant}")
print(f"  Min |diagonal|: {np.min(np.abs(diag)):.6e}")
print(f"  Max |diagonal|: {np.max(np.abs(diag)):.6e}")
print(f"  Min off-diagonal sum: {np.min(off_diag_sum):.6e}")
print(f"  Max off-diagonal sum: {np.max(off_diag_sum):.6e}")

# Eigenvalues (for small matrices)
if N <= 100:
    print(f"\nEigenvalue analysis:")
    eigenvalues = np.linalg.eigvalsh(A)
    print(f"  Smallest eigenvalue: {eigenvalues[0]:.6e}")
    print(f"  Largest eigenvalue: {eigenvalues[-1]:.6e}")
    print(f"  Number of near-zero eigenvalues (<1e-10): {np.sum(np.abs(eigenvalues) < 1e-10)}")
    
    if eigenvalues[0] > 0:
        condition_number = eigenvalues[-1] / eigenvalues[0]
        print(f"  Condition number: {condition_number:.2e}")
    else:
        print(f"  Matrix is singular or indefinite!")
else:
    print(f"\n(Skipping eigenvalue analysis - matrix too large)")

# Row sum properties
row_sums = np.sum(A, axis=1)
print(f"\nRow sum properties:")
print(f"  Min row sum: {np.min(row_sums):.6e}")
print(f"  Max row sum: {np.max(row_sums):.6e}")
print(f"  Mean row sum: {np.mean(row_sums):.6e}")
print(f"  All row sums ~0: {np.allclose(row_sums, 0, atol=1e-10)}")

# Check structure
print(f"\nBand structure:")
# Find maximum bandwidth
for i in range(N):
    nz_cols = np.nonzero(np.abs(A[i, :]) > 1e-12)[0]
    if len(nz_cols) > 0:
        bandwidth = max(nz_cols) - min(nz_cols) + 1
        if i == 0:
            max_bandwidth = bandwidth
        else:
            max_bandwidth = max(max_bandwidth, bandwidth)
print(f"  Maximum bandwidth: {max_bandwidth}")

print(f"\n{'='*60}")
print("ASSESSMENT:")
print(f"{'='*60}")

issues = []
if not is_symmetric:
    issues.append(f"⚠ Non-symmetric (max diff: {max_asymmetry:.2e})")
if N <= 100 and eigenvalues[0] < 1e-10:
    issues.append("⚠ Matrix is singular/nearly singular")
if not is_diag_dominant:
    issues.append("⚠ Not diagonally dominant")
if sparsity < 0.9:
    issues.append(f"⚠ Low sparsity ({sparsity*100:.1f}%)")

if issues:
    print("\nPotential issues found:")
    for issue in issues:
        print(f"  {issue}")
else:
    print("\n✓ Matrix appears well-formed:")
    print(f"  - Sparse ({sparsity*100:.1f}% sparsity)")
    if is_symmetric:
        print(f"  - Symmetric")
    if is_diag_dominant:
        print(f"  - Diagonally dominant")
    if N <= 100 and eigenvalues[0] > 1e-10:
        print(f"  - Non-singular (cond = {condition_number:.2e})")
