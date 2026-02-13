import torch
import sys
sys.path.append('..')

from projection import build_poisson_matrix

# Test parameters
nx, ny, nz = 8, 8, 64
Lx, Ly, Lz = 0.1, 0.1, 2.0
dx, dy = Lx/nx, Ly/ny

# Generate grid
from utils import generate_grid
gamma = 1.5
z_f, z_c, dz_f, dz_c = generate_grid(gamma, nz, Lz)

print("Building Poisson matrix...")
A = build_poisson_matrix(nx, ny, nz, dx, dy, dz_c, dz_f)

print(f"Matrix size: {A.shape}")
print(f"Matrix is symmetric: {torch.allclose(A, A.T)}")

# Check if matrix is singular
eigenvalues = torch.linalg.eigvalsh(A)
print(f"\nEigenvalue analysis:")
print(f"  Smallest eigenvalue: {eigenvalues[0]:.6e}")
print(f"  Largest eigenvalue:  {eigenvalues[-1]:.6e}")
print(f"  Number of near-zero eigenvalues (<1e-10): {torch.sum(torch.abs(eigenvalues) < 1e-10).item()}")

if torch.abs(eigenvalues[0]) < 1e-10:
    print("\n✗ PROBLEM: Matrix is singular (has zero eigenvalue)!")
    print("   This is expected with Neumann BCs - pressure defined up to constant.")
    print("   Need to fix pressure at one point to make matrix non-singular.")
else:
    condition_number = eigenvalues[-1] / eigenvalues[0]
    print(f"\nCondition number: {condition_number:.2e}")
    if condition_number > 1e10:
        print("✗ PROBLEM: Matrix is very ill-conditioned!")
    elif condition_number > 1e6:
        print("⚠ WARNING: Matrix is poorly conditioned")
    else:
        print("✓ Matrix conditioning is acceptable")
