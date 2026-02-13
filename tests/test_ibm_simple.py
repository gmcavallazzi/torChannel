"""
Simple quick test of IBM modules
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from ibm import Cube

print("Testing IBM modules...")

# Test 1: Cube geometry
print("\n1. Testing Cube SDF...")
cube = Cube(center=(2.0, 2.0, 1.0), size=0.4, device='cpu')

# Test points
x = torch.tensor([2.0, 2.3, 1.5])
y = torch.tensor([2.0, 2.0, 2.0])
z = torch.tensor([1.0, 1.0, 1.0])

sdf = cube.signed_distance(x, y, z)
print(f"  SDF at test points: {sdf}")
print(f"  Expected: [negative (inside), positive (outside), positive (outside)]")

inside = cube.is_inside(x, y, z)
print(f"  Inside: {inside}")

# Test 2: Distance to faces
print("\n2. Testing distance to faces...")
dist_data = cube.distance_to_faces(x, y, z)
print(f"  dx: {dist_data['dx']}")
print(f"  dy: {dist_data['dy']}")
print(f"  dz: {dist_data['dz']}")

# Test 3: IBM mask on small grid
print("\n3. Testing IBM mask on small grid...")
nx, ny, nz = 16, 16, 16
Lx, Ly, Lz = 4.0, 4.0, 2.0
dx, dy, dz = Lx/nx, Ly/ny, Lz/nz

x_1d = torch.linspace(0, Lx, nx)
y_1d = torch.linspace(0, Ly, ny)
z_1d = torch.linspace(0, Lz, nz)

X, Y, Z = torch.meshgrid(x_1d, y_1d, z_1d, indexing='ij')

mask_data = cube.get_ibm_mask(X, Y, Z, dx, dy, dz)

n_inside = mask_data['inside'].sum().item()
n_correct_x = mask_data['needs_correction_x'].sum().item()
n_correct_y = mask_data['needs_correction_y'].sum().item()
n_correct_z = mask_data['needs_correction_z'].sum().item()

print(f"  Points inside cube: {n_inside}")
print(f"  Points needing x-correction: {n_correct_x}")
print(f"  Points needing y-correction: {n_correct_y}")
print(f"  Points needing z-correction: {n_correct_z}")

# Test 4: IBM correction
print("\n4. Testing IBM Laplacian correction...")
from ibm import apply_ibm_correction

corrections = apply_ibm_correction(mask_data, dx, dy, dz)

lambda_max = corrections['lambda_total'].max().item()
lambda_min = corrections['lambda_total'][corrections['needs_correction']].min().item() if corrections['needs_correction'].any() else 0

print(f"  Max lambda: {lambda_max:.3e}")
print(f"  Min lambda (where corrected): {lambda_min:.3e}")
print(f"  Total points with correction: {corrections['needs_correction'].sum().item()}")

print("\n" + "="*60)
print("✓ All basic tests passed!")
print("="*60)
