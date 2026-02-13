"""
Test IBM visualization tools
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from ibm import Cube, visualize_ibm_setup
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

print("="*60)
print("Testing IBM Visualization")
print("="*60)

# Create grid
nx, ny, nz = 32, 32, 24
Lx, Ly, Lz = 4.0, 4.0, 2.0

x = torch.linspace(0, Lx, nx)
y = torch.linspace(0, Ly, ny)
z = torch.linspace(0, Lz, nz)

X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

# Create cube at center
cube = Cube(center=(Lx/2, Ly/2, Lz/2), size=0.4, device='cpu')

print(f"\nGrid: {nx} × {ny} × {nz}")
print(f"Domain: [{0}, {Lx}] × [{0}, {Ly}] × [{0}, {Lz}]")
print(f"Cube center: ({cube.center[0]:.2f}, {cube.center[1]:.2f}, {cube.center[2]:.2f})")
print(f"Cube size: {cube.size}")

# Get IBM mask
print("\nComputing IBM mask...")
dx = Lx / nx
dy = Ly / ny
dz = Lz / nz

mask_data = cube.get_ibm_mask(X, Y, Z, dx, dy, dz)

n_inside = mask_data['inside'].sum().item()
n_correct_x = mask_data['needs_correction_x'].sum().item()
n_correct_y = mask_data['needs_correction_y'].sum().item()
n_correct_z = mask_data['needs_correction_z'].sum().item()

print(f"  Points inside cube: {n_inside}")
print(f"  Points needing correction:")
print(f"    x-direction: {n_correct_x}")
print(f"    y-direction: {n_correct_y}")
print(f"    z-direction: {n_correct_z}")

# Create comprehensive visualization
print("\nGenerating visualization...")
fig = visualize_ibm_setup(cube, X, Y, Z, mask_data,
                          save_path='ibm_setup_visualization.png')

print("\n" + "="*60)
print("✓ Visualization complete!")
print("  File saved: ibm_setup_visualization.png")
print("="*60)
