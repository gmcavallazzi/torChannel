#!/usr/bin/env python3
"""
Test script to verify IBM can load precomputed sphere points
"""

import torch
import yaml
import numpy as np
from utils import generate_grid

# Set double precision
torch.set_default_dtype(torch.float64)

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Generate grid data (needed for IBM initialization)
nx = config['grid']['nx']
ny = config['grid']['ny']
nz = config['grid']['nz']

Lx = config['domain']['Lx']
Ly = config['domain']['Ly']
Lz = config['domain']['Lz']

gamma = config['flow']['gamma']

# Create grid
dx = Lx / nx
dy = Ly / ny

x_c = torch.linspace(dx/2, Lx - dx/2, nx)
x_f = torch.linspace(0, Lx, nx+1)
y_c = torch.linspace(dy/2, Ly - dy/2, ny)
y_f = torch.linspace(0, Ly, ny+1)

z_f, z_c, dz_f, dz_c = generate_grid(gamma, nz, Lz, device='cpu')

grid_data = {
    'x_c': x_c,
    'x_f': x_f,
    'y_c': y_c,
    'y_f': y_f,
    'z_c': z_c,
    'z_f': z_f,
    'dx': dx,
    'dy': dy,
    'dz_c': dz_c,
    'dz_f': dz_f
}

print("="*70)
print("TESTING IBM LOADING WITH PRECOMPUTED POINTS")
print("="*70)

# Import IBM class
from ibm import IBM_RKPM

# Initialize IBM (should load the npz file)
ibm = IBM_RKPM(config, grid_data, device='cpu')

print("\n" + "="*70)
print("IBM INITIALIZATION SUCCESSFUL")
print("="*70)
print(f"Number of Lagrangian points: {ibm.n_lag}")
print(f"Sphere center: {ibm.center}")
print(f"Sphere radius: {ibm.radius}")
print(f"x_lag range: [{np.min(ibm.x_lag):.4f}, {np.max(ibm.x_lag):.4f}]")
print(f"y_lag range: [{np.min(ibm.y_lag):.4f}, {np.max(ibm.y_lag):.4f}]")
print(f"z_lag range: [{np.min(ibm.z_lag):.4f}, {np.max(ibm.z_lag):.4f}]")
print(f"dS sum: {np.sum(ibm.dS):.6f} (should be ≈ {4*np.pi*ibm.radius**2:.6f})")
print("="*70)
