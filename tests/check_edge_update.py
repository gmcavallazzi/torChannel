import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/Users/giorgio.cavallazzi/Library/CloudStorage/OneDrive-City,UniversityofLondon/python_DNS_playground/DNS_homemade')

from solver import ChannelFlow

torch.set_default_dtype(torch.float64)

print("Checking if edge points are being updated...")

# Load solver and run a few steps
solver = ChannelFlow(config_file='config.yaml')

print(f"\nInitial state:")
print(f"u[0, 32, 32] = {solver.u[0, 32, 32]:.6e}")
print(f"u[nx, 32, 32] = {solver.u[solver.nx, 32, 32]:.6e}")
print(f"Difference: {abs(solver.u[0, 32, 32] - solver.u[solver.nx, 32, 32]):.6e}")

for i in range(5):
    solver.step_adams_bashforth2(solver.dt)
    
print(f"\nAfter 5 steps:")
print(f"u[0, 32, 32] = {solver.u[0, 32, 32]:.6e}")
print(f"u[nx, 32, 32] = {solver.u[solver.nx, 32, 32]:.6e}")
print(f"Difference: {abs(solver.u[0, 32, 32] - solver.u[solver.nx, 32, 32]):.6e}")

# Check what indices are being updated in project_velocity
print(f"\nChecking update range:")
print(f"nx = {solver.nx}")
print(f"project_velocity updates u[1:nx+1] = u[1:{solver.nx+1}]")
print(f"This updates indices 1 through {solver.nx}")
print(f"u[0] is set by BC: u[0] = u[nx] = u[{solver.nx}]")

# The issue might be that u[nx] gets updated but u[0] doesn't get the BC applied after
print(f"\nLet's check if BC is applied after projection...")
print(f"In step_adams_bashforth2:")
print(f"1. apply_bc_u() is called at START")
print(f"2. project_velocity updates u[1:nx+1]")
print(f"3. apply_bc_u/v/w() is called at END")

# Extract a profile to visualize
iz = solver.nz // 2
iy = solver.ny // 2

u_profile = solver.u[0:solver.nx+1, iy, iz].numpy()
print(f"\nu profile along x at mid-y, mid-z:")
print(f"First 5 points: {u_profile[:5]}")
print(f"Last 5 points: {u_profile[-5:]}")
print(f"u[0] should equal u[nx]: {u_profile[0]:.6e} vs {u_profile[-1]:.6e}")
