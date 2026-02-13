import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/Users/giorgio.cavallazzi/Library/CloudStorage/OneDrive-City,UniversityofLondon/python_DNS_playground/DNS_homemade')

from solver import ChannelFlow

torch.set_default_dtype(torch.float64)

print("Checking for edge discontinuities in periodic BCs...")

# Load solver and run a few steps
solver = ChannelFlow(config_file='config.yaml')

for i in range(10):
    solver.step_adams_bashforth2(solver.dt)

# Extract XY slices at mid-height
iz = solver.nz // 2
u_xy = solver.u[1:-1 , 1:-1, iz+1].numpy()
v_xy = solver.v[1:-1, 1:-1, iz+1].numpy()

# Check if left and right edges match (periodic BC)
print(f"\nu(xy) slice:")
print(f"  Left edge (x=0):   mean={u_xy[0, :].mean():.6f}, std={u_xy[0, :].std():.6f}")
print(f"  Right edge (x=Lx): mean={u_xy[-1, :].mean():.6f}, std={u_xy[-1, :].std():.6f}")
print(f"  Difference: {np.abs(u_xy[0, :] - u_xy[-1, :]).max():.3e}")

print(f"\nv(xy) slice:")
print(f"  Left edge (x=0):   mean={v_xy[0, :].mean():.6f}, std={v_xy[0, :].std():.6f}")
print(f"  Right edge (x=Lx): mean={v_xy[-1, :].mean():.6f}, std={v_xy[-1, :].std():.6f}")
print(f"  Difference: {np.abs(v_xy[0, :] - v_xy[-1, :]).max():.3e}")

# Create a plot showing edge comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# u field
im0 = axes[0, 0].imshow(u_xy.T, origin='lower', cmap='RdBu_r')
axes[0, 0].set_title('u(x,y) at mid-height')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('y')
plt.colorbar(im0, ax=axes[0, 0])

# u edge profiles
axes[0, 1].plot(u_xy[0, :], label='Left edge (x=0)', marker='o', markersize=3)
axes[0, 1].plot(u_xy[-1, :], label='Right edge (x=Lx)', marker='x', markersize=3)
axes[0, 1].set_title('u at edges (should overlap)')
axes[0, 1].set_xlabel('y index')
axes[0, 1].set_ylabel('u')
axes[0, 1].legend()
axes[0, 1].grid(True)

# v field
im1 = axes[1, 0].imshow(v_xy.T, origin='lower', cmap='RdBu_r')
axes[1, 0].set_title('v(x,y) at mid-height')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('y')
plt.colorbar(im1, ax=axes[1, 0])

# v edge profiles
axes[1, 1].plot(v_xy[0, :], label='Left edge (x=0)', marker='o', markersize=3)
axes[1, 1].plot(v_xy[-1, :], label='Right edge (x=Lx)', marker='x', markersize=3)
axes[1, 1].set_title('v at edges (should overlap)')
axes[1, 1].set_xlabel('y index')
axes[1, 1].set_ylabel('v')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('results/periodic_bc_check.png', dpi=150)
print(f"\nSaved edge comparison plot to results/periodic_bc_check.png")
