import torch
import sys
sys.path.append('/Users/giorgio.cavallazzi/Library/CloudStorage/OneDrive-City,UniversityofLondon/python_DNS_playground/DNS_homemade')

from solver import ChannelFlow

# Set double precision
torch.set_default_dtype(torch.float64)

print("="*80)
print("Testing Periodic Boundary Conditions in X")
print("="*80)

# Create solver
solver = ChannelFlow(config_file='config.yaml')

print("\n1. Checking initial BCs after initialization...")
print("-"*80)

# Check u periodic BC
u_diff_periodic = torch.abs(solver.u[0, :, :] - solver.u[-1, :, :])
print(f"u[0] vs u[-1] (should be equal):") 
print(f"  Max difference: {torch.max(u_diff_periodic):.3e}")
print(f"  Mean difference: {torch.mean(u_diff_periodic):.3e}")

# Check v periodic BC  
v_diff_left = torch.abs(solver.v[0, :, :] - solver.v[-2, :, :])
v_diff_right = torch.abs(solver.v[-1, :, :] - solver.v[1, :, :])
print(f"\nv[0] vs v[-2] (should be equal):")
print(f"  Max difference: {torch.max(v_diff_left):.3e}")
print(f"v[-1] vs v[1] (should be equal):")
print(f"  Max difference: {torch.max(v_diff_right):.3e}")

# Check w periodic BC
w_diff_left = torch.abs(solver.w[0, :, :] - solver.w[-2, :, :])
w_diff_right = torch.abs(solver.w[-1, :, :] - solver.w[1, :, :])
print(f"\nw[0] vs w[-2] (should be equal):")
print(f"  Max difference: {torch.max(w_diff_left):.3e}")
print(f"w[-1] vs w[1] (should be equal):")
print(f"  Max difference: {torch.max(w_diff_right):.3e}")

print("\n2. Running two timesteps and checking BCs...")
print("-"*80)

for step in range(2):
    solver.step_adams_bashforth2(solver.dt)
    
    u_diff = torch.abs(solver.u[0, :, :] - solver.u[-1, :, :])
    v_diff_l = torch.abs(solver.v[0, :, :] - solver.v[-2, :, :])
    v_diff_r = torch.abs(solver.v[-1, :, :] - solver.v[1, :, :])
    w_diff_l = torch.abs(solver.w[0, :, :] - solver.w[-2, :, :])
    w_diff_r = torch.abs(solver.w[-1, :, :] - solver.w[1, :, :])
    
    print(f"\nAfter step {step+1}:")
    print(f"  u periodic BC error: {torch.max(u_diff):.3e}")
    print(f"  v periodic BC error: max({torch.max(v_diff_l):.3e}, {torch.max(v_diff_r):.3e})")
    print(f"  w periodic BC error: max({torch.max(w_diff_l):.3e}, {torch.max(w_diff_r):.3e})")

print("\n" + "="*80)
print("Test Complete")
print("="*80)
