
import torch
import yaml
import sys
import os

# Add parent directory to path to import solver
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver import ChannelFlow
from ibm import IBM_RKPM

def debug_solver():
    print("Loading config...", flush=True)
    with open('config_test.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Force CPU for debugging if needed, or keep as is
    # config['compute']['device'] = 'cpu' 
    
    print("Initializing Solver...", flush=True)
    solver = ChannelFlow('config_test.yaml')
    
    print(f"Grid shapes: u={solver.u.shape}, v={solver.v.shape}, w={solver.w.shape}")
    print(f"Grid limits: z_c range [{solver.z_c.min():.4f}, {solver.z_c.max():.4f}]")
    
    # Check Initial Condition
    print("\nChecking Initial Conditions...", flush=True)
    if torch.isnan(solver.u).any() or torch.isnan(solver.v).any() or torch.isnan(solver.w).any():
        print("ERROR: NaNs found in initial velocity fields!")
    else:
        print("Initial velocity fields are clean (no NaNs).")
        
    # Check IBM Init
    if hasattr(solver, 'ibm'):
        print("\nChecking IBM...", flush=True)
        print(f"IBM initialized. n_lag = {solver.ibm.n_lag}")
        print(f"Lagrangian points range: x[{solver.ibm.x_lag.min():.4f}, {solver.ibm.x_lag.max():.4f}]")
        
        # Test Interpolation
        print("Testing Interpolation...", flush=True)
        u_lag = solver.ibm.interpolate(solver.u, 'u')
        print(f"Interpolated u_lag shape: {u_lag.shape}")
        print(f"u_lag stats: min={u_lag.min():.4e}, max={u_lag.max():.4e}, mean={u_lag.mean():.4e}")
        
        if torch.isnan(u_lag).any():
            print("ERROR: NaNs in interpolated u_lag!")
            
        # Test Spreading
        print("Testing Spreading...", flush=True)
        f_lag = torch.ones_like(u_lag)
        f_euler = solver.ibm.spread(f_lag, 'u')
        print(f"Spread f_euler shape: {f_euler.shape}")
        print(f"f_euler stats: min={f_euler.min():.4e}, max={f_euler.max():.4e}")
        
        if torch.isnan(f_euler).any():
            print("ERROR: NaNs in spread force!")

        # Check shapes against solver arrays
        print(f"Solver u shape: {solver.u.shape}")
        print(f"Spread f_euler shape: {f_euler.shape}")
        
        # Verify slicing logic
        try:
            test_add = solver.u[:, 1:-1, 1:-1] + f_euler
            print("Slicing compatibility check: PASSED")
        except Exception as e:
            print(f"Slicing compatibility check: FAILED - {e}")

    # Check Projection
    print("\nChecking Initial Projection...", flush=True)
    solver.project_divergence_free()
    max_div = solver.compute_divergence()
    print(f"Max divergence after projection: {max_div:.4e}")
    
    if torch.isnan(solver.u).any():
        print("ERROR: NaNs found after projection!")

if __name__ == "__main__":
    debug_solver()
