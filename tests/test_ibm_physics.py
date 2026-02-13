import torch
import numpy as np
import yaml
from solver import ChannelFlow
from ibm import IBM_RKPM

def test_ibm_physics():
    print("========================================")
    print("IBM PHYSICS VERIFICATION (FORCE BALANCE)")
    print("========================================")
    
    # 1. Setup Config
    config = {
        'grid': {'nx': 32, 'ny': 32, 'nz': 32},
        'domain': {'Lx': 2.0, 'Ly': 1.0, 'Lz': 1.0},
        'flow': {'Re': 100.0, 'Re_tau': 180.0, 'U_bulk': 1.0, 'gamma': 1.3}, # Non-uniform grid
        'time': {'dt': 0.001, 'n_steps': 1, 'CFL_target': 0.5, 'scheme': 'FE'}, # Explicit Euler for clarity
        'initialization': {'type': 'uniform', 'perturbation_intensity': 0.0},
        'ibm': {
            'enabled': True,
            'obstacle_type': 'sphere',
            'sphere': {'center': [1.0, 0.5, 0.5], 'radius': 0.1, 'n_points': 1} # Single point for clear force check
        },
        'compute': {'device': 'cpu'},
        'output': {'results_folder': 'results_test_physics', 'n_out': 1, 'n_save': 1},
        'statistics': {'enabled': False}
    }
    
    # Write temp config
    with open('config_physics.yaml', 'w') as f:
        yaml.dump(config, f)
        
    # 2. Initialize Solver
    solver = ChannelFlow('config_physics.yaml')
    
    # 3. Override Initialization
    # Set velocity to ZERO everywhere
    solver.u.fill_(0.0)
    solver.v.fill_(0.0)
    solver.w.fill_(0.0)
    
    # Set Constant Forcing
    F_x = 1.0
    solver.forcing = F_x
    
    print(f"Initial State:")
    print(f"  Velocity u: 0.0")
    print(f"  Forcing Fx: {F_x}")
    print(f"  Lagrangian Point: {solver.ibm.x_lag[0]:.4f}, {solver.ibm.y_lag[0]:.4f}, {solver.ibm.z_lag[0]:.4f}")
    
    # 4. Perform One Time Step (Manual)
    # We want to see what the IBM calculates.
    dt = solver.dt
    
    # Predictor (Forward Euler with Forcing): u* = u^n + dt * (RHS + F)
    # Since u^n=0 and RHS(diffusion/advection)=0, u* = dt * F
    u_pred = solver.u + dt * solver.forcing
    
    print(f"\nPredictor Step (dt={dt}):")
    print(f"  u_pred (expected {dt*F_x:.6f}): {u_pred.mean().item():.6f}")
    
    # Interpolate to Lagrangian point
    u_lag = solver.ibm.interpolate(u_pred[:, 1:-1, 1:-1], 'u')
    print(f"  u_lag (interpolated): {u_lag[0].item():.6f}")
    
    # Compute IBM Force
    # f_lag = (u_target - u_lag) / dt = (0 - u_lag) / dt
    # Expected: (0 - dt*F) / dt = -F
    f_lag = (0.0 - u_lag) / dt
    
    print(f"\nIBM Force Calculation:")
    print(f"  f_lag computed: {f_lag[0].item():.6f}")
    print(f"  Expected f_lag: {-F_x:.6f}")
    
    error = abs(f_lag[0].item() - (-F_x))
    print(f"  Error: {error:.6e}")
    
    if error < 1e-2:
        print("\nSUCCESS: IBM force balances physical forcing!")
    else:
        print("\nFAILURE: IBM force mismatch.")
        
    # 5. Check Spread Force
    print("\nSpreading Force to Grid...")
    f_euler = solver.ibm.spread(f_lag, 'u')
    
    # Check peak spread force
    print(f"  Max grid force: {f_euler.max().item():.6f}")
    print(f"  Min grid force: {f_euler.min().item():.6f}")
    
    # Check total force (integral)
    # f_euler is (nx+1, ny, nz).
    # We need to integrate over the domain.
    # Volume of u-cells: dx * dy * dz_c (approx)
    # dz_c is (nz,).
    
    # Construct volume tensor for u-grid
    # u is staggered in X, but cell volumes are mostly determined by Z spacing in this channel flow
    # Use dz_f (cell heights) for volume integration, not dz_c (center distances)
    dz_f = solver.dz_f.cpu() # (nz,)
    # Expand to (nx+1, ny, nz)
    vol_u = (solver.dx * solver.dy * dz_f.view(1, 1, -1)).expand(solver.nx+1, solver.ny, solver.nz)
    
    total_grid_force = torch.sum(f_euler * vol_u).item()
    expected_total = f_lag[0].item() * solver.ibm.dS_t[0].item() * solver.ibm.epsilon_u[0].item()
    
    print(f"  Total Grid Force (Integral): {total_grid_force:.6e}")
    print(f"  Expected Total (F_lag * dS * eps): {expected_total:.6e}")

if __name__ == "__main__":
    test_ibm_physics()
