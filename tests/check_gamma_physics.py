import torch
import numpy as np
import yaml
import os
from solver import ChannelFlow
from utils import compute_divergence

def check_gamma_physics():
    print("========================================")
    print("CHECKING GAMMA DIVERGENCE (PHYSICS CONFIG)")
    print("========================================")
    
    gammas = [1.3, 2.4, 2.5, 2.6]
    
    # Ensure output directory exists
    os.makedirs('results_test_physics_gamma', exist_ok=True)
    
    for g in gammas:
        print(f"\n>>> Testing gamma = {g} <<<")
        
        # 1. Setup Config (based on test_ibm_physics.py)
        config = {
            'grid': {'nx': 32, 'ny': 32, 'nz': 32},
            'domain': {'Lx': 2.0, 'Ly': 1.0, 'Lz': 1.0},
            'flow': {'Re': 100.0, 'Re_tau': 180.0, 'U_bulk': 1.0, 'gamma': g}, # Variable gamma
            'time': {'dt': 0.001, 'n_steps': 1, 'CFL_target': 0.5, 'scheme': 'FE'},
            'initialization': {'type': 'uniform', 'perturbation_intensity': 0.0},
            'ibm': {
                'enabled': True,
                'obstacle_type': 'sphere',
                'sphere': {'center': [1.0, 0.5, 0.5], 'radius': 0.1, 'n_points': 1}
            },
            'compute': {'device': 'cpu'}, # Force CPU for deterministic check if needed, or keep auto
            'output': {'results_folder': f'results_test_physics_gamma/g_{g}', 'n_out': 1, 'n_save': 1},
            'statistics': {'enabled': False}
        }
        
        # Write temp config
        config_name = f'results_test_physics_gamma/config_gamma_{g}.yaml'
        with open(config_name, 'w') as f:
            yaml.dump(config, f)
            
        try:
            # 2. Initialize Solver
            # This triggers generate_grid -> initialize_flow -> project_velocity
            solver = ChannelFlow(config_name)
            
            # 3. Check Divergence
            div = compute_divergence(solver.u, solver.v, solver.w, 
                                     solver.nx, solver.ny, solver.nz,
                                     solver.dx, solver.dy, solver.dz_f)
            
            max_div = torch.max(torch.abs(div)).item()
            
            print(f"\n[Gamma {g} Analysis]")
            print(f"  Max Divergence: {max_div}")
            
            if np.isnan(max_div):
                print("  RESULT: NaN detected in divergence")
            elif max_div < 1e-9: 
                print("  RESULT: Divergence is effectively 0 (Success)")
            else:
                print(f"  RESULT: Divergence is non-zero (High)")

            # 4. Check Force Balance (RKPM)
            # Set velocity to ZERO everywhere
            solver.u.fill_(0.0)
            solver.v.fill_(0.0)
            solver.w.fill_(0.0)
            
            # Set Constant Forcing
            F_x = 1.0
            solver.forcing = F_x
            dt = solver.dt
            
            # Predictor: u* = dt * F
            u_pred = solver.u + dt * solver.forcing
            
            # Interpolate
            u_lag = solver.ibm.interpolate(u_pred[:, 1:-1, 1:-1], 'u')
            
            # Compute IBM Force
            # f_lag = (0 - u_lag) / dt
            f_lag = (0.0 - u_lag) / dt
            
            error = abs(f_lag[0].item() - (-F_x))
            print(f"  RKPM Force Error: {error:.6e}")
            
            if error < 1e-10:
                print("  RESULT: RKPM Force Balance SUCCESS")
            else:
                print("  RESULT: RKPM Force Balance FAILURE")
                
        except Exception as e:
            print(f"  RESULT: FAILED with error: {e}")
            # print traceback if needed
            # import traceback
            # traceback.print_exc()

if __name__ == "__main__":
    check_gamma_physics()
