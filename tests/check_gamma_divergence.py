import yaml
import torch
import numpy as np
import os
from solver import ChannelFlow
from utils import compute_divergence

def check_gamma():
    # Load base config
    if not os.path.exists('config.yaml'):
        print("Error: config.yaml not found")
        return

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    gammas = [2.4, 2.5, 2.6]
    
    print("==================================================")
    print("Checking divergence for gamma values:", gammas)
    print("==================================================")

    # Ensure output directory exists for temp configs
    os.makedirs('temp_configs', exist_ok=True)

    for g in gammas:
        print(f"\n>>> Testing gamma = {g} <<<")
        
        # Update config
        config['flow']['gamma'] = g
        
        # Ensure we don't overwrite main results
        config['output']['results_folder'] = f'results_gamma_test_{g}'
        
        # Write temp config
        temp_config_name = f'temp_configs/config_gamma_{g}.yaml'
        with open(temp_config_name, 'w') as f:
            yaml.dump(config, f)
            
        try:
            # Initialize solver
            # This will trigger grid generation and initial projection
            # We use a try-except block to catch any errors during initialization (e.g. singular matrix)
            solver = ChannelFlow(temp_config_name)
            
            # Compute divergence manually to be sure
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
                
        except Exception as e:
            print(f"  RESULT: FAILED with error: {e}")
            import traceback
            traceback.print_exc()
            
    print("\n==================================================")
    print("Done.")

if __name__ == "__main__":
    check_gamma()
