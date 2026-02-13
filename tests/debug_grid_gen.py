import torch
import numpy as np
from utils import generate_grid

def debug_grid():
    print("========================================")
    print("DEBUGGING GRID GENERATION")
    print("========================================")
    
    gammas = [2.4, 2.5, 2.6]
    nz = 32
    Lz = 1.0
    device = 'cpu'
    
    for g in gammas:
        print(f"\n>>> Gamma = {g} <<<")
        try:
            z_f, z_c, dz_f, dz_c = generate_grid(g, nz, Lz, device=device)
            
            print(f"  z_f range: [{z_f.min().item():.6f}, {z_f.max().item():.6f}]")
            print(f"  dz_f stats: min={dz_f.min().item():.6e}, max={dz_f.max().item():.6e}, mean={dz_f.mean().item():.6e}")
            print(f"  dz_c stats: min={dz_c.min().item():.6e}, max={dz_c.max().item():.6e}, mean={dz_c.mean().item():.6e}")
            
            if torch.isnan(z_f).any() or torch.isnan(dz_f).any():
                print("  RESULT: NaNs detected in grid!")
            elif (dz_f <= 0).any():
                print("  RESULT: Negative or zero cell size detected!")
            else:
                print("  RESULT: Grid seems valid (no NaNs, positive cell sizes).")
                
            # Check stretching ratio
            ratios = dz_f[1:] / dz_f[:-1]
            print(f"  Max stretching ratio: {ratios.max().item():.4f}")
            print(f"  Min stretching ratio: {ratios.min().item():.4f}")
            
        except Exception as e:
            print(f"  RESULT: FAILED with error: {e}")

if __name__ == "__main__":
    debug_grid()
