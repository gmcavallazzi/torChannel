
import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ibm import IBM_RKPM

def test_ibm_functions():
    print("Testing IBM Functions...", flush=True)
    
    # Set default dtype to float64 to match expected solver behavior
    torch.set_default_dtype(torch.float64)
    device = torch.device('cpu')
    
    # 1. Setup Minimal Grid
    nx, ny, nz = 16, 16, 16
    Lx, Ly, Lz = 1.0, 1.0, 1.0
    dx = Lx / nx
    dy = Ly / ny
    
    # Grid arrays (numpy for init)
    x_c = np.linspace(dx/2, Lx - dx/2, nx)
    x_f = np.linspace(0, Lx, nx + 1)
    y_c = np.linspace(dy/2, Ly - dy/2, ny)
    y_f = np.linspace(0, Ly, ny + 1)
    
    # Uniform z-grid for simplicity
    z_f = np.linspace(0, Lz, nz + 1)
    z_c = 0.5 * (z_f[:-1] + z_f[1:])
    
    grid_data = {
        'x_c': x_c, 'x_f': x_f,
        'y_c': y_c, 'y_f': y_f,
        'z_c': z_c, 'z_f': z_f,
        'dx': dx, 'dy': dy
    }
    
    # Mock Config
    config = {
        'ibm': {
            'enabled': True,
            'obstacle_type': 'sphere',
            'sphere': {
                'center': [0.5, 0.5, 0.5],
                'radius': 0.1,
                'n_points': 100
            }
        }
    }
    
    # 2. Initialize IBM
    print("\nInitializing IBM_RKPM...", flush=True)
    ibm = IBM_RKPM(config, grid_data, device=device)
    
    print(f"Lagrangian points: {ibm.n_lag}")
    print(f"Internal dtype check: x_lag_t is {ibm.x_lag_t.dtype}")
    
    # 3. Test Interpolation (Uniform Field)
    print("\nTesting Interpolation (Uniform Field u=1.0)...", flush=True)
    # u shape: (nx+1, ny+2, nz+2) - with ghost cells
    u_euler = torch.ones((nx+1, ny+2, nz+2), device=device, dtype=torch.float64)
    
    u_lag = ibm.interpolate(u_euler, 'u')
    
    print(f"u_lag shape: {u_lag.shape}")
    print(f"u_lag dtype: {u_lag.dtype}")
    print(f"u_lag min/max: {u_lag.min():.6f} / {u_lag.max():.6f}")
    
    if not torch.allclose(u_lag, torch.tensor(1.0, dtype=torch.float64)):
        print("FAIL: Uniform interpolation failed!")
    else:
        print("PASS: Uniform interpolation correct.")
        
    # 4. Test Spreading
    print("\nTesting Spreading...", flush=True)
    f_lag = torch.ones(ibm.n_lag, device=device, dtype=torch.float64)
    
    try:
        f_euler = ibm.spread(f_lag, 'u')
        print(f"f_euler shape: {f_euler.shape}")
        print(f"f_euler dtype: {f_euler.dtype}")
        print("PASS: Spreading executed without error.")
    except Exception as e:
        print(f"FAIL: Spreading raised error: {e}")
        return

    # 5. Test Dtype Consistency
    if f_euler.dtype != torch.float64:
        print(f"FAIL: f_euler dtype mismatch! Expected float64, got {f_euler.dtype}")
    else:
        print("PASS: f_euler dtype is float64.")

if __name__ == "__main__":
    test_ibm_functions()
