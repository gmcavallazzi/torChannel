import numpy as np
import torch
from ibm import IBM_RKPM

def test_quadratic_rkpm():
    print("Testing Quadratic RKPM Implementation (3D)...")
    
    # 1. Setup Mock Config and Grid
    config = {
        'ibm': {
            'obstacle_type': 'sphere',
            'sphere': {
                'radius': 0.5,
                'center': [1.5, 1.5, 1.5],
                'n_points': 300 # Sufficient points
            }
        }
    }
    
    # Create a simple stretched grid
    nx, ny, nz = 32, 32, 32
    x_c = np.linspace(0, 3, nx)
    x_f = np.linspace(0, 3, nx+1) # Uniform
    y_c = np.linspace(0, 3, ny)
    y_f = np.linspace(0, 3, ny+1) # Uniform
    
    # Stretched Z grid (tanh)
    z_f = np.linspace(0, 3, nz+1)
    # Apply some stretching
    z_f = 3.0 * (np.tanh(2.0 * (z_f/3.0 - 0.5)) + np.tanh(1.0)) / (2.0 * np.tanh(1.0))
    z_c = 0.5 * (z_f[:-1] + z_f[1:])
    
    grid_data = {
        'x_c': x_c, 'x_f': x_f,
        'y_c': y_c, 'y_f': y_f,
        'z_c': z_c, 'z_f': z_f,
        'dx': x_c[1] - x_c[0],
        'dy': y_c[1] - y_c[0]
    }
    
    # 2. Initialize IBM
    ibm = IBM_RKPM(config, grid_data, device='cpu')
    
    # 3. Check Coefficients for one component (e.g., u)
    print("\nChecking u-velocity coefficients...")
    support_u = ibm.support_u
    
    # Analyze results
    total_negative = 0
    total_weights = 0
    
    for i, data in enumerate(support_u):
        wdt = data['wdt'].numpy()
        
        # Check for negative weights
        neg_count = (wdt < -1e-10).sum()
        total_negative += neg_count
        total_weights += len(wdt)
        
        if i == 0:
            print(f"Lag Point 0 wdt stats:")
            print(f"  Min: {wdt.min():.6e}")
            print(f"  Max: {wdt.max():.6e}")
            print(f"  Mean: {wdt.mean():.6e}")
            print(f"  Negatives: {neg_count}/{len(wdt)}")
            
    print(f"\nOverall Statistics:")
    print(f"  Total Weights: {total_weights}")
    print(f"  Total Negative: {total_negative} ({total_negative/total_weights*100:.2f}%)")
    
    if total_negative == 0:
        print("\nSUCCESS: No negative weights found!")
    else:
        print("\nWARNING: Negative weights found. This is expected for high-order RKPM but should be minimal.")

    # 4. Check Epsilon
    print("\nChecking Epsilon (Volume Regularization)...")
    epsilon_u = ibm.epsilon_u.cpu().numpy()
    print(f"Epsilon statistics:")
    print(f"  Min: {epsilon_u.min():.6f}")
    print(f"  Max: {epsilon_u.max():.6f}")
    print(f"  Mean: {epsilon_u.mean():.6f}")
    
    # Verify Partition of Unity
    # Correct check: Interpolate(Spread(1)) should be 1.0 at the markers.
    print("\nVerifying Partition of Unity (Interpolate(Spread(1)) == 1)...")
    
    # 1. Spread unity (with epsilon and dS)
    eps_with_dS = ibm.epsilon_u * ibm.dS_t
    unity_field = ibm._compute_unity_field(ibm.support_u, eps_with_dS)
    
    # 2. Interpolate back
    unity_at_lag = torch.zeros(ibm.n_lag, device=ibm.device, dtype=torch.float64)
    for i in range(ibm.n_lag):
        data = ibm.support_u[i]
        ix, iy, iz = data['ix'], data['iy'], data['iz']
        wdt = data['wdt']
        unity_at_lag[i] = torch.sum(unity_field[ix, iy, iz] * wdt)
        
    unity_np = unity_at_lag.cpu().numpy()
    print(f"Unity at Lag Stats:")
    print(f"  Min: {unity_np.min():.6f}")
    print(f"  Max: {unity_np.max():.6f}")
    print(f"  Mean: {unity_np.mean():.6f}")
    
    max_err = np.max(np.abs(unity_np - 1.0))
    if max_err < 1e-5:
        print(f"SUCCESS: Partition of unity satisfied (Max error: {max_err:.6e}).")
    else:
        print(f"WARNING: Partition of unity NOT satisfied (Max error: {max_err:.6e}).")

if __name__ == "__main__":
    test_quadratic_rkpm()
