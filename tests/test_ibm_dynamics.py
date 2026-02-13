import torch
import unittest
import numpy as np
import sys
import os

# Add parent directory to path to import ibm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ibm import IBM_RKPM

class TestIBMDynamics(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        self.nx, self.ny, self.nz = 32, 32, 32
        self.Lx, self.Ly, self.Lz = 1.0, 1.0, 1.0
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.dz = self.Lz / self.nz
        
        # Uniform grid for simplicity
        self.x_grid = np.linspace(0, self.Lx, self.nx+1)
        self.y_grid = np.linspace(0, self.Ly, self.ny+1)
        self.z_grid = np.linspace(0, self.Lz, self.nz+1)
        
        # Cell volumes (uniform)
        self.vol_element = self.dx * self.dy * self.dz
        
        # Construct grid_data dictionary
        x_f = self.x_grid
        y_f = self.y_grid
        z_f = self.z_grid
        
        x_c = (x_f[:-1] + x_f[1:]) / 2
        y_c = (y_f[:-1] + y_f[1:]) / 2
        z_c = (z_f[:-1] + z_f[1:]) / 2
        
        grid_data = {
            'x_c': x_c, 'x_f': x_f,
            'y_c': y_c, 'y_f': y_f,
            'z_c': z_c, 'z_f': z_f,
            'dx': self.dx, 'dy': self.dy
        }
        
        # Create IBM instance with the Cube configuration
        # We use the same config structure as in config.yaml
        cube_config = {
            'ibm': {
                'obstacle_type': 'cube',
                'cube': {
                    'center': [0.5, 0.5, 0.5],
                    'dimensions': [0.05, 0.05, 0.05], # Same as config.yaml
                    'n_points_per_face': 50
                }
            }
        }
        
        self.ibm = IBM_RKPM(
            config=cube_config,
            grid_data=grid_data,
            device=self.device
        )
        
        print(f"Initialized IBM with Cube. n_lag = {self.ibm.n_lag}")
        
        # We do NOT overwrite points manually. We use the generated cube.
        
        # Re-compute coefficients (already done in init, but let's be sure if we changed anything)
        # Actually init calls generate_lagrangian_points and compute_rkpm_coefficients.
        # So we are good.

    def test_ab2_stability(self):
        """Test stability of AB2 scheme with IBM force."""
        print("\n=== Testing AB2 Stability ===")
        nx = len(self.ibm.x_f)
        ny = len(self.ibm.y_c)
        nz = len(self.ibm.z_c)
        dt = 0.001
        
        # Initialize u = 1.0
        u = torch.ones((nx, ny, nz), device=self.device, dtype=torch.float64) * 1.0
        
        # Step 1: Euler (Initialization)
        u_lag = self.ibm.interpolate(u, 'u')
        f_lag = (0.0 - u_lag) / dt
        f_euler_prev = self.ibm.spread(f_lag, 'u')
        
        # Update u (Euler)
        u = u + dt * f_euler_prev
        
        print(f"Step 1 (Euler): u_lag_new = {self.ibm.interpolate(u, 'u').mean().item()}")
        
        # Step 2: AB2
        # Predictor (Advection/Diffusion would be here, but we assume 0 for isolation)
        # We only look at IBM force stability.
        
        u_lag = self.ibm.interpolate(u, 'u')
        f_lag = (0.0 - u_lag) / dt
        f_euler_curr = self.ibm.spread(f_lag, 'u')
        
        # AB2 Update: u_new = u + dt * (1.5 * f_curr - 0.5 * f_prev)
        u_new = u + dt * (1.5 * f_euler_curr - 0.5 * f_euler_prev)
        
        u_lag_new = self.ibm.interpolate(u_new, 'u')
        print(f"Step 2 (AB2): u_lag_new = {u_lag_new.mean().item()}")
        
        # Check for oscillation/explosion
        # If u_lag_new flips sign and grows (e.g. < -1.0), it's unstable.
        self.assertTrue(abs(u_lag_new.mean().item()) < 1.0, f"AB2 Instability detected! u_lag_new = {u_lag_new.mean().item()}")

if __name__ == '__main__':
    unittest.main()
