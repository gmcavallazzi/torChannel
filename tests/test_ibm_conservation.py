import torch
import unittest
import numpy as np
import sys
import os

# Add parent directory to path to import ibm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ibm import IBM_RKPM

class TestIBMConservation(unittest.TestCase):
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
        # For staggered grid, we need x_c, x_f, etc.
        # Uniform grid:
        # x_f (faces): 0, dx, 2dx, ...
        # x_c (centers): dx/2, 3dx/2, ...
        
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
        
        # Create IBM instance with a simple flat plate
        # Plate in YZ plane at x = 0.5
        # Note: __init__ will compute coefficients for the cube defined in config.
        # We will overwrite this immediately.
        self.ibm = IBM_RKPM(
            config={'ibm': {'obstacle_type': 'cube', 'cube': {'center': [0.5, 0.5, 0.5], 'dimensions': [0.2, 0.2, 0.2]}}},
            grid_data=grid_data,
            device=self.device
        )
        
        # Manually overwrite Lagrangian points to be a single point for clear testing
        self.ibm.n_lag = 1
        self.ibm.x_lag = np.array([0.5])
        self.ibm.y_lag = np.array([0.5])
        self.ibm.z_lag = np.array([0.5])
        self.ibm.dS = np.array([self.dy * self.dz]) # Arbitrary area
        
        # Re-initialize tensors
        self.ibm.x_lag_t = torch.tensor(self.ibm.x_lag, device=self.device, dtype=torch.float64)
        self.ibm.y_lag_t = torch.tensor(self.ibm.y_lag, device=self.device, dtype=torch.float64)
        self.ibm.z_lag_t = torch.tensor(self.ibm.z_lag, device=self.device, dtype=torch.float64)
        self.ibm.dS_t = torch.tensor(self.ibm.dS, device=self.device, dtype=torch.float64)
        
        # Re-compute coefficients for the new single point
        # u: x-face, y-center, z-center
        self.ibm.support_u = self.ibm.compute_rkpm_coefficients(self.ibm.x_f, self.ibm.y_c, self.ibm.z_c, 'u')

    def test_partition_of_unity(self):
        """Test if the sum of weights for a point equals 1 (Partition of Unity)."""
        # Get support for the single point
        support = self.ibm.support_u[0]
        weights = support['weights']
        
        # The weights in RKPM should sum to 1 approximating the integral of delta function?
        # In discrete form: Interpolation of 1 should give 1.
        # u_lag = sum(u_euler * weights)
        # If u_euler = 1 everywhere, u_lag should be 1.
        
        sum_weights = torch.sum(weights).item()
        print(f"Sum of weights: {sum_weights}")
        self.assertTrue(abs(sum_weights - 1.0) < 1e-2, f"Weights do not sum to 1: {sum_weights}")

    def test_interpolation_constant(self):
        """Test if interpolating a constant field returns the constant value."""
        # Create a constant field u = 5.0
        # Shape matches the grid passed to compute_rkpm_coefficients
        # For 'u', we used x_f, y_c, z_c
        nx = len(self.ibm.x_f)
        ny = len(self.ibm.y_c)
        nz = len(self.ibm.z_c)
        
        # Note: ibm.interpolate expects a tensor of shape (nx, ny, nz)
        # It uses the indices stored in support.
        
        field = torch.ones((nx, ny, nz), device=self.device, dtype=torch.float64) * 5.0
        
        val = self.ibm.interpolate(field, 'u')
        print(f"Interpolated value: {val.item()}")
        self.assertTrue(abs(val.item() - 5.0) < 1e-2, f"Interpolation failed: got {val.item()}, expected 5.0")

    def test_force_conservation(self):
        """Test if spreading a force conserves the total force (F = ma)."""
        # Define a Lagrangian acceleration a_lag
        a_lag_val = 10.0
        a_lag = torch.tensor([a_lag_val], device=self.device, dtype=torch.float64)
        
        # Spread to Eulerian grid
        # spread returns field_euler (Acceleration)
        field_euler = self.ibm.spread(a_lag, 'u')
        
        # Total Lagrangian Force F_lag = m_lag * a_lag
        # m_lag = rho * vol_lag
        # vol_lag approx dS * h
        # In our code, we assume rho=1 for simplicity in solver usually, but let's check the spreading logic.
        # spread logic: a_euler = a_lag * weights * (dS / vol_euler**(2/3))
        
        # Total Eulerian Force F_euler = sum(m_euler * a_euler)
        # m_euler = rho * vol_euler
        # F_euler = sum(vol_euler * a_euler)
        
        # We want F_euler approx F_lag
        # F_lag = vol_lag * a_lag
        
        # Let's calculate F_euler
        # field_euler is defined on the grid.
        # We need to sum (field_euler * vol_euler)
        
        # Get the indices where force is applied
        support = self.ibm.support_u[0]
        ix = support['ix']
        iy = support['iy']
        iz = support['iz']
        
        # In this uniform grid test, vol_euler is constant
        vol_euler = self.vol_element
        
        # Sum of acceleration on grid
        sum_a_euler = torch.sum(field_euler).item()
        
        # Total Force
        F_euler = sum_a_euler * vol_euler
        
        # Expected Force
        # We need to know what 'vol_lag' is implicitly defined as in the code.
        # In the code: factor = dS / vol**(2/3)
        # a_euler = a_lag * weights * dS / vol**(2/3)
        # F_euler = sum(vol * a_euler) = sum(vol * a_lag * weights * dS / vol**(2/3))
        #         = a_lag * dS * vol**(1/3) * sum(weights)
        #         = a_lag * dS * h_grid * 1.0
        #         = a_lag * (dS * h_grid)
        #         = a_lag * vol_lag_approx
        
        # So F_euler should be a_lag * dS * h_grid
        h_grid = self.vol_element**(1.0/3.0)
        F_lag_expected = a_lag_val * self.ibm.dS[0] * h_grid
        
        print(f"Total Eulerian Force: {F_euler}")
        print(f"Expected Lagrangian Force: {F_lag_expected}")
        
        # Check conservation
        self.assertTrue(abs(F_euler - F_lag_expected) < 0.1 * F_lag_expected, 
                        f"Force conservation failed: Euler {F_euler}, Lag {F_lag_expected}")

if __name__ == '__main__':
    unittest.main()
