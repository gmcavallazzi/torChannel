import unittest
import numpy as np
import torch
import sys
import os
import yaml

# Add parent directory to path to import ibm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ibm import IBM_RKPM

class TestIBM(unittest.TestCase):
    def setUp(self):
        # Create a mock config
        self.config = {
            'ibm': {
                'obstacle_type': 'sphere',
                'sphere': {
                    'center': [1.0, 1.0, 1.0],
                    'radius': 0.2,
                    'n_points': 100
                }
            }
        }
        
        # Create a mock grid
        nx, ny, nz = 32, 32, 32
        Lx, Ly, Lz = 2.0, 2.0, 2.0
        dx = Lx / nx
        dy = Ly / ny
        
        x_c = np.linspace(dx/2, Lx - dx/2, nx)
        x_f = np.linspace(0, Lx, nx + 1)
        y_c = np.linspace(dy/2, Ly - dy/2, ny)
        y_f = np.linspace(0, Ly, ny + 1)
        
        # Uniform z grid for simplicity in test
        z_c = np.linspace(Lz/nz/2, Lz - Lz/nz/2, nz)
        z_f = np.linspace(0, Lz, nz + 1)
        
        self.grid_data = {
            'x_c': x_c, 'x_f': x_f,
            'y_c': y_c, 'y_f': y_f,
            'z_c': z_c, 'z_f': z_f,
            'dx': dx, 'dy': dy
        }
        
        self.device = 'cpu'
        self.ibm = IBM_RKPM(self.config, self.grid_data, device=self.device)

    def test_sphere_points(self):
        """Test if generated points are on the sphere surface"""
        x, y, z = self.ibm.x_lag, self.ibm.y_lag, self.ibm.z_lag
        center = self.config['ibm']['sphere']['center']
        radius = self.config['ibm']['sphere']['radius']
        
        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
        
        # Check if all points are at distance 'radius' from center
        self.assertTrue(np.allclose(dist, radius, atol=1e-5))
        
        # Check number of points
        self.assertEqual(len(x), self.config['ibm']['sphere']['n_points'])

    def test_interpolation_uniform(self):
        """Test interpolation of a uniform field"""
        # Create a uniform field u = 1.0
        u = torch.ones((self.ibm.nx+1, self.ibm.ny, self.ibm.nz), device=self.device)
        
        u_lag = self.ibm.interpolate(u, 'u')
        
        # Interpolated value should be 1.0 everywhere
        # Note: RKPM reproduces constants exactly (0th order consistency)
        self.assertTrue(torch.allclose(u_lag, torch.tensor(1.0), atol=1e-4))

    def test_interpolation_linear(self):
        """Test interpolation of a linear field u = x"""
        # u is at x-faces: 0, dx, 2dx...
        u = torch.zeros((self.ibm.nx+1, self.ibm.ny, self.ibm.nz), device=self.device)
        x_f = torch.tensor(self.grid_data['x_f'], device=self.device, dtype=torch.float32)
        
        # Broadcast x_f to 3D
        u = x_f.view(-1, 1, 1).expand(-1, self.ibm.ny, self.ibm.nz)
        
        u_lag = self.ibm.interpolate(u, 'u')
        
        # Expected values: x coordinates of Lagrangian points
        x_lag_t = torch.tensor(self.ibm.x_lag, device=self.device, dtype=torch.float32)
        
        # RKPM with linear basis should reproduce linear functions exactly
        # But we use a compact support, so near boundaries it might degrade if support is cut.
        # Our sphere is in the center [1,1,1] of [2,2,2] domain, so it's far from boundaries.
        
        # Check error
        error = torch.abs(u_lag - x_lag_t)
        max_error = torch.max(error).item()
        print(f"Max linear interpolation error: {max_error}")
        
        self.assertTrue(max_error < 1e-3)

    def test_spread_conservation(self):
        """Test if spreading conserves force (approximately)"""
        # Apply a constant force at all lagrangian points
        f_lag = torch.ones(self.ibm.n_lag, device=self.device)
        
        # Spread to Euler grid
        f_euler = self.ibm.spread(f_lag, 'u')
        
        # Integral of f_euler should match sum of (f_lag * dS) ??
        # Force density f_euler is force per unit volume.
        # Integral f_euler dV = Total Force.
        # Total Force on Lag points = sum( f_lag * dS ) ?
        # Wait, f_lag in our formulation: F = (U - U_lag)/dt. This is acceleration (m/s^2) or force per unit mass?
        # In solver: rhs += f_euler. RHS of momentum eq is Force/Volume / rho? No, usually Force/Volume if conservative.
        # But here we are solving incompressible NS.
        # If f_lag is "force per unit volume" at the lag point? No, that doesn't make sense.
        # f_lag is usually "force per unit area" or just "force" depending on scaling.
        # In ibm.py: factor = dS / vol.
        # f_euler += f_lag * weights * dS / vol.
        # Integral f_euler dV = sum_grid ( f_euler * vol )
        # = sum_grid ( sum_lag ( f_lag * weights * dS / vol ) * vol )
        # = sum_lag ( f_lag * dS * sum_grid(weights) )
        # RKPM shape functions (weights) should satisfy partition of unity: sum_grid(weights) = 1.
        # So Integral f_euler dV = sum_lag ( f_lag * dS ).
        
        # Let's check this.
        
        total_force_lag = torch.sum(f_lag * self.ibm.dS_t)
        
        # Volume of cells
        # Uniform grid in this test
        vol = self.ibm.dx * self.ibm.dy * (self.ibm.z_c[1] - self.ibm.z_c[0])
        total_force_euler = torch.sum(f_euler) * vol
        
        print(f"Total Lag Force: {total_force_lag.item()}")
        print(f"Total Euler Force: {total_force_euler.item()}")
        
        # Should be close
        self.assertTrue(abs(total_force_lag.item() - total_force_euler.item()) / total_force_lag.item() < 0.05)

if __name__ == '__main__':
    unittest.main()
