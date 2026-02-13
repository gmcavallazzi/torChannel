import torch
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add current directory to path so we can import solver
sys.path.append(os.getcwd())

# Mock the dependencies of solver before importing it
# This is necessary because solver imports many things that might fail or require setup
sys.modules['operators'] = MagicMock()
sys.modules['utils'] = MagicMock()
sys.modules['initflow'] = MagicMock()
sys.modules['projection'] = MagicMock()
sys.modules['projection_fft'] = MagicMock()
sys.modules['statistics'] = MagicMock()
sys.modules['ibm'] = MagicMock()

# Now import ChannelFlow from solver
# We need to make sure we can import it even if some imports inside solver.py fail
# But we mocked the modules, so it should be fine.
from solver import ChannelFlow

class TestIMEXPredictorCorrector(unittest.TestCase):
    def setUp(self):
        # Create a mock instance of ChannelFlow
        # We bypass __init__ by creating a raw instance and setting attributes manually
        self.solver = ChannelFlow.__new__(ChannelFlow)
        
        # Set up attributes required by step_imex
        self.solver.nx, self.solver.ny, self.solver.nz = 4, 4, 4
        self.solver.dx, self.solver.dy = 0.1, 0.1
        self.solver.dz_c = torch.ones(5) * 0.1
        self.solver.dz_f = torch.ones(5) * 0.1
        self.solver.nu = 0.01
        self.solver.device = torch.device('cpu')
        
        # Velocities (Staggered grid)
        # u: (nx+1, ny+2, nz+2)
        # v: (nx+2, ny+1, nz+2)
        # w: (nx+2, ny+2, nz+1)
        self.shape_u = (5, 6, 6)
        self.shape_v = (6, 5, 6)
        self.shape_w = (6, 6, 5)
        
        self.solver.u = torch.zeros(self.shape_u)
        self.solver.v = torch.zeros(self.shape_v)
        self.solver.w = torch.zeros(self.shape_w)
        
        # RHS buffers
        self.solver.rhs_u_curr = None
        self.solver.rhs_u_prev = None
        self.solver.rhs_v_curr = None
        self.solver.rhs_v_prev = None
        self.solver.rhs_w_curr = None
        self.solver.rhs_w_prev = None
        
        # IBM
        self.solver.ibm = MagicMock()
        
        # Other attributes
        self.solver.cell_vol_ratio = 1.0
        self.solver.total_volume = 1.0
        self.solver.U_bulk = 1.0
        self.solver.solver_type = 'direct'
        self.solver.poisson_matrix = MagicMock()
        self.solver.fft_data = MagicMock()
        self.solver.current_step = 0
        
        # Mock methods of the instance
        self.solver.apply_bc_uvw = MagicMock()
        self.solver.compute_momentum_rhs_explicit = MagicMock(return_value=(
            torch.ones(self.shape_u) * 1.0, # rhs_u = 1
            torch.ones(self.shape_v) * 2.0, # rhs_v = 2
            torch.ones(self.shape_w) * 3.0  # rhs_w = 3
        ))
        
    @patch('solver.solve_implicit_diffusion_u')
    @patch('solver.solve_implicit_diffusion_v')
    @patch('solver.solve_implicit_diffusion_w')
    @patch('solver.compute_divergence')
    @patch('solver.solve_poisson')
    @patch('solver.project_velocity')
    @patch('solver.compute_bulk_velocity')
    @patch('solver.apply_bc_all')
    @patch('solver.advection_u')
    @patch('solver.advection_v')
    @patch('solver.advection_w')
    @patch('solver.diffusion_xy_u')
    @patch('solver.diffusion_xy_v')
    @patch('solver.diffusion_xy_w')
    def test_step_imex_pc_logic(self, mock_diff_w, mock_diff_v, mock_diff_u, 
                                mock_adv_w, mock_adv_v, mock_adv_u,
                                mock_apply_bc_all, mock_compute_bulk, mock_project, 
                                mock_solve_poisson, mock_div, 
                                mock_solve_imp_w, mock_solve_imp_v, mock_solve_imp_u):
        
        # Setup operator mocks to return Tensors with correct shapes
        mock_diff_u.return_value = torch.ones(self.shape_u) * 2.0
        mock_diff_v.return_value = torch.ones(self.shape_v) * 3.0
        mock_diff_w.return_value = torch.ones(self.shape_w) * 4.0
        
        mock_adv_u.return_value = torch.ones(self.shape_u) * 1.0
        mock_adv_v.return_value = torch.ones(self.shape_v) * 1.0
        mock_adv_w.return_value = torch.ones(self.shape_w) * 1.0
        
        # Resulting RHS explicit = diff - adv
        # rhs_u = 2 - 1 = 1.0
        # rhs_v = 3 - 1 = 2.0
        # rhs_w = 4 - 1 = 3.0
        
        # Predictor is now always Euler: u_pred = u + dt * rhs_explicit
        # u = 0, dt = 0.1, rhs = 1.0 => u_pred = 0.1
        
        mock_solve_imp_u.side_effect = lambda u, *args: u
        mock_solve_imp_v.side_effect = lambda v, *args: v
        mock_solve_imp_w.side_effect = lambda w, *args: w
        mock_compute_bulk.return_value = 1.0
        mock_project.return_value = (self.solver.u, self.solver.v, self.solver.w)
        mock_solve_poisson.return_value = torch.zeros((4, 4, 4))
        
        # Initialize p
        self.solver.p = torch.zeros((4, 4, 4))
        
        # Setup IBM mocks
        # interpolate returns 10.0
        self.solver.ibm.interpolate.return_value = torch.tensor(10.0)
        
        # spread returns force matching the slice shape
        # u slice: [:, 1:-1, 1:-1] -> (5, 4, 4)
        # v slice: [1:-1, :, 1:-1] -> (4, 5, 4)
        # w slice: [1:-1, 1:-1, :] -> (4, 4, 5)
        
        def spread_side_effect(f_lag, component):
            if component == 'u':
                return torch.ones((5, 4, 4)) * 5.0
            elif component == 'v':
                return torch.ones((4, 5, 4)) * 5.0
            elif component == 'w':
                return torch.ones((4, 4, 5)) * 5.0
            return None
            
        self.solver.ibm.spread.side_effect = spread_side_effect
        
        dt = 0.1
        
        # Run step_imex
        self.solver.step_imex(dt)
        
        # VERIFICATION 1: Predictor Step
        # The predictor velocity passed to interpolate should be:
        # u_pred = u_old + dt * rhs_explicit
        # u_old = 0, rhs_explicit = 1, dt = 0.1 => u_pred = 0.1
        # Note: u_pred is also passed through solve_implicit_diffusion, which we mocked as identity.
        
        # Check calls to interpolate
        # We expect 3 calls (u, v, w)
        self.assertEqual(self.solver.ibm.interpolate.call_count, 3)
        
        # Check arguments of first call (u)
        # interpolate(u_pred[:, 1:-1, 1:-1], 'u')
        args, _ = self.solver.ibm.interpolate.call_args_list[0]
        u_pred_arg = args[0]
        
        print(f"DEBUG: u_pred_arg type: {type(u_pred_arg)}")
        print(f"DEBUG: u_pred_arg: {u_pred_arg}")
        
        # We expect u_pred_arg to be approx 0.1 inside the domain
        # Since u starts at 0 and we add dt*1.0 = 0.1
        expected_val = 0.1
        self.assertTrue(torch.allclose(u_pred_arg, torch.tensor(expected_val)), 
                        f"Expected u_pred to be {expected_val}, got {u_pred_arg}")
        
        # VERIFICATION 2: Force Calculation
        # f_lag = (0 - u_lag) / dt = (0 - 10.0) / 0.1 = -100.0
        # This is calculated inside step_imex, not exposed.
        # But spread is called with this force.
        
        # Check calls to spread
        # spread(f_lag, 'u')
        # We expect 3 calls
        self.assertEqual(self.solver.ibm.spread.call_count, 3)
        
        args, _ = self.solver.ibm.spread.call_args_list[0]
        f_lag_arg = args[0]
        expected_force = -100.0
        self.assertTrue(torch.allclose(f_lag_arg, torch.tensor(expected_force)),
                        f"Expected forcing to be {expected_force}, got {f_lag_arg}")
        
        # VERIFICATION 3: Corrector Step (RHS Update)
        # With the fix, rhs_u_curr should NOT be modified in place.
        # It should remain "clean" (1.0) to be used as rhs_prev in the next step.
        # The force (5.0) is only added to a temporary rhs_total for the velocity update.
        
        # rhs_u_curr is stored in self.solver.rhs_u_prev after the swap at the end of step_imex
        
        rhs_final = self.solver.rhs_u_prev[:, 1:-1, 1:-1]
        expected_rhs = 1.0 # Clean RHS (without IBM force)
        self.assertTrue(torch.allclose(rhs_final, torch.tensor(expected_rhs)),
                        f"Expected final RHS (history) to be {expected_rhs} (clean), got {rhs_final.mean()}. Fix failed: IBM force leaked into history.")
        
        print("Verification passed!")

if __name__ == '__main__':
    unittest.main()
