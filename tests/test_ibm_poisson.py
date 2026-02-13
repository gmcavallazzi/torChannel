"""
Test IBM implementation using Poisson equation with known analytical solution

Solves:  ∇²φ = f   in domain with cube obstacle
    BC:  φ = 0      on cube surface (via IBM)
         periodic    in x, y directions
         φ = φ_analytical on z boundaries

Uses manufactured solution to verify second-order convergence.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from ibm import Cube, apply_ibm_correction


class PoissonIBMSolver:
    """
    3D Poisson solver with IBM for cube obstacle
    """

    def __init__(self, Lx, Ly, Lz, nx, ny, nz, cube_center, cube_size,
                 use_z_stretch=False, device='cpu'):
        """
        Parameters:
        -----------
        Lx, Ly, Lz : float
            Domain size
        nx, ny, nz : int
            Number of grid points
        cube_center : tuple
            (xc, yc, zc)
        cube_size : float
            Edge length of cube
        use_z_stretch : bool
            Use tanh stretching in z-direction
        device : str
            'cpu' or 'cuda'
        """
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        self.nx, self.ny, self.nz = nx, ny, nz
        self.device = device

        # Create grid
        self.dx = Lx / nx
        self.dy = Ly / ny

        # Z-grid (possibly stretched)
        if use_z_stretch:
            # Tanh stretching concentrated near bottom
            z_uniform = torch.linspace(0, 1, nz, device=device)
            beta = 2.0  # Stretching parameter
            z_stretched = Lz * (1 + torch.tanh(beta * (z_uniform - 0.5)) / torch.tanh(beta/2)) / 2
            self.z = z_stretched
            self.dz = torch.diff(self.z)
            self.dz_grid = torch.zeros(nz, device=device)
            self.dz_grid[1:-1] = (self.z[2:] - self.z[:-2]) / 2  # Central spacing
            self.dz_grid[0] = self.z[1] - self.z[0]
            self.dz_grid[-1] = self.z[-1] - self.z[-2]
        else:
            self.dz = Lz / nz
            self.z = torch.linspace(0, Lz, nz, device=device)
            self.dz_grid = torch.full((nz,), self.dz, device=device)

        # Create 3D grids
        x = torch.linspace(0, Lx, nx, device=device)
        y = torch.linspace(0, Ly, ny, device=device)

        self.X, self.Y, self.Z = torch.meshgrid(x, y, self.z, indexing='ij')

        # Create cube
        self.cube = Cube(cube_center, cube_size, device=device)

        # Get IBM mask and corrections
        print("Computing IBM mask...")
        dz_3d = self.dz_grid.view(1, 1, -1).expand(nx, ny, nz)
        self.mask_data = self.cube.get_ibm_mask(self.X, self.Y, self.Z,
                                                  self.dx, self.dy, dz_3d)
        self.corrections = apply_ibm_correction(self.mask_data,
                                                  self.dx, self.dy, dz_3d)

        n_corrected = self.corrections['needs_correction'].sum().item()
        print(f"  Points needing correction: {n_corrected}")

        # Solution array
        self.phi = torch.zeros((nx, ny, nz), device=device)

    def analytical_solution(self, avoid_cube=True):
        """
        Manufactured solution that avoids the cube region

        φ(x,y,z) = sin(2πx/Lx) * cos(2πy/Ly) * (z²/Lz²) * mask

        where mask = 1 outside cube, 0 inside (smooth transition)
        """
        kx = 2 * np.pi / self.Lx
        ky = 2 * np.pi / self.Ly

        phi_ana = (torch.sin(kx * self.X) *
                   torch.cos(ky * self.Y) *
                   (self.Z / self.Lz)**2)

        if avoid_cube:
            # Set to zero inside and very close to cube
            sdf = self.cube.signed_distance(self.X, self.Y, self.Z)
            # Smooth transition over ~2 grid cells
            transition_width = 2 * min(self.dx, self.dy, self.dz_grid.min().item())
            mask = torch.sigmoid(sdf / transition_width * 10)
            phi_ana = phi_ana * mask

        return phi_ana

    def compute_source_term(self):
        """
        Compute f = -∇²φ_analytical

        For verification: we'll solve ∇²φ_numerical = f,
        then compare φ_numerical to φ_analytical
        """
        phi_ana = self.analytical_solution()

        # Compute Laplacian of analytical solution
        lap = torch.zeros_like(phi_ana)

        # d²/dx²
        lap[1:-1, :, :] += (phi_ana[2:, :, :] - 2*phi_ana[1:-1, :, :] +
                            phi_ana[:-2, :, :]) / self.dx**2

        # d²/dy²
        lap[:, 1:-1, :] += (phi_ana[:, 2:, :] - 2*phi_ana[:, 1:-1, :] +
                            phi_ana[:, :-2, :]) / self.dy**2

        # d²/dz² (handle stretched grid)
        for k in range(1, self.nz-1):
            dz_k = self.dz_grid[k]
            lap[:, :, k] += (phi_ana[:, :, k+1] - 2*phi_ana[:, :, k] +
                             phi_ana[:, :, k-1]) / dz_k**2

        # f = -∇²φ
        f = -lap

        # Set to zero inside cube (no equation solved there)
        f = torch.where(self.mask_data['inside'], torch.zeros_like(f), f)

        return f

    def solve_poisson(self, f, max_iter=10000, tol=1e-6):
        """
        Solve ∇²φ = f using Jacobi iteration

        Parameters:
        -----------
        f : torch.Tensor
            Source term
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance

        Returns:
        --------
        phi : torch.Tensor
            Solution
        converged : bool
            Whether iteration converged
        """
        phi = self.phi.clone()
        phi_new = phi.clone()

        # Jacobi iteration
        for iteration in range(max_iter):
            phi_old = phi.clone()

            # Interior points
            for i in range(1, self.nx-1):
                for j in range(1, self.ny-1):
                    for k in range(1, self.nz-1):
                        # Skip if inside cube
                        if self.mask_data['inside'][i, j, k]:
                            phi_new[i, j, k] = 0.0
                            continue

                        # Standard Laplacian
                        dx2_inv = 1.0 / self.dx**2
                        dy2_inv = 1.0 / self.dy**2
                        dz2_inv = 1.0 / self.dz_grid[k]**2

                        phi_xx = (phi[i+1, j, k] + phi[i-1, j, k]) * dx2_inv
                        phi_yy = (phi[i, j+1, k] + phi[i, j-1, k]) * dy2_inv
                        phi_zz = (phi[i, j, k+1] + phi[i, j, k-1]) * dz2_inv

                        # Central coefficient (without IBM)
                        coeff_center = -2.0 * (dx2_inv + dy2_inv + dz2_inv)

                        # Add IBM correction
                        lambda_ibm = self.corrections['lambda_total'][i, j, k]
                        coeff_center_ibm = coeff_center - lambda_ibm

                        # Update
                        phi_new[i, j, k] = (f[i, j, k] - phi_xx - phi_yy - phi_zz) / coeff_center_ibm

            # Periodic BC in x, y
            phi_new[0, :, :] = phi_new[-2, :, :]
            phi_new[-1, :, :] = phi_new[1, :, :]
            phi_new[:, 0, :] = phi_new[:, -2, :]
            phi_new[:, -1, :] = phi_new[:, 1, :]

            # Dirichlet at z boundaries (use analytical solution)
            phi_ana = self.analytical_solution()
            phi_new[:, :, 0] = phi_ana[:, :, 0]
            phi_new[:, :, -1] = phi_ana[:, :, -1]

            # Check convergence
            residual = torch.max(torch.abs(phi_new - phi_old)).item()

            if iteration % 1000 == 0:
                print(f"  Iteration {iteration}: residual = {residual:.3e}")

            phi = phi_new.clone()

            if residual < tol:
                print(f"  Converged in {iteration} iterations (residual={residual:.3e})")
                self.phi = phi
                return phi, True

        print(f"  Did NOT converge after {max_iter} iterations (residual={residual:.3e})")
        self.phi = phi
        return phi, False

    def compute_error(self):
        """
        Compute error relative to analytical solution
        """
        phi_ana = self.analytical_solution()

        # Only compare in fluid region
        fluid_mask = ~self.mask_data['inside']

        error = torch.abs(self.phi - phi_ana)
        error_fluid = error[fluid_mask]

        l2_error = torch.sqrt(torch.mean(error_fluid**2)).item()
        linf_error = torch.max(error_fluid).item()

        return l2_error, linf_error


def test_convergence(use_z_stretch=False):
    """
    Run grid convergence study
    """
    print("="*80)
    print(f"Grid Convergence Test (z_stretch={use_z_stretch})")
    print("="*80)

    # Domain
    Lx, Ly, Lz = 4.0, 4.0, 2.0

    # Cube at center
    cube_center = (Lx/2, Ly/2, Lz/2)
    cube_size = 0.4

    # Test different resolutions
    resolutions = [16, 24, 32, 48, 64]
    l2_errors = []
    linf_errors = []
    grid_sizes = []

    for N in resolutions:
        print(f"\n--- Resolution: {N}³ ---")

        solver = PoissonIBMSolver(Lx, Ly, Lz, N, N, N,
                                   cube_center, cube_size,
                                   use_z_stretch=use_z_stretch,
                                   device='cpu')

        # Compute source term
        f = solver.compute_source_term()

        # Solve
        phi, converged = solver.solve_poisson(f, max_iter=5000, tol=1e-7)

        if not converged:
            print("  WARNING: Did not converge!")

        # Compute error
        l2, linf = solver.compute_error()
        print(f"  L2 error:   {l2:.6e}")
        print(f"  Linf error: {linf:.6e}")

        l2_errors.append(l2)
        linf_errors.append(linf)
        grid_sizes.append(Lx / N)

    # Compute convergence rates
    print("\n" + "="*80)
    print("Convergence Analysis")
    print("="*80)

    for i in range(1, len(resolutions)):
        rate_l2 = np.log(l2_errors[i-1] / l2_errors[i]) / np.log(grid_sizes[i-1] / grid_sizes[i])
        rate_linf = np.log(linf_errors[i-1] / linf_errors[i]) / np.log(grid_sizes[i-1] / grid_sizes[i])
        print(f"N={resolutions[i-1]}->{resolutions[i]}: L2 rate={rate_l2:.3f}, Linf rate={rate_linf:.3f}")

    # Plot
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.loglog(grid_sizes, l2_errors, 'o-', label='L2 error')
    plt.loglog(grid_sizes, linf_errors, 's-', label='L∞ error')
    plt.loglog(grid_sizes, np.array(grid_sizes)**2 * l2_errors[0] / grid_sizes[0]**2,
               '--', label='O(Δx²)')
    plt.xlabel('Grid spacing Δx')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.title(f'Convergence (z_stretch={use_z_stretch})')

    plt.subplot(1, 2, 2)
    rates_l2 = []
    for i in range(1, len(resolutions)):
        rate = np.log(l2_errors[i-1] / l2_errors[i]) / np.log(grid_sizes[i-1] / grid_sizes[i])
        rates_l2.append(rate)
    plt.plot(resolutions[1:], rates_l2, 'o-')
    plt.axhline(y=2.0, color='r', linestyle='--', label='Second-order')
    plt.xlabel('Resolution')
    plt.ylabel('Convergence rate')
    plt.legend()
    plt.grid(True)
    plt.title('Observed convergence rate')

    plt.tight_layout()
    plt.savefig(f'test_ibm_poisson_convergence_stretch_{use_z_stretch}.png', dpi=150)
    print(f"\nPlot saved: test_ibm_poisson_convergence_stretch_{use_z_stretch}.png")

    return l2_errors, linf_errors, grid_sizes


if __name__ == '__main__':
    print("\n" + "="*80)
    print("IBM POISSON EQUATION TEST")
    print("="*80)

    # Test 1: Uniform grid
    print("\nTest 1: Uniform grid")
    test_convergence(use_z_stretch=False)

    # Test 2: Stretched grid
    print("\nTest 2: Stretched z-grid")
    test_convergence(use_z_stretch=True)

    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)
