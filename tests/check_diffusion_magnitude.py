import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import generate_grid
from operators import diffusion_u

def check_diffusion():
    nx, ny, nz = 4, 4, 16
    Lx, Ly, Lz = 0.1, 0.1, 2.0
    Re = 1000.0
    nu = 1.0 / Re
    gamma = 1.5
    U_bulk = 1.0
    
    dx = Lx / nx
    dy = Ly / ny
    
    z_f, z_c, dz_f, dz_c = generate_grid(gamma, nz, Lz)
    
    # Parabolic profile
    # u = 1.5 * U_bulk * (1 - (2*z/Lz - 1)^2)
    # Center of channel is z=Lz/2 = 1.
    # z_norm = 2*z/Lz - 1.
    
    u = torch.zeros(nx+1, ny+2, nz+2)
    
    # Set profile
    for k in range(nz+2):
        z = z_c[k]
        z_norm = 2 * z / Lz - 1
        # Apply profile even to ghosts for now, to check interior diffusion
        val = 1.5 * U_bulk * (1 - z_norm**2)
        u[:, :, k] = val
        
    # Apply solver's BCs to ghosts (overwrite analytical ghosts)
    # u[0] = -u[1] (Dirichlet at walls)
    # Wall indices in z:
    # z_c indices 0..nz+1.
    # Fluid cells 1..nz.
    # Ghosts 0 and nz+1.
    # Solver sets u[:,:,0] = -u[:,:,1].
    u[:, :, 0] = -u[:, :, 1]
    u[:, :, -1] = -u[:, :, -2]
    
    # Compute diffusion
    diff = diffusion_u(u, nx, ny, nz, dx, dy, dz_c, dz_f, nu)
    
    # Check diffusion at center
    # Center index k approx nz/2 + 1
    k_center = nz // 2 + 1
    print(f"Diffusion at center (k={k_center}): {diff[0, 0, k_center-1]:.6e}")
    
    # Check weighted mean diffusion
    # cell_vol = dx * dy * dz_f
    # dz_f is 1D tensor of length nz.
    # diff is (nx+1, ny+2, nz+2).
    # We need to use interior of diff: diff[:, :, 1:nz+1]
    diff_interior = diff[:, :, 1:nz+1]
    
    cell_vol = dx * dy * dz_f.view(1, 1, -1)
    total_vol = torch.sum(cell_vol) * (nx+1) * (ny+2) # Total volume of domain?
    # Actually total_vol should be sum of cell_vols.
    # But cell_vol varies only in z.
    # We sum over all x, y, z.
    # sum(cell_vol) * (nx+1) * (ny+2) is correct if we sum over all u points.
    # But u points include ghosts in y?
    # u interior is 1:nx+1? No, u is staggered in x.
    # Interior u cells are 1:nx.
    # Interior y cells are 1:ny+1? No, 1:ny.
    # Let's just sum over the computed diffusion region.
    # diffusion_u computes diff_u[1:nx, 1:ny+1, 1:nz+1].
    # So we should sum over this region.
    
    diff_region = diff[1:nx, 1:ny+1, 1:nz+1]
    # diff_region shape (nx-1, ny, nz).
    # cell_vol shape (1, 1, nz).
    
    weighted_sum = torch.sum(diff_region * cell_vol)
    total_region_vol = torch.sum(cell_vol) * (nx-1) * ny
    
    weighted_mean = weighted_sum / total_region_vol
    print(f"Weighted mean diffusion: {weighted_mean:.6e}")
    
    # Unweighted mean
    mean_diff = torch.mean(diff)
    print(f"Unweighted mean diffusion: {mean_diff:.6e}")
    
    # Print diffusion profile along z
    print("\nDiffusion profile (z-line):")
    for k in range(nz):
        print(f"  z={z_c[k+1]:.4f}, diff={diff[0,0,k]:.6e}, dz_f={dz_f[k]:.4f}")
    
    # Theoretical diffusion
    # u'' = -3.0 * U_bulk / (Lz/2)**2 = -3.0
    # nu * u'' = -0.003
    print(f"Theoretical diffusion: {-3.0 * nu:.6e}")
    
    # Check wall flux approximation
    # u_1 = u[1]. z_1 = z_c[1].
    # flux = u_1 / z_1.
    # Analytical flux = 3.
    u_1 = u[0, 0, 1]
    z_1 = z_c[1] # Note: z_c[0] is ghost. z_c[1] is first fluid.
    # Wait, z_c in utils is 1D tensor.
    # z_c[0] is ghost.
    
    flux_approx = u_1 / z_1
    print(f"Wall flux approx (u_1/z_1): {flux_approx:.6e}")
    print(f"z_1: {z_1:.6e}")
    
    # Check if z_1 is small enough
    # If z_1 is large, error is large.
    
    # Check bulk velocity calculation
    from utils import compute_bulk_velocity
    
    # cell_vol_ratio shape (nx, ny, nz)
    cell_vol_ratio = dx * dy * dz_f.view(1, 1, -1).expand(nx, ny, nz)
    total_volume = Lx * Ly * Lz
    
    u_bulk_discrete = compute_bulk_velocity(u, cell_vol_ratio, total_volume)
    print(f"Discrete bulk velocity: {u_bulk_discrete:.6e}")
    print(f"Analytical bulk velocity: {U_bulk:.6e}")

if __name__ == "__main__":
    check_diffusion()
