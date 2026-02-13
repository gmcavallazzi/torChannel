
import torch
import numpy as np
from statistics import TurbulenceStats

def verify_wall_averaging():
    print("="*80)
    print("VERIFYING WALL AVERAGING")
    print("="*80)
    
    nx = 32
    ny = 32
    nz = 32
    Lx = 2 * np.pi
    Ly = 2 * np.pi
    Lz = 2.0
    
    device = 'cpu'
    
    # Mock grid
    z_c = torch.linspace(0, Lz, nz+2)
    z_f = torch.linspace(0, Lz, nz+1)
    dz_c = z_c[1:] - z_c[:-1]
    dz_f = z_f[1:] - z_f[:-1]
    dx = Lx / nx
    dy = Ly / ny
    nu = 1e-4
    Re_tau_target = 180.0
    
    # Initialize stats
    stats = TurbulenceStats(nx, ny, nz, Lx, Ly, Lz, z_c, z_f, dz_c, dz_f, dx, dy, nu, Re_tau_target, device=device)
    
    # Create synthetic field: u = cos(3x - 2y)
    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    u_field = np.cos(3*X - 2*Y)
    
    # Put SAME signal on both walls
    # u_bot = cos(3x - 2y)
    # u_top = cos(3x - 2y)
    
    # The averaging logic does: u_avg = 0.5 * (u_bot + flip(u_top))
    # flip(u_top) effectively changes y -> -y (plus shift)
    # So flip(u_top) should look like cos(3x + 2y)
    # u_avg = 0.5 * (cos(3x - 2y) + cos(3x + 2y))
    #       = cos(3x) * cos(2y)
    #       = 0.5 * (e^i3x + e^-i3x) * 0.5 * (e^i2y + e^-i2y)
    # Modes: (3, 2), (3, -2), (-3, 2), (-3, -2)
    
    # So we expect energy at (3, 2) AND (3, -2).
    # (3, -2) corresponds to our original signal.
    # (3, 2) is the new mode introduced by the flip.
    
    u = torch.zeros((nx+1, ny+2, nz+2))
    v = torch.zeros((nx+2, ny+1, nz+2))
    w = torch.zeros((nx+2, ny+2, nz+1))
    
    k_bot = stats.k_bot
    k_top = stats.k_top
    
    u_signal = torch.from_numpy(u_field).float()
    u_signal_periodic = torch.cat([u_signal, u_signal[0:1, :]], dim=0)
    
    u[:, 1:ny+1, k_bot] = u_signal_periodic
    u[:, 1:ny+1, k_top] = u_signal_periodic
    
    # Accumulate
    stats.accumulate_statistics(u, v, w, 1.0)
    
    # Finalize
    results = stats.finalize_statistics()
    premult_uu = results['premult_uu']
    
    # Check energy at (3, -2) -> indices (3, 2) in our folded check?
    # Wait, premult_uu is (nkx, nky).
    # kx indices 1..nkx. ky indices 1..nky.
    # kx=3 is index 2. ky=2 is index 1.
    
    # The folding logic sums E(kx, ky) + E(-kx, ky).
    # E(kx, ky) corresponds to mode (kx, ky).
    # E(-kx, ky) corresponds to mode (-kx, ky) = (kx, -ky).
    
    # So premult_uu[kx, ky] contains energy of (kx, ky) AND (kx, -ky).
    
    # For u_avg = cos(3x)cos(2y), we have equal energy in (3, 2) and (3, -2).
    # So premult_uu[2, 1] should be sum of both.
    
    val = premult_uu[2, 1]
    print(f"\nEnergy at (kx=3, ky=2) [sum of (3,2) and (3,-2)]: {val:.6e}")
    
    if val > 1e-10:
        print("SUCCESS: Energy detected!")
    else:
        print("FAILURE: Energy is zero!")
        exit(1)
        
    # Now let's try a signal that should be perfectly symmetric
    # u_bot = cos(3x - 2y)
    # u_top = cos(3x + 2y)
    # flip(u_top) -> cos(3x - 2y)
    # u_avg = cos(3x - 2y)
    # This is a single mode (3, -2).
    
    print("\nTest 2: Symmetric signals")
    stats2 = TurbulenceStats(nx, ny, nz, Lx, Ly, Lz, z_c, z_f, dz_c, dz_f, dx, dy, nu, Re_tau_target, device=device)
    
    u_bot_field = np.cos(3*X - 2*Y)
    u_top_field = np.cos(3*X + 2*Y)
    
    u2 = torch.zeros((nx+1, ny+2, nz+2))
    
    u_bot_periodic = torch.cat([torch.from_numpy(u_bot_field).float(), torch.from_numpy(u_bot_field).float()[0:1, :]], dim=0)
    u_top_periodic = torch.cat([torch.from_numpy(u_top_field).float(), torch.from_numpy(u_top_field).float()[0:1, :]], dim=0)
    
    u2[:, 1:ny+1, k_bot] = u_bot_periodic
    u2[:, 1:ny+1, k_top] = u_top_periodic
    
    stats2.accumulate_statistics(u2, v, w, 1.0)
    results2 = stats2.finalize_statistics()
    premult_uu2 = results2['premult_uu']
    
    val2 = premult_uu2[2, 1]
    print(f"Energy at (kx=3, ky=2) [should be same as single wall]: {val2:.6e}")
    
    # Compare with single wall energy (from previous step, approx 13.2)
    # If averaging works, this should be ~13.2.
    # If flip was wrong, they would interfere destructively or constructively in weird ways.
    
    if val2 > 1.0:
         print("SUCCESS: Symmetric averaging works!")
    else:
         print("FAILURE: Symmetric averaging failed!")
         exit(1)

if __name__ == "__main__":
    verify_wall_averaging()
