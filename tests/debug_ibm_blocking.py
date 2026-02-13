import numpy as np
import torch
import yaml
from ibm import IBM_RKPM

def debug_ibm_blocking():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cpu')
    
    # Grid parameters from config
    nx = config['grid']['nx']
    ny = config['grid']['ny']
    nz = config['grid']['nz']
    Lx = config['domain']['Lx']
    Ly = config['domain']['Ly']
    Lz = config['domain']['Lz']
    
    # Create grid
    x_c = np.linspace(0, Lx, nx)
    x_f = np.linspace(0, Lx, nx+1) # Note: solver uses nx for x_f in staggered? No, usually nx+1 for faces?
    # Let's check solver.py convention. Usually x_f has nx points if periodic?
    # In solver.py: self.x_f = np.linspace(0, Lx, nx+1)[:-1] if periodic.
    # Let's assume periodic for now, so nx points.
    x_f = np.linspace(0, Lx, nx+1)[:-1]
    y_c = np.linspace(0, Ly, ny)
    y_f = np.linspace(0, Ly, ny+1)[:-1]
    
    # Z is non-uniform usually, but let's use uniform for simplicity or match config
    z_c = np.linspace(0, Lz, nz)
    z_f = np.linspace(0, Lz, nz+1)
    dz_c = np.ones_like(z_c) * (Lz/nz)
    dz_f = np.ones_like(z_f) * (Lz/nz)
    
    grid_data = {
        'x_c': x_c, 'x_f': x_f,
        'y_c': y_c, 'y_f': y_f,
        'z_c': z_c, 'z_f': z_f,
        'dz_c': dz_c, 'dz_f': dz_f,
        'dx': Lx/nx, 'dy': Ly/ny
    }
    
    print("Initializing IBM...")
    ibm = IBM_RKPM(config, grid_data, device=device)
    
    # Create a uniform velocity field U=1
    # Shapes: u(nx, ny, nz), v(nx, ny, nz), w(nx, ny, nz) roughly
    # In solver: u is (nx, ny, nz). 
    # Let's use torch tensors
    u = torch.ones((nx, ny, nz), device=device, dtype=torch.float64)
    v = torch.zeros((nx, ny, nz), device=device, dtype=torch.float64)
    w = torch.zeros((nx, ny, nz), device=device, dtype=torch.float64)
    
    dt = 0.01
    
    print(f"\nTest Configuration:")
    print(f"  Uniform Velocity U = 1.0")
    print(f"  Time step dt = {dt}")
    
    # 1. Interpolate
    print("\n1. Interpolating velocity to Lagrangian points...")
    u_lag = ibm.interpolate(u, 'u')
    print(f"  Mean U_lag: {u_lag.mean().item():.6f}")
    print(f"  Min U_lag:  {u_lag.min().item():.6f}")
    print(f"  Max U_lag:  {u_lag.max().item():.6f}")
    
    # 2. Compute Force
    # Target is 0.0
    print("\n2. Computing Lagrangian Force...")
    f_lag = (0.0 - u_lag) / dt
    print(f"  Mean F_lag: {f_lag.mean().item():.6f}")
    
    # 3. Spread Force
    print("\n3. Spreading Force to Grid...")
    f_euler = ibm.spread(f_lag, 'u')
    
    # 4. Apply Force
    print("\n4. Applying Force to Velocity...")
    # u_new = u + dt * f_euler
    # Note: f_euler shape might be (nx+1, ny, nz) or similar depending on spread implementation
    # ibm.spread returns shape (nx+1, ny, nz) for u?
    # Let's check shape
    print(f"  f_euler shape: {f_euler.shape}")
    print(f"  u shape:       {u.shape}")
    
    # Adjust slicing if needed. ibm.spread returns physical domain + ghosts?
    # In ibm.py: fx = torch.zeros(self.nx+1, self.ny, self.nz...)
    # If periodic, we might need to handle boundaries.
    # For this test, let's just add to the matching region.
    
    # Assuming periodic X, solver usually has u shape (nx, ny, nz).
    # ibm.spread returns (nx+1, ny, nz).
    # We'll just take the first nx points.
    u_new = u.clone()
    u_new += dt * f_euler[:nx, :, :]
    
    # 5. Check Result
    print("\n5. Checking Velocity at Lagrangian Points after Forcing...")
    u_lag_new = ibm.interpolate(u_new, 'u')
    
    print(f"  Mean U_lag_new: {u_lag_new.mean().item():.6f}")
    print(f"  Min U_lag_new:  {u_lag_new.min().item():.6f}")
    print(f"  Max U_lag_new:  {u_lag_new.max().item():.6f}")
    
    # Calculate Reduction Factor
    reduction = 1.0 - (u_lag_new.abs().mean() / u_lag.abs().mean())
    print(f"\n  Velocity Reduction Factor: {reduction.item()*100:.2f}%")
    
    if u_lag_new.abs().mean() < 1e-2:
        print("\nSUCCESS: Velocity effectively blocked at Lagrangian points.")
    else:
        print("\nFAILURE: Velocity NOT blocked. Force is too weak or spread incorrectly.")

if __name__ == "__main__":
    debug_ibm_blocking()
