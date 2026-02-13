import torch
from utils import load_flow_fields

def initialize_flow(nx, ny, nz, z_c, Ly, Lz, U_bulk=1.0, init_type='parabolic', perturbation_intensity=0.0, n_vortices=4, device='cpu', top_wall_bc_type='dirichlet'):
    """Initialize velocity and pressure fields. Creates tensors on specified device (CPU or CUDA)."""
    u = torch.zeros(nx+1, ny+2, nz+2, device=device)
    v = torch.zeros(nx+2, ny+1, nz+2, device=device)
    w = torch.zeros(nx+2, ny+2, nz+1, device=device)
    p = torch.zeros(nx+2, ny+2, nz+2, device=device)

    if init_type == 'parabolic' or init_type == 'vortices':
        # Base parabolic profile
        if top_wall_bc_type == 'neumann':
            # Free-slip top wall: Half-parabola with max velocity at top
            # u(z) = U_max * (z/Lz) * (2 - z/Lz)
            # U_max = 1.5 * U_bulk (same as full parabola)
            U_max = 1.5 * U_bulk
            z_norm = z_c / Lz  # z/Lz in [0, 1]
            u_profile = U_max * z_norm * (2 - z_norm)
        else:
            # No-slip top wall: Full parabola with max velocity at center
            U_max = 1.5 * U_bulk
            z_norm = 2 * z_c / Lz - 1
            u_profile = U_max * (1 - z_norm**2)
            
        u[:, :, :] = u_profile[None, None, :]
    
    if init_type == 'uniform':
        u[:, :, :] = U_bulk
    
    elif init_type == 'vortices':
        # Base profile already set above
        pass # Continue to vortex addition

        # Add counter-rotating vortices using streamfunction
        # psi(y, z) = A * sin(ky * y) * sin^2(pi * z / Lz)
        # v = -dpsi/dz, w = dpsi/dy
        # u perturbation = 0 (or small random noise)
        
        A = perturbation_intensity * U_bulk
        # ky determines the number of vortices in y
        # n_vortices = 2 means one pair (one positive, one negative lobe) -> ky = 2*pi/Ly
        # n_vortices = 4 means two pairs -> ky = 4*pi/Ly
        ky = n_vortices * torch.pi / Ly
        
        # Get device from input tensors
        device = z_c.device
        
        # Grid for v (staggered in y)
        dy = Ly / ny
        y_v = torch.linspace(0, Ly, ny+1, device=device)  # y-faces
        
        # Grid for w (staggered in z)
        y_c = 0.5 * (y_v[:-1] + y_v[1:])
        
        # Compute streamfunction and derivatives
        # For v: at y_v, z_c (use only interior z_c points)
        z_c_interior = z_c[1:nz+1]  # Shape (nz) - interior points only
        yy_v, zz_v = torch.meshgrid(y_v, z_c_interior, indexing='ij')
        psi_v = A * torch.sin(ky * yy_v) * torch.sin(torch.pi * zz_v / Lz)**2
        
        # v = -dpsi/dz
        v_pert = -A * torch.sin(ky * yy_v) * 2 * torch.sin(torch.pi * zz_v / Lz) * torch.cos(torch.pi * zz_v / Lz) * (torch.pi / Lz)
        
        # Broadcast v_pert (shape ny+1, nz) to v (shape nx+2, ny+1, nz+2)
        v[1:nx+1, :, 1:nz+1] += v_pert[None, :, :]
        
        # For w: at y_c, z_f (z-faces)
        # Approximate z_f for w perturbation using interior z_c
        z_f_local = torch.cat([torch.tensor([0.0], device=device), 
                               0.5 * (z_c_interior[:-1] + z_c_interior[1:]), 
                               torch.tensor([Lz], device=device)])
        
        yy_w, zz_w = torch.meshgrid(y_c, z_f_local, indexing='ij')
        psi_w = A * torch.sin(ky * yy_w) * torch.sin(torch.pi * zz_w / Lz)**2
        
        # w = dpsi/dy
        w_pert = A * torch.cos(ky * yy_w) * ky * torch.sin(torch.pi * zz_w / Lz)**2
        
        # Broadcast w_pert (shape ny, nz+1) to w (shape nx+2, ny+2, nz+1)
        w[1:nx+1, 1:ny+1, :] += w_pert[None, :, :]

    # Add random perturbations
    if perturbation_intensity > 0.0 and init_type != 'vortices':
        print(f"Adding random perturbation with intensity {perturbation_intensity} * U_bulk", flush=True)
        noise_scale = perturbation_intensity * U_bulk
        
        # Add noise to internal cells, avoiding boundaries
        # For Z (last dim), use 2:-2 to avoid the first and last fluid points next to the wall
        u[1:-1, 1:-1, 2:-2] += (torch.rand_like(u[1:-1, 1:-1, 2:-2]) - 0.5) * 2 * noise_scale
        v[1:-1, 1:-1, 2:-2] += (torch.rand_like(v[1:-1, 1:-1, 2:-2]) - 0.5) * 2 * noise_scale
        w[1:-1, 1:-1, 2:-2] += (torch.rand_like(w[1:-1, 1:-1, 2:-2]) - 0.5) * 2 * noise_scale

    # Apply BCs to ghost cells
    u[:, :, 0] = -u[:, :, 1]
    u[:, :, -1] = -u[:, :, -2]

    v[:, :, 0] = -v[:, :, 1]
    v[:, :, -1] = -v[:, :, -2]

    w[:, :, 0] = 0.0
    w[:, :, -1] = 0.0

    return u, v, w, p

def initialize_flow_from_file(field_file, device='cpu', reset_time=False):
    """
    Initialize velocity and pressure fields from a previously saved field file.

    This function loads a saved flow field (without projections or any post-processing)
    and returns the raw velocity and pressure fields ready for use as initial conditions.

    Args:
        field_file: Path to the .npz file containing saved flow fields
        device: Device to load tensors to ('cpu' or 'cuda')
        reset_time: If True, reset step and time to 0; if False, continue from saved values

    Returns:
        u, v, w, p: velocity and pressure fields as torch tensors on the specified device
        initial_step: Starting step number (0 if reset_time=True, otherwise from file)
        initial_time: Starting time (0.0 if reset_time=True, otherwise from file)

    Note:
        The loaded fields will have boundary conditions already applied from the saved state.
        However, it's recommended to re-apply boundary conditions after loading to ensure
        consistency with the current simulation setup.
    """
    print(f"Loading flow fields from: {field_file}", flush=True)

    # Load the saved fields
    fields = load_flow_fields(field_file, device=device)

    # Extract velocity and pressure fields
    u = fields['u']
    v = fields['v']
    w = fields['w']
    p = fields['p']

    # Extract time information
    if reset_time:
        initial_step = 0
        initial_time = 0.0
        print(f"Loaded fields from step {fields['step']}, time = {fields['time']:.6f}", flush=True)
        print(f"  Resetting step and time to 0", flush=True)
    else:
        initial_step = int(fields['step'])
        initial_time = float(fields['time'])
        print(f"Loaded fields from step {fields['step']}, time = {fields['time']:.6f}", flush=True)
        print(f"  Continuing from step {initial_step}, time = {initial_time:.6f}", flush=True)

    print(f"  u_tau = {fields['u_tau']:.6e}, forcing = {fields['forcing']:.6e}", flush=True)
    print(f"  Field shapes: u={u.shape}, v={v.shape}, w={w.shape}, p={p.shape}", flush=True)

    # Re-apply boundary conditions to ensure consistency
    # Dirichlet BC in z for u
    u[:, :, 0] = -u[:, :, 1]
    u[:, :, -1] = -u[:, :, -2]

    # Dirichlet BC in z for v
    v[:, :, 0] = -v[:, :, 1]
    v[:, :, -1] = -v[:, :, -2]

    # Dirichlet BC in z for w (w=0 at walls)
    w[:, :, 0] = 0.0
    w[:, :, -1] = 0.0

    return u, v, w, p, initial_step, initial_time
