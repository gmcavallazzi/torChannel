import torch

def initialize_fft_solver(nx, ny, nz, dx, dy, dz_c, dz_f, top_wall_bc_type='dirichlet',
                          bc_y='periodic', bc_x='periodic'):
    """
    Precompute wavenumbers and tridiagonal matrices for FFT-based Poisson solver.
    Returns a dictionary with precomputed data.

    Fully vectorized implementation - no Python loops.

    Args:
        nx, ny, nz: Grid dimensions
        dx, dy: Grid spacings in x, y
        dz_c, dz_f: Grid spacings in z (center and face)
        top_wall_bc_type: Boundary condition type at top wall
            - 'dirichlet': no-slip velocity BC (u=v=w=0) → Neumann pressure BC (∂p/∂z=0)
            - 'neumann': free-slip velocity BC (∂u/∂z=∂v/∂z=0, w=0) → Dirichlet pressure BC (p=0)
        bc_y: spanwise boundary condition
            - 'periodic' (default): FFT in y (current behaviour, unchanged)
            - 'wall': no-slip walls in y (a duct). The spanwise transform becomes a
              DCT-II (homogeneous Neumann pressure on the uniform y mesh), which
              decouples into the SAME tridiagonal-in-z solves. Implemented with a
              precomputed cosine matrix (built once, numerically inverted) so the
              forward/inverse pair round-trips to machine precision.

    Note on velocity-pressure BC coupling:
        The pressure BC is determined by the incompressibility constraint:
        - No-slip (u=v=w=0): Taking ∂/∂z of ∇·u=0 at wall gives ∂²w/∂z²=0, and since w=∂w/∂z=0,
          we get ∂p/∂z=0 (Neumann pressure BC)
        - Free-slip (∂u/∂z=∂v/∂z=0, w=0): Momentum equation at wall gives p=0 (Dirichlet pressure BC)
    """
    device = dz_c.device

    # Streamwise (x) eigenbasis: FFT (periodic) or DCT-II (non-periodic, e.g. inflow/
    # outflow with Neumann pressure). x is uniform, so a cosine transform decouples
    # into the same tridiagonal-in-z solves — identical machinery to the duct y-walls.
    Mx = None
    Mx_inv = None
    if bc_x == 'wall':
        k_idx = torch.arange(nx, device=device, dtype=dz_c.dtype)
        kx_mod = (2.0 / dx) * torch.sin(torch.pi * k_idx / (2.0 * nx))
        nkx = nx
        n_idx = (torch.arange(nx, device=device, dtype=dz_c.dtype) + 0.5).view(1, nx)
        M = torch.cos(torch.pi * n_idx * k_idx.view(nx, 1) / nx)   # (nx, nx)
        Mx = M.to(torch.complex128)
        Mx_inv = torch.linalg.inv(M).to(torch.complex128)
    elif bc_x == 'periodic':
        kx = torch.fft.fftfreq(nx, d=dx/(2*torch.pi), device=device)
        kx_mod = (2.0 / dx) * torch.sin(kx * dx / 2.0)
        nkx = len(kx)
    else:
        raise ValueError(f"unknown bc_x {bc_x!r} (expected 'periodic' or 'wall')")

    # Spanwise (y) eigenbasis: FFT (periodic) or DCT-II (no-slip walls / duct).
    My = None
    My_inv = None
    if bc_y == 'wall':
        # DCT-II modified wavenumbers for cell-centred Neumann pressure on a UNIFORM
        # y mesh: eigenvalue of the 3-point Laplacian for mode cos(pi*k*(n+1/2)/ny)
        # is -(2/dy*sin(pi*k/(2*ny)))^2 — same (2/dy)*sin(.) form as the periodic case.
        k_idx = torch.arange(ny, device=device, dtype=dz_c.dtype)
        ky_mod = (2.0 / dy) * torch.sin(torch.pi * k_idx / (2.0 * ny))
        nky = ny
        # Forward DCT-II matrix M[k,n] = cos(pi*(n+1/2)*k/ny); inverse by linalg.inv
        n_idx = (torch.arange(ny, device=device, dtype=dz_c.dtype) + 0.5).view(1, ny)
        kcol = k_idx.view(ny, 1)
        M = torch.cos(torch.pi * n_idx * kcol / ny)        # (ny, ny)
        My = M.to(torch.complex128)
        My_inv = torch.linalg.inv(M).to(torch.complex128)  # (ny, ny)
    elif bc_y == 'periodic':
        ky = torch.fft.rfftfreq(ny, d=dy/(2*torch.pi), device=device)
        ky_mod = (2.0 / dy) * torch.sin(ky * dy / 2.0)
        nky = len(ky)  # This is ny//2 + 1
    else:
        raise ValueError(f"unknown bc_y {bc_y!r} (expected 'periodic' or 'wall')")

    if bc_x == 'wall' and bc_y != 'wall':
        raise NotImplementedError(
            "bc_x='wall' (inflow/outflow) currently requires bc_y='wall' (a duct)")

    # Vectorized: Create meshgrid of wavenumbers
    # Shape: (nkx, nky)
    kx_grid, ky_grid = torch.meshgrid(kx_mod, ky_mod, indexing='ij')
    k_horiz_sq = kx_grid**2 + ky_grid**2  # Shape: (nkx, nky)

    # Vectorized tridiagonal matrix construction
    # Build coefficients for all z-levels at once
    # Shape: (nz,)
    dz_left = dz_c[:-1]   # dz_c[k] for k=0..nz-1
    dz_right = dz_c[1:]   # dz_c[k+1] for k=0..nz-1
    dz_avg = dz_f         # dz_f[k] for k=0..nz-1

    coeff_left = 1.0 / (dz_left * dz_avg)   # Shape: (nz,)
    coeff_right = 1.0 / (dz_right * dz_avg)  # Shape: (nz,)

    # Expand k_horiz_sq to include z dimension: (nkx, nky, nz)
    k_horiz_sq_3d = k_horiz_sq.unsqueeze(-1).expand(nkx, nky, nz)

    # Build tridiagonal diagonals
    # Lower diagonal: a[k] = coeff_left[k] for k > 0, else 0
    tri_a = torch.zeros(nkx, nky, nz, device=device)
    tri_a[:, :, 1:] = coeff_left[1:].view(1, 1, -1)

    # Upper diagonal: c[k] = coeff_right[k] for k < nz-1, else 0
    tri_c = torch.zeros(nkx, nky, nz, device=device)
    tri_c[:, :, :-1] = coeff_right[:-1].view(1, 1, -1)

    # Main diagonal: b[k] = -(coeff_left[k] + coeff_right[k] + k_horiz_sq)
    coeff_sum = coeff_left + coeff_right  # Shape: (nz,)
    tri_b = -(coeff_sum.view(1, 1, -1) + k_horiz_sq_3d)  # Shape: (nkx, nky, nz)

    # Apply boundary conditions at walls
    # Bottom wall: always Neumann pressure BC (no-slip velocity → ∂p/∂z = 0)
    # Costa's approach: modify stencil for ghost cell BC
    tri_b[:, :, 0] += coeff_left[0]

    # Top wall:
    # Always use Neumann pressure BC (∂p/∂z = 0) for rigid walls (w=0).
    # This applies to both 'dirichlet' (no-slip) and 'neumann' (free-slip) velocity BCs.
    #
    # Physical reasoning:
    # For a rigid wall (w=0), the wall-normal momentum equation reduces to ∂p/∂z ≈ 0 
    # (assuming viscous terms are small/zero at the wall).
    # The previous assumption of Dirichlet pressure (p=0) for free-slip is typically for 
    # free surfaces, not rigid lids. Using p=0 with w=0 leads to uncorrectable divergence 
    # at the boundary cell because the boundary velocity is fixed.
    tri_b[:, :, -1] += coeff_right[-1]

    # Pin the singular (kx=0, ky=0) pressure mode. With Neumann pressure in every
    # direction this mode has a constant nullspace, so its z-tridiagonal is singular
    # (the batched Thomas solve would divide by ~0). Fix the gauge by pinning p=0 at
    # z=0 for that mode (row 0 -> identity); the RHS entry is zeroed in the solve.
    # The velocity projection uses only grad(p), so fixing this constant changes nothing.
    tri_a[0, 0, 0] = 0.0
    tri_b[0, 0, 0] = 1.0
    tri_c[0, 0, 0] = 0.0

    # Pre-allocate workspace for pressure field (GPU optimization)
    # Reusing this workspace avoids repeated allocations every timestep
    workspace_p = torch.zeros(nx+2, ny+2, nz+2, device=device)

    return {
        'tri_a': tri_a,
        'tri_b': tri_b,
        'tri_c': tri_c,
        'nx': nx,
        'ny': ny,
        'nz': nz,
        'workspace_p': workspace_p,
        'top_wall_bc_type': top_wall_bc_type,
        'bc_y': bc_y,
        'My': My,           # DCT-II forward matrix (complex), None if periodic
        'My_inv': My_inv,   # DCT-II inverse matrix (complex), None if periodic
        'bc_x': bc_x,
        'Mx': Mx,           # DCT-II forward matrix in x (complex), None if periodic
        'Mx_inv': Mx_inv,
    }

def solve_poisson_fft(div, fft_data):
    """
    Solve Poisson equation using precomputed FFT data.
    Periodic BC in x always. In y: periodic (FFT) or no-slip walls (DCT-II), set by
    fft_data['bc_y']. Pressure BC in z is Neumann (rigid walls, w=0).

    Returns p with ghost cells.
    """
    nx = fft_data['nx']
    ny = fft_data['ny']
    nz = fft_data['nz']
    tri_a = fft_data['tri_a']
    tri_b = fft_data['tri_b']
    tri_c = fft_data['tri_c']
    bc_y = fft_data['bc_y']
    bc_x = fft_data.get('bc_x', 'periodic')

    # ---- forward transforms (x then y; FFT if periodic, DCT-II matrix if wall) ----
    if bc_x == 'wall':
        # inflow/outflow: Neumann pressure in x via DCT-II. Implemented for a duct
        # (bc_y == 'wall'), so x AND y are cosine transforms + tridiagonal in z.
        Mx = fft_data['Mx']; Mx_inv = fft_data['Mx_inv']
        My = fft_data['My']; My_inv = fft_data['My_inv']
        Xx = torch.einsum('kn,nij->kij', Mx, div.to(torch.complex128))  # DCT-II in x
        div_hat = torch.einsum('km,imj->ikj', My, Xx)                   # DCT-II in y
        nky = ny
    elif bc_y == 'wall':
        My = fft_data['My']; My_inv = fft_data['My_inv']
        Xk = torch.fft.fft(div, dim=0)                       # (nx, ny, nz) complex
        div_hat = torch.einsum('kn,inz->ikz', My, Xk)        # DCT-II along y
        nky = ny
    else:
        div_hat = torch.fft.rfft2(div, dim=(0, 1))           # (nx, ny//2+1, nz)
        nky = div_hat.shape[1]

    nkx = div_hat.shape[0]

    # Zero the RHS of the pinned (kx=0, ky=0, z=0) gauge row (see initialize).
    div_hat[0, 0, 0] = 0.0

    # ---- tridiagonal solve in z (one per (kx, ky) mode) ----
    p_hat_flat = solve_tridiagonal(tri_a.reshape(-1, nz), tri_b.reshape(-1, nz),
                                   tri_c.reshape(-1, nz), div_hat.reshape(-1, nz))
    p_hat = p_hat_flat.reshape(nkx, nky, nz)

    # ---- inverse transforms (y then x) ----
    if bc_x == 'wall':
        p_y = torch.einsum('mk,ikj->imj', My_inv, p_hat)               # inverse DCT-II in y
        p_interior = torch.einsum('nk,kij->nij', Mx_inv, p_y).real     # inverse DCT-II in x
    elif bc_y == 'wall':
        p_y = torch.einsum('nk,ikz->inz', My_inv, p_hat)
        p_interior = torch.fft.ifft(p_y, dim=0).real
    else:
        p_interior = torch.fft.irfft2(p_hat, s=(nx, ny), dim=(0, 1))

    p = fft_data['workspace_p']
    p.zero_()
    p[1:nx+1, 1:ny+1, 1:nz+1] = p_interior

    # x ghosts: Neumann (inflow/outflow) or periodic
    if bc_x == 'wall':
        p[0] = p[1]; p[nx+1] = p[nx]
    else:
        p[0] = p[nx]; p[nx+1] = p[1]

    # y ghosts
    if bc_y == 'wall':
        p[:, 0] = p[:, 1]; p[:, ny+1] = p[:, ny]
    else:
        p[:, 0] = p[:, ny]; p[:, ny+1] = p[:, 1]

    # z ghosts: Neumann at both rigid walls
    p[:, :, 0] = p[:, :, 1]
    p[:, :, nz+1] = p[:, :, nz]

    return p

@torch.jit.script
def solve_tridiagonal(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
                     d: torch.Tensor) -> torch.Tensor:
    """
    Solve batched tridiagonal systems using Thomas algorithm (vectorized).
    JIT-compiled for GPU performance.

    Args:
        a: Lower diagonal (batch, n)
        b: Main diagonal (batch, n)
        c: Upper diagonal (batch, n)
        d: RHS (batch, n)

    Returns:
        x: Solution (batch, n)
    """
    # Dimensions
    batch_size, n = d.shape

    # Get device/dtype
    device = d.device
    dtype = d.dtype

    # Allocate temporaries
    c_prime = torch.zeros((batch_size, n), dtype=dtype, device=device)
    d_prime = torch.zeros((batch_size, n), dtype=dtype, device=device)
    x = torch.zeros((batch_size, n), dtype=dtype, device=device)

    # Forward sweep
    # i = 0
    c_prime[:, 0] = c[:, 0] / b[:, 0]
    d_prime[:, 0] = d[:, 0] / b[:, 0]

    for i in range(1, n):
        denom = b[:, i] - a[:, i] * c_prime[:, i-1]
        # Avoid division by zero if any denom is 0 (unlikely for Poisson)
        # For now assume well-posed

        if i < n - 1:
            c_prime[:, i] = c[:, i] / denom

        d_prime[:, i] = (d[:, i] - a[:, i] * d_prime[:, i-1]) / denom

    # Back substitution
    x[:, n-1] = d_prime[:, n-1]
    for i in range(n-2, -1, -1):
        x[:, i] = d_prime[:, i] - c_prime[:, i] * x[:, i+1]

    return x
