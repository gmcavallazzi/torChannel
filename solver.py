import os
import torch
import yaml
import numpy as np
import operators
from utils import generate_grid, plot_grid, save_grid_csv, plot_profile, compute_u_tau, compute_bulk_velocity, test_poisson_matrix_indexing, print_divergence_field, compute_divergence, print_poisson_matrix, save_flow_fields
from initflow import initialize_flow, initialize_flow_from_file
from operators import diffusion_u, diffusion_v, diffusion_w, advection_u, advection_v, advection_w, diffusion_xy_u, diffusion_xy_v, diffusion_xy_w, solve_implicit_diffusion_u, solve_implicit_diffusion_v, solve_implicit_diffusion_w
from projection import build_poisson_matrix, solve_poisson, project_velocity
from projection_fft import initialize_fft_solver, solve_poisson_fft
from statistics import TurbulenceStats


@torch.jit.script
def apply_bc_all(u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, top_wall_bc_type: str = 'dirichlet') -> None:
    """
    Apply boundary conditions to all velocity components in a single fused kernel.
    This reduces 3 separate kernel launches to 1, improving GPU performance.

    Boundary conditions:
    - Periodic in x and y (with staggered grid adjustments)
    - Bottom wall (z=0): Always no-slip (Dirichlet)
      * u = v = w = 0
    - Top wall (z=Lz): Depends on top_wall_bc_type
      * 'dirichlet' (no-slip): u = v = w = 0
      * 'neumann' (free-slip): ∂u/∂z = ∂v/∂z = 0, w = 0

    Args:
        u: Streamwise velocity (staggered in x)
        v: Spanwise velocity (staggered in y)
        w: Wall-normal velocity (staggered in z)
        top_wall_bc_type: 'dirichlet' (no-slip) or 'neumann' (free-slip)
    """
    # U-velocity boundary conditions
    # Periodic BC in x (u is staggered in x, shape nx+1)
    u[0, :, :] = u[-1, :, :]
    # Periodic BC in y (u is NOT staggered in y)
    u[:, 0, :] = u[:, -2, :]
    u[:, -1, :] = u[:, 1, :]
    # BC in z: bottom wall always Dirichlet, top wall depends on type
    u[:, :, 0] = -u[:, :, 1]  # Bottom: Dirichlet (no-slip, u=0)
    if top_wall_bc_type == 'neumann':
        u[:, :, -1] = u[:, :, -2]  # Top: Neumann (free-slip, ∂u/∂z=0)
    else:
        u[:, :, -1] = -u[:, :, -2]  # Top: Dirichlet (no-slip, u=0)

    # V-velocity boundary conditions
    # Periodic BC in x (v is NOT staggered in x)
    v[0, :, :] = v[-2, :, :]
    v[-1, :, :] = v[1, :, :]
    # Periodic BC in y (v is staggered in y, shape ny+1)
    v[:, 0, :] = v[:, -1, :]
    # BC in z: bottom wall always Dirichlet, top wall depends on type
    v[:, :, 0] = -v[:, :, 1]  # Bottom: Dirichlet (no-slip, v=0)
    if top_wall_bc_type == 'neumann':
        v[:, :, -1] = v[:, :, -2]  # Top: Neumann (free-slip, ∂v/∂z=0)
    else:
        v[:, :, -1] = -v[:, :, -2]  # Top: Dirichlet (no-slip, v=0)

    # W-velocity boundary conditions
    # Periodic BC in x (w is NOT staggered in x)
    w[0, :, :] = w[-2, :, :]
    w[-1, :, :] = w[1, :, :]
    # Periodic BC in y (w is NOT staggered in y)
    w[:, 0, :] = w[:, -2, :]
    w[:, -1, :] = w[:, 1, :]
    # BC in z: w=0 at both walls (no penetration, always Dirichlet)
    # This is true for BOTH 'dirichlet' and 'neumann' top_wall_bc_type
    w[:, :, 0] = 0.0   # Bottom wall: w = 0
    w[:, :, -1] = 0.0  # Top wall: w = 0 (even for 'neumann' type)

class ChannelFlow:

    def __init__(self, config_file='config.yaml'):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Device setup (GPU support)
        device_config = config.get('compute', {}).get('device', 'auto')
        if device_config == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device_config == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        print(f"\n{'='*80}", flush=True)
        print(f"Device: {self.device}", flush=True)
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)
            
            # Enable GPU performance optimizations
            torch.backends.cudnn.benchmark = True  # Auto-tune cuDNN kernels for this GPU
            torch.backends.cuda.matmul.allow_tf32 = True  # Enable TensorFloat-32 for matmul
            torch.backends.cudnn.allow_tf32 = True  # Enable TensorFloat-32 for convolutions
            print("GPU optimizations enabled: cuDNN benchmark, TF32", flush=True)
        print(f"{'='*80}\n", flush=True)

        self.nx = config['grid']['nx']
        self.ny = config['grid']['ny']
        self.nz = config['grid']['nz']

        self.Lx = config['domain']['Lx']
        self.Ly = config['domain']['Ly']
        self.Lz = config['domain']['Lz']
        self.stretching_type = config['domain'].get('stretching_type', 'symmetric')

        # Validate stretching type
        if self.stretching_type not in ['symmetric', 'bottom', 'hybrid']:
            raise ValueError(f"Invalid stretching type: {self.stretching_type}. Must be 'symmetric', 'bottom', or 'hybrid'")

        self.nu = 1.0 / config['flow']['Re']
        self.Re_tau = config['flow']['Re_tau']
        self.U_bulk = config['flow']['U_bulk']
        self.gamma = config['flow']['gamma']

        # Read boundary condition configuration
        bc_config = config.get('boundary_conditions', {})
        top_wall_config = bc_config.get('top_wall', {})
        self.top_wall_bc_type = top_wall_config.get('type', 'dirichlet')

        # Validate BC type
        if self.top_wall_bc_type not in ['dirichlet', 'neumann']:
            raise ValueError(f"Invalid top wall BC type: {self.top_wall_bc_type}. Must be 'dirichlet' or 'neumann'")

        print(f"Top wall BC type: {self.top_wall_bc_type}", flush=True)

        self.dt = config['time']['dt']
        self.n_steps = config['time']['n_steps']
        self.t_max = config['time'].get('t_max', 1000.0)
        self.cfl_target = config['time']['CFL_target']
        self.dt_update_interval = config['time'].get('dt_update_interval', 0)
        self.dt_max = config['time'].get('dt_max', 0.01)
        self.dt_min = config['time'].get('dt_min', 0.0001)
        self.time_scheme = config['time'].get('scheme', 'IMEX')  # Time stepping scheme: "IMEX", "FE", or "RK3"

        # Output settings
        output_config = config.get('output', {})
        self.results_folder = output_config.get('results_folder', 'results')
        
        # Use new n_out/n_save if available, fallback to old parameters
        self.n_out = output_config.get('n_out', output_config.get('print_interval', 10))
        self.n_save = output_config.get('n_save', output_config.get('slice_interval', 100))
        
        os.makedirs(self.results_folder, exist_ok=True)

        # Check if we're restarting from a file
        field_file = config['initialization'].get('field_file', None)
        is_restart = field_file is not None

        # Check if we're restarting statistics from a file
        stats_config = config.get('statistics', {})
        stats_restart_file = stats_config.get('restart_state_file', None)

        # Only clean results folder on fresh start (not on restart) AND if explicitly enabled
        clean_results = output_config.get('clean_results_on_fresh_start', False)
        
        if not is_restart and clean_results:
            print(f"Fresh start: Cleaning results folder: {self.results_folder}", flush=True)
            for filename in os.listdir(self.results_folder):
                file_path = os.path.join(self.results_folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        import shutil
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}", flush=True)
        elif not is_restart and not clean_results:
             print(f"Fresh start: Preserving existing files in {self.results_folder} (clean_results_on_fresh_start=False)", flush=True)
        else:
            print(f"Restart detected: Preserving existing files in {self.results_folder}", flush=True)
            print(f"  Timeseries data will be appended if it exists", flush=True)

        # Set double precision explicitly
        torch.set_default_dtype(torch.float64)

        # Generate grid (now created on correct device)
        if self.stretching_type == 'hybrid':
            # Import hybrid grid generator
            from utils import generate_hybrid_grid

            # Read hybrid grid parameters from config
            nz_uniform = config['grid'].get('nz_uniform', 75)
            nz_stretched = config['grid'].get('nz_stretched', 75)
            z_transition = config['domain'].get('z_transition', 0.25)
            gamma_stretched = config['domain'].get('gamma_stretched', 1.8)

            print(f"Generating hybrid grid:", flush=True)
            print(f"  Uniform region: nz={nz_uniform}, z ∈ [0, {z_transition}]", flush=True)
            print(f"  Stretched region: nz={nz_stretched}, z ∈ [{z_transition}, {self.Lz}], gamma={gamma_stretched}", flush=True)

            self.z_f, self.z_c, self.dz_f, self.dz_c = generate_hybrid_grid(
                nz_uniform, nz_stretched, z_transition, self.Lz,
                gamma_stretched, device=self.device
            )

            # Update nz to match actual grid size
            self.nz = len(self.dz_f)
            print(f"  Total grid cells: {self.nz}", flush=True)
        else:
            self.z_f, self.z_c, self.dz_f, self.dz_c = generate_grid(
                self.gamma, self.nz, self.Lz,
                device=self.device,
                stretching_type=self.stretching_type
            )

        print(f"Grid stretching type: {self.stretching_type}", flush=True)

        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        
        # Calculate cell volumes for bulk velocity
        # Vectorized cell volume computation using broadcasting
        # dz_c has (nz+1) elements matching original grid structure
        # self.cell_vol = (self.dx * self.dy * self.dz_f.view(1, 1, -1)).expand(self.nx, self.ny, self.nz)
        # Use dz_c for volume integration as requested
        self.cell_vol = (self.dx * self.dy * self.dz_f.view(1, 1, -1)).expand(self.nx, self.ny, self.nz)
            
        self.cell_vol_ratio = self.cell_vol # Renamed for clarity, but keeping variable name to match usage
        self.total_volume = self.Lx * self.Ly * self.Lz

        save_grid_csv(self.z_f, self.z_c, self.dz_f, self.dz_c, self.nz, self.results_folder)
        plot_grid(self.z_f, self.z_c, self.results_folder)

        # Initialize flow
        print("Initializing flow...", flush=True)

        # Check if we should load from a saved field file
        field_file = config['initialization'].get('field_file', None)
        reset_time = config['initialization'].get('reset_time', False)

        if is_restart:
            self.u, self.v, self.w, self.p, self.initial_step, self.time = initialize_flow_from_file(field_file, device=self.device, reset_time=reset_time)
            self.initial_time = self.time
            self.forcing = 0.0
        else:
            # Bulk forcing state (pressure gradient)
            self.forcing = 0.0
            self.u, self.v, self.w, self.p = initialize_flow(self.nx, self.ny, self.nz, self.z_c, self.Ly, self.Lz, 
                                                             U_bulk=self.U_bulk, 
                                                             init_type=config['initialization']['type'],
                                                             perturbation_intensity=config['initialization'].get('perturbation_intensity', 0.0),
                                                             n_vortices=config['initialization'].get('n_vortices', 4),
                                                             device=self.device,
                                                             top_wall_bc_type=self.top_wall_bc_type)
            self.initial_step = 0
            self.time = 0.0
            self.initial_time = 0.0
            
        print(f"DEBUG: u.dtype = {self.u.dtype}", flush=True)
        if torch.isnan(self.u).any() or torch.isnan(self.v).any() or torch.isnan(self.w).any():
             print("DEBUG: NaNs detected immediately after initialize_flow!", flush=True)

        # Rescale u to match U_bulk exactly (discrete integration)
        # This prevents a large initial forcing kick that distorts the profile
        # Only rescale if NOT restarting from a file
        if field_file is None:
            u_bulk_init = compute_bulk_velocity(self.u, self.cell_vol_ratio, self.total_volume)
            if abs(u_bulk_init) > 1e-9:
                self.u *= (self.U_bulk / u_bulk_init)
            else:
                print(f"WARNING: Initial bulk velocity is zero (init_type='{config['initialization']['type']}'). Skipping rescaling.", flush=True)
        else:
            print("Restarting from file: Skipping velocity rescaling to preserve divergence-free condition.", flush=True)


        self.solver_type = config.get('solver', {}).get('type', 'fft')

        if self.solver_type == 'direct':
            self.poisson_matrix = build_poisson_matrix(self.nx, self.ny, self.nz,
                                                         self.dx, self.dy, self.dz_c, self.dz_f,
                                                         top_wall_bc_type=self.top_wall_bc_type)
            # print_poisson_matrix(self.poisson_matrix, self.nx, self.ny, self.nz, self.results_folder)
        elif self.solver_type == 'fft':
            # Initialize FFT-based Poisson solver
            self.fft_data = initialize_fft_solver(self.nx, self.ny, self.nz,
                                                    self.dx, self.dy, self.dz_c, self.dz_f,
                                                    top_wall_bc_type=self.top_wall_bc_type)
        else:
            raise ValueError(f"Unknown solver type: {self.solver_type}")

        # Initialize statistics collector if enabled (n_stats > 0)
        stats_config = config.get('statistics', {})
        self.n_stats = stats_config.get('n_stats', 0)
        self.t_stats = stats_config.get('t_stats', 10.0)

        if self.n_stats > 0:
            z_plus_target = stats_config.get('z_plus_target', 15.0)
            stats_output_file = stats_config.get('output_file', 'turbulence_stats.npz')
            self.stats_output_path = os.path.join(self.results_folder, stats_output_file)

            # Statistics state file for checkpointing
            stats_state_file = stats_config.get('state_file', 'turbulence_stats_state.npz')
            self.stats_state_path = os.path.join(self.results_folder, stats_state_file)

            # Check if we should load from a saved statistics state
            stats_restart_file = stats_config.get('restart_state_file', None)

            print(f"\nStatistics collection enabled:", flush=True)
            print(f"  Collection starts at t = {self.t_stats:.2f}", flush=True)
            print(f"  Collection interval: every {self.n_stats} steps", flush=True)
            print(f"  Output file: {self.stats_output_path}", flush=True)
            print(f"  State checkpoint file: {self.stats_state_path}", flush=True)

            self.turbulence_stats = TurbulenceStats(
                self.nx, self.ny, self.nz,
                self.Lx, self.Ly, self.Lz,
                self.z_c, self.z_f, self.dz_c, self.dz_f,
                self.dx, self.dy, self.nu,
                self.Re_tau, z_plus_target=z_plus_target,
                device=self.device
            )

            # Load statistics state if restarting
            if stats_restart_file is not None:
                print(f"  Loading statistics state from: {stats_restart_file}", flush=True)
                self.turbulence_stats.load_state(stats_restart_file)
        else:
            self.turbulence_stats = None
            print(f"\nStatistics collection disabled (n_stats = 0)", flush=True)

        # Save initial fields
        u_tau_init = compute_u_tau(self.u, self.z_c, self.nu, top_wall_bc_type=self.top_wall_bc_type)
        save_flow_fields(self.u, self.v, self.w, self.p, self.z_c, self.z_f,
                        self.Lx, self.Ly, 0, 0.0, u_tau_init, 0.0,
                        self.results_folder, 'fields_init.npz')

        # Project initial field to be divergence-free
        # Project initial field to be divergence-free
        # Step 1: Compute divergence
        div = compute_divergence(self.u, self.v, self.w, self.nx, self.ny, self.nz,
                                  self.dx, self.dy, self.dz_f)
        print(f"DEBUG: Initial max(div) before projection: {torch.max(torch.abs(div)):.6e}", flush=True)
        
        # Step 2: Solve Poisson (using a dummy dt=1.0 for projection, or just solve ∇²p = div)
        # We want u_new = u - ∇p such that ∇.u_new = 0
        # ∇.u = ∇²p
        # So we solve ∇²p = div
        
        if self.solver_type == 'direct':
            self.p = solve_poisson(self.poisson_matrix, div, self.nx, self.ny, self.nz, self.top_wall_bc_type)
        elif self.solver_type == 'fft':
            self.p = solve_poisson_fft(div, self.fft_data)
            
        print(f"DEBUG: Max pressure after Poisson solve: {torch.max(torch.abs(self.p)):.6e}", flush=True)
        # Step 3: Correct velocity
        # Note: project_velocity uses dt to scale pressure gradient: u = u - dt*dp/dx
        # Here we solved ∇²p = div, so we want u = u - ∇p.
        # So we pass dt=1.0 to project_velocity
        self.u, self.v, self.w = project_velocity(self.u, self.v, self.w, self.p,
                                                    self.nx, self.ny, self.nz,
                                                    self.dx, self.dy, self.dz_c, self.dz_f, 1.0)
                                                    
        # Reapply BCs
        self.apply_bc_uvw()  # Fused boundary conditions

        # Check divergence after projection
        div_final = compute_divergence(self.u, self.v, self.w, self.nx, self.ny, self.nz,
                                        self.dx, self.dy, self.dz_f)
        max_div_final = torch.max(torch.abs(div_final))
        print(f"Initial divergence after projection: max(|div|) = {max_div_final:.6e}", flush=True)

        # Initialize storage for Adams-Bashforth 2 (dual buffer for swap pattern)
        self.rhs_u_prev = None
        self.rhs_v_prev = None
        self.rhs_w_prev = None
        self.rhs_u_curr = None
        self.rhs_v_curr = None
        self.rhs_w_curr = None

        # Initialize time tracking (use initial_time from loaded file if applicable)
        self.time = self.initial_time
        self.current_step = self.initial_step  # Track current step for diagnostics

    def apply_bc_u(self):
        """Apply boundary conditions to u-velocity (legacy, use apply_bc_uvw for better performance)"""
        # Periodic BC in x (u is staggered in x, shape nx+1)
        # u[0] is left face, u[nx] is right face. Periodic: u[0] = u[nx].
        # u indices: 0..nx. u[-1] is u[nx].
        self.u[0, :, :] = self.u[-1, :, :]
        # Periodic BC in y (u is NOT staggered in y)
        self.u[:, 0, :] = self.u[:, -2, :]
        self.u[:, -1, :] = self.u[:, 1, :]

        # BC in z: bottom wall always Dirichlet, top wall depends on type
        self.u[:, :, 0] = -self.u[:, :, 1]  # Bottom: Dirichlet (no-slip)
        if self.top_wall_bc_type == 'neumann':
            self.u[:, :, -1] = self.u[:, :, -2]  # Top: Neumann (free-slip, du/dz=0)
        else:
            self.u[:, :, -1] = -self.u[:, :, -2]  # Top: Dirichlet (no-slip, u=0)

    def apply_bc_v(self):
        """Apply boundary conditions to v-velocity (legacy, use apply_bc_uvw for better performance)"""
        # Periodic BC in x (v is NOT staggered in x)
        self.v[0, :, :] = self.v[-2, :, :]
        self.v[-1, :, :] = self.v[1, :, :]
        # Periodic BC in y (v is staggered in y, shape ny+1)
        # v[0] is bottom face, v[ny] is top face. Periodic: v[0] = v[ny].
        # v indices: 0..ny. v[-1] is v[ny].
        self.v[:, 0, :] = self.v[:, -1, :]
        # BC in z: bottom wall always Dirichlet, top wall depends on type
        self.v[:, :, 0] = -self.v[:, :, 1]  # Bottom: Dirichlet (no-slip)
        if self.top_wall_bc_type == 'neumann':
            self.v[:, :, -1] = self.v[:, :, -2]  # Top: Neumann (free-slip, dv/dz=0)
        else:
            self.v[:, :, -1] = -self.v[:, :, -2]  # Top: Dirichlet (no-slip, v=0)

    def apply_bc_w(self):
        """Apply boundary conditions to w-velocity (legacy, use apply_bc_uvw for better performance)"""
        # Periodic BC in x (w is NOT staggered in x)
        self.w[0, :, :] = self.w[-2, :, :]
        self.w[-1, :, :] = self.w[1, :, :]
        # Periodic BC in y (w is NOT staggered in y)
        self.w[:, 0, :] = self.w[:, -2, :]
        self.w[:, -1, :] = self.w[:, 1, :]
        # Dirichlet BC in z (w is staggered in z, w=0 at walls)
        self.w[:, :, 0] = 0.0
        self.w[:, :, -1] = 0.0

    def apply_bc_uvw(self):
        """Apply boundary conditions to all velocity components (optimized fused kernel)"""
        apply_bc_all(self.u, self.v, self.w, self.top_wall_bc_type)

    def compute_cfl_dt(self):
        """
        Compute timestep based on CFL condition using staggered grid velocities.
        Uses fused CUDA kernel for optimal GPU performance.

        For each cell, computes CFL contribution in each direction by:
        - Interpolating velocities to appropriate locations
        - Computing sum: u/dx + v/dy + w/dz for that direction
        - Taking maximum across all cells and directions
        """
        # Use fused CFL kernel (GPU-optimized, single kernel launch)
        dti = operators.compute_cfl_fused(
            self.u, self.v, self.w,
            self.nx, self.ny, self.nz,
            self.dx, self.dy,
            self.dz_f, self.dz_c
        )

        # Avoid division by zero
        if dti < 1e-10:
            dti = 1.0

        # Compute dtmax: 1.0 / dti (for AB2 time integration)
        dtmax = 1.0 / dti

        # Apply CFL target scaling
        dt_new = self.cfl_target * dtmax

        # Clamp to min/max
        dt_new = min(max(dt_new, self.dt_min), self.dt_max)

        return dt_new

    def compute_momentum_rhs_explicit(self):
        """
        Compute RHS for momentum equations: advection + diffusion

        Uses GPU-optimized fused kernel when available, falls back to separate
        kernels on CPU or for debugging.
        """
        # Try to use enhanced fused kernel v2 on GPU (Phase 4 optimization)
        if self.device.type == 'cuda' and hasattr(operators, 'compute_momentum_rhs_fused_v2'):
            # Enhanced fused kernel v2: 20-35% faster with better memory access patterns
            rhs_u, rhs_v, rhs_w = operators.compute_momentum_rhs_fused_v2(
                self.u, self.v, self.w,
                self.nx, self.ny, self.nz,
                self.dx, self.dy, self.dz_c, self.dz_f,
                self.nu
            )
        else:
            # Original separate kernels (CPU fallback or if fused not available)
            adv_u = advection_u(self.u, self.v, self.w, self.nx, self.ny, self.nz, 
                               self.dx, self.dy, self.dz_f)
            adv_v = advection_v(self.u, self.v, self.w, self.nx, self.ny, self.nz, 
                               self.dx, self.dy, self.dz_f)
            adv_w = advection_w(self.u, self.v, self.w, self.nx, self.ny, self.nz, 
                               self.dx, self.dy, self.dz_c)
            
            diff_u = diffusion_u(self.u, self.nx, self.ny, self.nz, 
                                self.dx, self.dy, self.dz_c, self.dz_f, self.nu)
            diff_v = diffusion_v(self.v, self.nx, self.ny, self.nz, 
                                self.dx, self.dy, self.dz_c, self.dz_f, self.nu)
            diff_w = diffusion_w(self.w, self.nx, self.ny, self.nz, 
                                self.dx, self.dy, self.dz_c, self.dz_f, self.nu) # Corrected dz_f, dz_c order
            
            rhs_u = diff_u - adv_u
            rhs_v = diff_v - adv_v
            rhs_w = diff_w - adv_w
        
        return rhs_u, rhs_v, rhs_w


    def step_forward_euler(self, dt):
        self.apply_bc_uvw()  # Fused boundary conditions

        # Compute RHS (diffusion - advection)
        rhs_u, rhs_v, rhs_w = self.compute_momentum_rhs_explicit()

        # Forward Euler time stepping: u = u + dt * RHS
        self.u += dt * rhs_u
        self.v += dt * rhs_v
        self.w += dt * rhs_w

        # Apply forcing to maintain constant flowrate
        # u_bulk_current = compute_bulk_velocity(self.u, self.cell_vol_ratio, self.total_volume)
        # forcing = (self.U_bulk - u_bulk_current) / dt
        
        forcing = 1.0
        
        self.u[1:self.nx+1, 1:self.ny+1, 1:self.nz+1] += dt * forcing

        # Update ghost cells for intermediate velocity before divergence computation
        self.apply_bc_uvw()  # Fused boundary conditions

        # === PROJECTION STEP ===
        # Step 1: Compute divergence of intermediate velocity
        div = compute_divergence(self.u, self.v, self.w, self.nx, self.ny, self.nz,
                                  self.dx, self.dy, self.dz_f)
        
        # Step 2: Solve Poisson equation for pressure: ∇²p = div/dt
        if self.solver_type == 'direct':
            self.p = solve_poisson(self.poisson_matrix, div / dt, self.nx, self.ny, self.nz, self.top_wall_bc_type)
        elif self.solver_type == 'fft':
            self.p = solve_poisson_fft(div / dt, self.fft_data)

        # Step 3: Project velocity to divergence-free field: u = u* - dt*∇p
        self.u, self.v, self.w = project_velocity(self.u, self.v, self.w, self.p,
                                                    self.nx, self.ny, self.nz,
                                                    self.dx, self.dy, self.dz_c, self.dz_f, dt)

        # Reapply boundary conditions after projection
        self.apply_bc_uvw()  # Fused boundary conditions

        return u_bulk_current, forcing

    def step_imex(self, dt):
        """
        IMEX (Implicit-Explicit) time stepping scheme.
        - Explicit treatment: advection + diffusion in x,y (using AB2)
        - Implicit treatment: diffusion in z (using backward Euler/CN)

        This allows larger timesteps by treating the stiff z-diffusion implicitly.
        """
        self.apply_bc_uvw()  # Fused boundary conditions (3x faster than separate calls)

        # ========== EXPLICIT PART: Advection + xy-diffusion ==========

        # Try to use fused IMEX kernel on GPU (Phase 4 optimization)
        if self.device.type == 'cuda' and hasattr(operators, 'compute_momentum_rhs_fused_imex'):
            # Fused IMEX kernel: 30-50% faster by combining 6 operations into 1
            rhs_u_explicit, rhs_v_explicit, rhs_w_explicit = operators.compute_momentum_rhs_fused_imex(
                self.u, self.v, self.w,
                self.nx, self.ny, self.nz,
                self.dx, self.dy, self.dz_c, self.dz_f,
                self.nu
            )
        else:
            # Original separate kernels (CPU fallback or debugging)
            # Compute advection terms
            adv_u = advection_u(self.u, self.v, self.w, self.nx, self.ny, self.nz,
                               self.dx, self.dy, self.dz_f)
            adv_v = advection_v(self.u, self.v, self.w, self.nx, self.ny, self.nz,
                               self.dx, self.dy, self.dz_f)
            adv_w = advection_w(self.u, self.v, self.w, self.nx, self.ny, self.nz,
                               self.dx, self.dy, self.dz_c)

            # Compute explicit diffusion in x and y only
            diff_xy_u = diffusion_xy_u(self.u, self.nx, self.ny, self.nz,
                                       self.dx, self.dy, self.nu)
            diff_xy_v = diffusion_xy_v(self.v, self.nx, self.ny, self.nz,
                                       self.dx, self.dy, self.nu)
            diff_xy_w = diffusion_xy_w(self.w, self.nx, self.ny, self.nz,
                                       self.dx, self.dy, self.nu)

            # Explicit RHS: diffusion_xy - advection
            rhs_u_explicit = diff_xy_u - adv_u
            rhs_v_explicit = diff_xy_v - adv_v
            rhs_w_explicit = diff_xy_w - adv_w

        # Add bulk forcing (pressure gradient) to RHS (after if-else)
        # This matches .susa where add_mom_forcing is called before time integration
        rhs_u_explicit += self.forcing

        # Store in current buffer (reuse allocation) - MOVED OUTSIDE if-else
        if self.rhs_u_curr is None:
            self.rhs_u_curr = rhs_u_explicit
            self.rhs_v_curr = rhs_v_explicit
            self.rhs_w_curr = rhs_w_explicit
        else:
            self.rhs_u_curr[:] = rhs_u_explicit
            self.rhs_v_curr[:] = rhs_v_explicit
            self.rhs_w_curr[:] = rhs_w_explicit

        # Standard AB2
        if self.rhs_u_prev is None:
            self.u += dt * self.rhs_u_curr
            self.v += dt * self.rhs_v_curr
            self.w += dt * self.rhs_w_curr
        else:
            self.u += dt * (1.5 * self.rhs_u_curr - 0.5 * self.rhs_u_prev)
            self.v += dt * (1.5 * self.rhs_v_curr - 0.5 * self.rhs_v_prev)
            self.w += dt * (1.5 * self.rhs_w_curr - 0.5 * self.rhs_w_prev)

        # Swap buffers: prev <- curr (pointer swap, no memory allocation)
        self.rhs_u_prev, self.rhs_u_curr = self.rhs_u_curr, self.rhs_u_prev
        self.rhs_v_prev, self.rhs_v_curr = self.rhs_v_curr, self.rhs_v_prev
        self.rhs_w_prev, self.rhs_w_curr = self.rhs_w_curr, self.rhs_w_prev
        # Solve (I - dt*nu*d²/dz²)u^(n+1) = u^*
        # where u^* is the velocity after explicit update
        self.apply_bc_uvw()  # Fused boundary conditions

        self.u = solve_implicit_diffusion_u(self.u, dt, self.nx, self.ny, self.nz,
                                            self.dz_c, self.dz_f, self.nu, top_wall_bc_type=self.top_wall_bc_type)
        self.v = solve_implicit_diffusion_v(self.v, dt, self.nx, self.ny, self.nz,
                                            self.dz_c, self.dz_f, self.nu, top_wall_bc_type=self.top_wall_bc_type)
        self.w = solve_implicit_diffusion_w(self.w, dt, self.nx, self.ny, self.nz,
                                            self.dz_c, self.dz_f, self.nu)

        # Update ghost cells for intermediate velocity before divergence computation
        self.apply_bc_uvw()  # Fused boundary conditions

        # ========== PROJECTION STEP ==========
        # Step 1: Compute divergence of intermediate velocity
        div = compute_divergence(self.u, self.v, self.w, self.nx, self.ny, self.nz,
                                 self.dx, self.dy, self.dz_f)

        # Step 2: Solve Poisson equation for pressure: ∇²p = div/dt
        if self.solver_type == 'direct':
            self.p = solve_poisson(self.poisson_matrix, div / dt, self.nx, self.ny, self.nz, self.top_wall_bc_type)
        elif self.solver_type == 'fft':
            self.p = solve_poisson_fft(div / dt, self.fft_data)

        # Diagnostic: check pressure solution (only every 100 steps to minimize GPU-CPU sync overhead)
        if self.current_step % 100 == 0:
            if torch.any(torch.isnan(self.p)) or torch.any(torch.isinf(self.p)):
                print(f"WARNING: Pressure has NaN or Inf values at step {self.current_step}!", flush=True)

        # Step 3: Project velocity to divergence-free field: u = u* - dt*∇p
        self.u, self.v, self.w = project_velocity(self.u, self.v, self.w, self.p,
                                                   self.nx, self.ny, self.nz,
                                                   self.dx, self.dy, self.dz_c, self.dz_f, dt)

        # Reapply boundary conditions after projection
        self.apply_bc_uvw()  # Fused boundary conditions

        # Update Bulk Forcing (Lagged, matches .susa)
        # Calculate u_bulk using the NEW velocity (at end of step)
        u_bulk_current = compute_bulk_velocity(self.u, self.cell_vol_ratio, self.total_volume)
        
        # Update forcing for NEXT step (PI Controller)
        # .susa uses damp=10.0 with Crank-Nicholson. We use a relaxation factor.
        # forcing += (target - current) / timescale
        # We want to correct the error over ~10 time steps to be stable.
        relaxation = 0.1
        self.forcing += (self.U_bulk - u_bulk_current) / dt * relaxation

        return u_bulk_current, self.forcing

    def step_rk3(self, dt):
        """
        Low-storage RK3 time stepping scheme (FUTURE IMPLEMENTATION).

        This is a placeholder for a low-storage Runge-Kutta 3 scheme
        to be implemented with user-provided coefficients.

        Low-storage RK3 schemes require only 2 storage locations for the solution
        and intermediate stages, making them memory-efficient for large simulations.

        Typical schemes:
        - Williamson (1980) RK3
        - Wray RK3
        - Other low-storage variants

        Args:
            dt: Time step size

        Returns:
            u_bulk_current: Current bulk velocity
            forcing: Applied forcing term

        TODO: Implement with user-provided RK3 coefficients (alpha, beta, gamma)
        """
        raise NotImplementedError("RK3 scheme not yet implemented. Use 'AB2', 'IMEX', or 'FE' time stepping.")

    def run_simulation(self):
        import time

        # Select time stepping method based on configuration
        if self.time_scheme == 'IMEX':
            step_function = self.step_imex
            scheme_name = "IMEX (AB2 + Implicit z-diffusion)"
        elif self.time_scheme == 'FE':
            step_function = self.step_forward_euler
            scheme_name = "Forward Euler (Explicit)"
        elif self.time_scheme == 'RK3':
            step_function = self.step_rk3
            scheme_name = "Low-Storage RK3 (Future Implementation)"
        else:
            raise ValueError(f"Unknown time stepping scheme: {self.time_scheme}. Use 'IMEX', 'FE', or 'RK3' (future).")

        # Compute friction velocity from Re_tau for grid display
        delta = self.Lz / 2.0  # Half-channel height
        u_tau_target = self.Re_tau * self.nu / delta

        # Compute grid spacings in friction units
        dz_min = torch.min(self.dz_f).item()
        dz_max = torch.max(self.dz_f).item()
        dx_plus = self.dx * u_tau_target / self.nu
        dy_plus = self.dy * u_tau_target / self.nu
        dz_min_plus = dz_min * u_tau_target / self.nu
        dz_max_plus = dz_max * u_tau_target / self.nu

        # Print initial header
        print("="*90, flush=True)
        print(f"Time stepping scheme: {scheme_name}", flush=True)
        print(f"Grid: {self.nx}×{self.ny}×{self.nz}  |  Domain: {self.Lx:.2f}×{self.Ly:.2f}×{self.Lz:.2f}  |  " +
              f"dx={self.dx:.3f}, dy={self.dy:.3f}, dz_min={dz_min:.3f}, dz_max={dz_max:.3f}  |  " +
              f"dx⁺={dx_plus:.1f}, dy⁺={dy_plus:.1f}, dz⁺_min={dz_min_plus:.1f}, dz⁺_max={dz_max_plus:.1f}", flush=True)
        print("="*90, flush=True)
        print(f"{'Step':>6} {'Time':>10} {'dt':>10} {'max(div)':>12} {'u_bulk':>10} {'u_tau':>10} {'forcing':>12}", flush=True)
        print("="*90, flush=True)
        
        # Wall-time tracking
        start_time = time.time()
        last_walltime_print = start_time

        # Time series data collection (pre-allocated for efficiency)
        # Allocate arrays for data between saves (chunk size: n_save // n_out + 1)
        chunk_size = self.n_save // self.n_out + 1
        timeseries_data = {
            'step': np.zeros(chunk_size, dtype=np.int32),
            'time': np.zeros(chunk_size, dtype=np.float64),
            'u_bulk': np.zeros(chunk_size, dtype=np.float64),
            'u_tau': np.zeros(chunk_size, dtype=np.float64),
            'forcing': np.zeros(chunk_size, dtype=np.float64),
            'index': 0  # Current fill index
        }
        
        step = self.initial_step
        while step < self.n_steps and self.time < self.t_max:
            step += 1
            self.current_step = step  # Update instance variable for diagnostics
            # Print header every 10*n_out steps (but not at step 0, already printed)
            if step > 0 and step % (10 * self.n_out) == 0:
                print(f"{'Step':>6} {'Time':>10} {'dt':>10} {'max(div)':>12} {'u_bulk':>10} {'u_tau':>10} {'forcing':>12}", flush=True)

                # Print wall-time every 10*n_out steps
                current_time = time.time()
                elapsed = current_time - last_walltime_print
                total_elapsed = current_time - start_time
                print(f"  Wall-time: {elapsed:.2f}s (last {10*self.n_out} steps), {total_elapsed:.2f}s (total)", flush=True)
                last_walltime_print = current_time

            # Adaptive timestep: update dt based on CFL condition
            if self.dt_update_interval > 0 and step % self.dt_update_interval == 0 and step > 0:
                dt_new = self.compute_cfl_dt()
                # Only update if change is significant (>5%)
                if abs(dt_new - self.dt) / self.dt > 0.05:
                    self.dt = dt_new
            
            # Run timestep and get diagnostics
            u_bulk, forcing = step_function(self.dt)

            # Update total simulated time
            self.time += self.dt

            # Compute diagnostics only when needed (every n_out steps)
            # This avoids GPU-CPU synchronization on every iteration
            if step % self.n_out == 0:
                # Compute diagnostics (triggers GPU-CPU sync)
                div_final = compute_divergence(self.u, self.v, self.w, self.nx, self.ny, self.nz,
                                                self.dx, self.dy, self.dz_f)
                max_div_tensor = torch.max(torch.abs(div_final))
                u_tau = compute_u_tau(self.u, self.z_c, self.nu, top_wall_bc_type=self.top_wall_bc_type)

                # Batch GPU-CPU transfer: collect all diagnostic scalars and transfer once
                # This reduces ~4 GPU-CPU sync points to 1
                u_bulk_t = u_bulk if torch.is_tensor(u_bulk) else torch.tensor(u_bulk, device=max_div_tensor.device)
                u_tau_t = u_tau if torch.is_tensor(u_tau) else torch.tensor(u_tau, device=max_div_tensor.device)
                forcing_t = forcing if torch.is_tensor(forcing) else torch.tensor(forcing, device=max_div_tensor.device)
                diag = torch.stack([max_div_tensor, u_bulk_t, u_tau_t, forcing_t])
                diag_cpu = diag.cpu().tolist()
                max_div = diag_cpu[0]
                u_bulk_scalar = diag_cpu[1]
                u_tau_scalar = diag_cpu[2]
                forcing_scalar = diag_cpu[3]

                # Collect time series data (pre-allocated array indexing)
                idx = timeseries_data['index']
                timeseries_data['step'][idx] = step
                timeseries_data['time'][idx] = self.time
                timeseries_data['u_bulk'][idx] = u_bulk_scalar
                timeseries_data['u_tau'][idx] = u_tau_scalar
                timeseries_data['forcing'][idx] = forcing_scalar
                timeseries_data['index'] += 1

            # Collect turbulence statistics if enabled and conditions met
            if self.n_stats > 0 and self.time >= self.t_stats and step % self.n_stats == 0:
                # Use u_tau from diagnostics if just computed, otherwise compute it
                # Keep u_tau as tensor on device to avoid GPU-CPU transfer
                if step % self.n_out == 0:
                    # u_tau already computed in diagnostics block above, still a tensor
                    u_tau_for_stats = u_tau
                else:
                    # Compute u_tau (stays on device as tensor)
                    u_tau_for_stats = compute_u_tau(self.u, self.z_c, self.nu, top_wall_bc_type=self.top_wall_bc_type)

                # Print message only when starting collection
                if self.turbulence_stats.n_samples == 0:
                    print(f"  [Stats] Starting statistics collection at t = {self.time:.3f}", flush=True)

                # Accumulate statistics (all operations stay on device)
                self.turbulence_stats.accumulate_statistics(self.u, self.v, self.w, u_tau_for_stats)

            # Save time series data and flow fields every n_save steps
            if step % self.n_save == 0:
                # Check for NaN/Inf in solution (only at save points to minimize overhead)
                if (torch.any(torch.isnan(self.u)) or torch.any(torch.isinf(self.u)) or
                    torch.any(torch.isnan(self.v)) or torch.any(torch.isinf(self.v)) or
                    torch.any(torch.isnan(self.w)) or torch.any(torch.isinf(self.w)) or
                    torch.any(torch.isnan(self.p)) or torch.any(torch.isinf(self.p))):
                    print(f"\n{'='*90}", flush=True)
                    print(f"ERROR: NaN or Inf detected in solution at step {step}, time = {self.time:.6f}", flush=True)
                    print(f"{'='*90}\n", flush=True)
                    # Save current state before breaking
                    u_tau_error = compute_u_tau(self.u, self.z_c, self.nu, top_wall_bc_type=self.top_wall_bc_type)
                    u_bulk_error = compute_bulk_velocity(self.u, self.cell_vol_ratio, self.total_volume)
                    forcing_error = (self.U_bulk - u_bulk_error) / self.dt
                    save_flow_fields(self.u, self.v, self.w, self.p, self.z_c, self.z_f,
                                    self.Lx, self.Ly, step, self.time,
                                    u_tau_error.item() if torch.is_tensor(u_tau_error) else u_tau_error,
                                    forcing_error.item() if torch.is_tensor(forcing_error) else forcing_error,
                                    self.results_folder, 'fields_error.npz')
                    print(f"Error state saved to fields_error.npz", flush=True)
                    break  # Exit simulation loop

                # Compute diagnostics for saving if not already computed this step
                if step % self.n_out != 0:
                    u_tau = compute_u_tau(self.u, self.z_c, self.nu, top_wall_bc_type=self.top_wall_bc_type)
                    # Batch GPU-CPU transfer for save-point diagnostics
                    u_tau_t = u_tau if torch.is_tensor(u_tau) else torch.tensor(u_tau, device=self.device)
                    forcing_t = forcing if torch.is_tensor(forcing) else torch.tensor(forcing, device=self.device)
                    save_diag = torch.stack([u_tau_t, forcing_t]).cpu().tolist()
                    u_tau_scalar = save_diag[0]
                    forcing_scalar = save_diag[1]

                # Save accumulated time series data to binary file
                if timeseries_data['index'] > 0:
                    import os
                    npz_file = os.path.join(self.results_folder, 'timeseries.npz')

                    # Extract filled portion of arrays
                    n_filled = timeseries_data['index']
                    chunk_step = timeseries_data['step'][:n_filled]
                    chunk_time = timeseries_data['time'][:n_filled]
                    chunk_u_bulk = timeseries_data['u_bulk'][:n_filled]
                    chunk_u_tau = timeseries_data['u_tau'][:n_filled]
                    chunk_forcing = timeseries_data['forcing'][:n_filled]

                    # Append to existing file or create new one
                    if os.path.exists(npz_file):
                        # Load existing data
                        existing = np.load(npz_file)
                        # Concatenate with new chunk
                        all_step = np.concatenate([existing['step'], chunk_step])
                        all_time = np.concatenate([existing['time'], chunk_time])
                        all_u_bulk = np.concatenate([existing['u_bulk'], chunk_u_bulk])
                        all_u_tau = np.concatenate([existing['u_tau'], chunk_u_tau])
                        all_forcing = np.concatenate([existing['forcing'], chunk_forcing])
                    else:
                        # First save - use chunk data directly
                        all_step = chunk_step
                        all_time = chunk_time
                        all_u_bulk = chunk_u_bulk
                        all_u_tau = chunk_u_tau
                        all_forcing = chunk_forcing

                    # Save to binary file (compressed for efficiency)
                    np.savez_compressed(npz_file,
                                       step=all_step,
                                       time=all_time,
                                       u_bulk=all_u_bulk,
                                       u_tau=all_u_tau,
                                       forcing=all_forcing)

                    # Reset index for next chunk
                    timeseries_data['index'] = 0

                # Save flow fields (use scalar versions)
                save_flow_fields(self.u, self.v, self.w, self.p, self.z_c, self.z_f,
                                self.Lx, self.Ly, step, self.time, u_tau_scalar, forcing_scalar,
                                self.results_folder, 'fields.npz')

                # Save statistics state checkpoint if statistics are being collected
                if self.turbulence_stats is not None and self.turbulence_stats.n_samples > 0:
                    self.turbulence_stats.save_state(self.stats_state_path)

            # Print stats only every n_out steps
            if step % self.n_out == 0:
                print(f"{step:6d} {self.time:10.6f} {self.dt:10.6f} {max_div:12.3e} {u_bulk_scalar:10.6f} {u_tau_scalar:10.6f} {forcing_scalar:12.3e}", flush=True)
        
        # Print header
        total_wall_time = time.time() - start_time
        print(f"{'='*90}", flush=True)
        print(f"Simulation complete: {step} steps, total time = {self.time:.6f}, final dt = {self.dt:.6f}", flush=True)
        print(f"Total wall time: {total_wall_time:.2f}s", flush=True)
        print("="*90 + "\n", flush=True)

        # Finalize and save turbulence statistics if enabled
        if self.turbulence_stats is not None and self.turbulence_stats.n_samples > 0:
            n_samples = self.turbulence_stats.n_samples
            print(f"\nFinalizing turbulence statistics ({n_samples} samples collected)...", flush=True)
            # Save final state checkpoint before finalizing
            self.turbulence_stats.save_state(self.stats_state_path)
            # Save finalized statistics
            self.turbulence_stats.save_statistics(self.stats_output_path)

        u_profile = self.u[0, 0, :]
        plot_profile(u_profile, self.z_c, 'u', 'z', 'Final velocity profile',
                     'u_profile_final.png', self.results_folder)

        # Final save at end of simulation
        u_tau_final = compute_u_tau(self.u, self.z_c, self.nu, top_wall_bc_type=self.top_wall_bc_type)
        u_bulk_final = compute_bulk_velocity(self.u, self.cell_vol_ratio, self.total_volume)
        forcing_final = (self.U_bulk - u_bulk_final) / self.dt
        save_flow_fields(self.u, self.v, self.w, self.p, self.z_c, self.z_f,
                        self.Lx, self.Ly, step, self.time, u_tau_final, forcing_final,
                        self.results_folder, 'fields_final.npz')