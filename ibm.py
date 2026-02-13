import numpy as np
import torch
import math
import os

class IBM_RKPM:
    """
    Immersed Boundary Method using Reproducing Kernel Particle Method (RKPM).
    Re-implemented based on Pinelli et al. (2010) and Roma et al. (1999) kernel.
    
    Key Features:
    - Roma Kernel (Support 1.5h)
    - Quadratic Basis (10 terms)
    - Moment Matrix with Volume Integration
    - Direct Epsilon Solver for Partition of Unity
    """
    def __init__(self, config, grid_data, device='cpu'):
        self.config = config
        self.device = device
        
        # Grid data (staggered)
        self.x_c = grid_data['x_c']
        self.x_f = grid_data['x_f']
        self.y_c = grid_data['y_c']
        self.y_f = grid_data['y_f']
        self.z_c = grid_data['z_c']
        self.z_f = grid_data['z_f']
        
        # Grid spacings (arrays for z)
        self.dz_c = grid_data['dz_c']
        self.dz_f = grid_data['dz_f']
        
        self.dx = grid_data['dx']
        self.dy = grid_data['dy']
        
        self.nx = len(self.x_c)
        self.ny = len(self.y_c)
        self.nz = len(self.z_c)
        
        # IBM parameters
        ibm_config = config['ibm']
        obstacle_type = ibm_config.get('obstacle_type', 'sphere')

        if obstacle_type == 'sphere':
            sphere_config = ibm_config['sphere']
            self.radius = sphere_config['radius']
            self.center = np.array(sphere_config['center'])
            self.n_lag = sphere_config.get('n_points', 2000)

            # Check for precomputed Thomson-distributed points
            precomputed_file = 'sphere_lagrangian_points.npz'
            if os.path.exists(precomputed_file):
                print(f"Loading precomputed Lagrangian points from {precomputed_file}...", flush=True)
                data = np.load(precomputed_file)
                self.x_lag = data['x']
                self.y_lag = data['y']
                self.z_lag = data['z']
                self.dS = data['dS']
                self.n_lag = len(self.x_lag)

                # Verify compatibility
                loaded_center = data['center']
                loaded_radius = data['radius']
                if not np.allclose(loaded_center, self.center, atol=1e-6):
                    print(f"WARNING: Precomputed center {loaded_center} differs from config {self.center}", flush=True)
                    print(f"         Regenerating points using Fibonacci sphere...", flush=True)
                    self.x_lag, self.y_lag, self.z_lag, self.dS = self.generate_sphere_points(
                        self.radius, self.center, self.n_lag
                    )
                elif not np.isclose(loaded_radius, self.radius, atol=1e-6):
                    print(f"WARNING: Precomputed radius {loaded_radius} differs from config {self.radius}", flush=True)
                    print(f"         Regenerating points using Fibonacci sphere...", flush=True)
                    self.x_lag, self.y_lag, self.z_lag, self.dS = self.generate_sphere_points(
                        self.radius, self.center, self.n_lag
                    )
                else:
                    print(f"  Loaded {self.n_lag} Thomson-distributed points", flush=True)
                    if 'cv' in data:
                        print(f"  Uniformity CV: {data['cv']:.6f}", flush=True)
                    if 'target_spacing' in data:
                        print(f"  Target spacing: {data['target_spacing']:.6f}", flush=True)
            else:
                print(f"No precomputed points found. Using Fibonacci sphere generation...", flush=True)
                # Generate Lagrangian points for sphere using Fibonacci
                self.x_lag, self.y_lag, self.z_lag, self.dS = self.generate_sphere_points(
                    self.radius, self.center, self.n_lag
                )

        elif obstacle_type == 'cube':
            cube_config = ibm_config['cube']
            self.dimensions = np.array(cube_config['dimensions'])
            self.center = np.array(cube_config['center'])
            n_points_per_face = cube_config.get('n_points_per_face', 300)

            # Generate Lagrangian points for cube
            self.x_lag, self.y_lag, self.z_lag, self.dS = self.generate_cube_points(
                self.dimensions, self.center, n_points_per_face
            )
            self.n_lag = len(self.x_lag)

        elif obstacle_type == 'canopy':
            # Import canopy generator
            from cylinder_lagrangian_generator import generate_canopy_lagrangian_points

            print("=" * 80, flush=True)
            print("CANOPY INITIALIZATION", flush=True)
            print("=" * 80, flush=True)

            # Generate all canopy filaments and Lagrangian points
            self.x_lag, self.y_lag, self.z_lag, self.dS, self.canopy_centers = \
                generate_canopy_lagrangian_points(config)

            self.n_lag = len(self.x_lag)
            n_filaments = len(self.canopy_centers)

            print(f"\nCanopy summary:", flush=True)
            print(f"  Filaments: {n_filaments}", flush=True)
            print(f"  Lagrangian points: {self.n_lag}", flush=True)
            print(f"  Points per filament: {self.n_lag / n_filaments:.1f}", flush=True)
            print("=" * 80, flush=True)

        else:
            raise ValueError(f"Unsupported obstacle type: '{obstacle_type}'. Use 'sphere', 'cube', or 'canopy'.")
        
        # Move to device
        self.x_lag_t = torch.tensor(self.x_lag, device=self.device, dtype=torch.float64)
        self.y_lag_t = torch.tensor(self.y_lag, device=self.device, dtype=torch.float64)
        self.z_lag_t = torch.tensor(self.z_lag, device=self.device, dtype=torch.float64)
        self.dS_t = torch.tensor(self.dS, device=self.device, dtype=torch.float64)
        
        # Load precomputed RKPM coefficients
        coefficients_file = config.get('ibm', {}).get('rkpm_coefficients_file', None)
        epsilon_file = config.get('ibm', {}).get('rkpm_epsilon_file', None)

        if coefficients_file is None or epsilon_file is None:
            raise ValueError(
                "RKPM coefficient files not specified in config!\n"
                "Please run: python compute_rkpm_coefficients.py config.yaml\n"
                "or mpirun -np N python compute_rkpm_coefficients.py config.yaml\n"
                "Then add to config:\n"
                "  ibm:\n"
                "    rkpm_coefficients_file: 'results/rkpm_coefficients.npz'\n"
                "    rkpm_epsilon_file: 'results/rkpm_epsilon.npz'"
            )

        print("Loading precomputed RKPM coefficients...", flush=True)
        support_u_list, support_v_list, support_w_list = self._load_rkpm_coefficients(coefficients_file)
        print("✓ Loaded RKPM coefficients", flush=True)

        print("Loading precomputed epsilon values...", flush=True)
        self.epsilon_u, self.epsilon_v, self.epsilon_w = self._load_epsilon(epsilon_file)
        print("✓ Loaded epsilon values", flush=True)

        # Vectorize support structures for GPU
        print("Vectorizing support structures for GPU...", flush=True)
        self.support_u = self.vectorize_support(support_u_list)
        self.support_v = self.vectorize_support(support_v_list)
        self.support_w = self.vectorize_support(support_w_list)

        print("RKPM coefficients and epsilon computed.", flush=True)

    def generate_sphere_points(self, radius, center, n_points):
        indices = np.arange(0, n_points, dtype=float) + 0.5
        phi = np.arccos(1 - 2*indices/n_points)
        theta = np.pi * (1 + 5**0.5) * indices

        x = center[0] + radius * np.cos(theta) * np.sin(phi)
        y = center[1] + radius * np.sin(theta) * np.sin(phi)
        z = center[2] + radius * np.cos(phi)

        total_area = 4 * np.pi * radius**2
        dS = np.full(n_points, total_area / n_points)

        return x, y, z, dS

    def generate_cube_points(self, dimensions, center, n_points_per_face):
        # Simplified cube generation (same as before)
        dx, dy, dz = dimensions
        xc, yc, zc = center
        
        # ... (Implementation omitted for brevity, assuming standard logic or copied from previous)
        # For robustness, let's include a basic implementation
        points = []
        areas = []
        
        # Helper to grid a face
        def grid_face(c1_range, c2_range, axis_fixed, fixed_val, area_total):
            n1 = int(np.sqrt(n_points_per_face * (c1_range[1]-c1_range[0]) / (c2_range[1]-c2_range[0])))
            n2 = int(n_points_per_face / n1)
            d1 = np.linspace(c1_range[0], c1_range[1], n1+2)[1:-1]
            d2 = np.linspace(c2_range[0], c2_range[1], n2+2)[1:-1]
            dA = area_total / (len(d1)*len(d2))
            for v1 in d1:
                for v2 in d2:
                    p = [0,0,0]
                    if axis_fixed == 0: p = [fixed_val, v1, v2]
                    elif axis_fixed == 1: p = [v1, fixed_val, v2]
                    else: p = [v1, v2, fixed_val]
                    points.append(p)
                    areas.append(dA)

        # X faces
        grid_face([yc-dy/2, yc+dy/2], [zc-dz/2, zc+dz/2], 0, xc-dx/2, dy*dz)
        grid_face([yc-dy/2, yc+dy/2], [zc-dz/2, zc+dz/2], 0, xc+dx/2, dy*dz)
        # Y faces
        grid_face([xc-dx/2, xc+dx/2], [zc-dz/2, zc+dz/2], 1, yc-dy/2, dx*dz)
        grid_face([xc-dx/2, xc+dx/2], [zc-dz/2, zc+dz/2], 1, yc+dy/2, dx*dz)
        # Z faces
        grid_face([xc-dx/2, xc+dx/2], [yc-dy/2, yc+dy/2], 2, zc-dz/2, dx*dy)
        grid_face([xc-dx/2, xc+dx/2], [yc-dy/2, yc+dy/2], 2, zc+dz/2, dx*dy)
        
        pts = np.array(points)
        return pts[:,0], pts[:,1], pts[:,2], np.array(areas)

    def roma_kernel(self, r, h):
        """
        Roma et al. (1999) Kernel.
        Support: [-1.5h, 1.5h]
        """
        ar = torch.abs(r)
        q = ar / h
        val = torch.zeros_like(ar)
        
        # Region 1: |r| <= 0.5
        mask1 = q <= 0.5
        if torch.any(mask1):
            term1 = torch.sqrt(torch.clamp(1.0 - 3.0 * q[mask1]**2, min=0.0))
            val[mask1] = (1.0/3.0) * (1.0 + term1)
            
        # Region 2: 0.5 < |r| <= 1.5
        mask2 = (q > 0.5) & (q <= 1.5)
        if torch.any(mask2):
            term2 = torch.sqrt(torch.clamp(1.0 - 3.0 * (1.0 - q[mask2])**2, min=0.0))
            val[mask2] = (1.0/6.0) * (5.0 - 3.0 * q[mask2] - term2)
            
        return val / h

    def _scalar_roma_kernel(self, r, h):
        """Scalar version of Roma kernel."""
        ar = abs(r)
        q = ar / h
        if q <= 0.5:
            return (1.0/3.0) * (1.0 + math.sqrt(max(0.0, 1.0 - 3.0 * q**2))) / h
        elif q <= 1.5:
            return (1.0/6.0) * (5.0 - 3.0 * q - math.sqrt(max(0.0, 1.0 - 3.0 * (1.0 - q)**2))) / h
        return 0.0

    def compute_rkpm_coefficients(self, x_grid, y_grid, z_grid, component, dz_grid=None):
        """
        Compute RKPM coefficients (b vectors) and weights.

        Optimization options (set in config['ibm']):
        - rkpm_basis: 'linear' (4 terms, faster) or 'quadratic' (10 terms, accurate)
        - rkpm_search_range: int (default 4, use 6 for higher accuracy)
        """
        # Check for optimization flags
        use_linear = self.config.get('ibm', {}).get('rkpm_basis', 'linear') == 'linear'
        search_range = self.config.get('ibm', {}).get('rkpm_search_range', 4)

        n_basis = 4 if use_linear else 10
        support_list = []

        print(f"  RKPM settings: basis={'linear' if use_linear else 'quadratic'} ({n_basis} terms), "
              f"search_range={search_range}", flush=True)
        
        nx_g, ny_g, nz_g = len(x_grid), len(y_grid), len(z_grid)
        
        # Ensure we have NumPy arrays for preprocessing (convert torch tensors if needed)
        x_grid_np = x_grid.cpu().numpy() if torch.is_tensor(x_grid) else np.array(x_grid)
        y_grid_np = y_grid.cpu().numpy() if torch.is_tensor(y_grid) else np.array(y_grid)
        z_grid_np = z_grid.cpu().numpy() if torch.is_tensor(z_grid) else np.array(z_grid)
        dz_grid_np = dz_grid.cpu().numpy() if dz_grid is not None and torch.is_tensor(dz_grid) else (np.array(dz_grid) if dz_grid is not None else None)

        # Convert grid to tensors for later use
        x_grid_t = torch.tensor(x_grid_np, device=self.device, dtype=torch.float64)
        y_grid_t = torch.tensor(y_grid_np, device=self.device, dtype=torch.float64)
        z_grid_t = torch.tensor(z_grid_np, device=self.device, dtype=torch.float64)

        # Progress reporting
        progress_interval = max(1, self.n_lag // 20)  # Report 20 times
        print(f"  Processing {self.n_lag} Lagrangian points...", flush=True)

        for i in range(self.n_lag):
            # Progress update
            if (i + 1) % progress_interval == 0 or i == self.n_lag - 1:
                print(f"    Progress: {i+1}/{self.n_lag} ({(i+1)/self.n_lag*100:.1f}%)", flush=True)
            xl, yl, zl = self.x_lag[i], self.y_lag[i], self.z_lag[i]

            # 1. Find nearest grid point
            idx_x = (np.abs(x_grid_np - xl)).argmin()
            idx_y = (np.abs(y_grid_np - yl)).argmin()
            idx_z = (np.abs(z_grid_np - zl)).argmin()

            # 2. Determine support size (h) - Adaptive
            def get_h(grid, idx, dist):
                spacings = []
                if idx > 0: spacings.append(abs(grid[idx] - grid[idx-1]))
                if idx < len(grid)-1: spacings.append(abs(grid[idx+1] - grid[idx]))
                if not spacings: return 1.0
                R_min, R_max = min(spacings), max(spacings)
                return ((5.0 * R_max + R_min) / 6.0 + dist / 9.0) * 1.5 # Safety factor

            hx = get_h(x_grid_np, idx_x, abs(x_grid_np[idx_x] - xl))
            hy = get_h(y_grid_np, idx_y, abs(y_grid_np[idx_y] - yl))
            hz = get_h(z_grid_np, idx_z, abs(z_grid_np[idx_z] - zl))
            
            # 3. Find neighbors in support cage
            # Roma support is 1.5h. Cage should be slightly larger.
            support_radius = 1.5
            # search_range set at function level (line 258)
            
            ix_min = max(0, idx_x - search_range)
            ix_max = min(nx_g, idx_x + search_range + 1)
            iy_min = max(0, idx_y - search_range)
            iy_max = min(ny_g, idx_y + search_range + 1)
            iz_min = max(0, idx_z - search_range)
            iz_max = min(nz_g, idx_z + search_range + 1)
            
            neighbors = []
            M = np.zeros((n_basis, n_basis))

            # Scaling matrix H_inv (depends on basis order)
            if use_linear:
                # Linear basis: [1, x, y, z]
                H_inv = np.array([1.0, 1.0/hx, 1.0/hy, 1.0/hz])
            else:
                # Quadratic basis: [1, x, y, z, xy, xz, yz, x^2, y^2, z^2]
                H_inv = np.array([
                    1.0,
                    1.0/hx, 1.0/hy, 1.0/hz,
                    1.0/(hx*hy), 1.0/(hx*hz), 1.0/(hy*hz),
                    1.0/(hx**2), 1.0/(hy**2), 1.0/(hz**2)
                ])
            
            for iz in range(iz_min, iz_max):
                dz_val = z_grid_np[iz] - zl
                if abs(dz_val) > support_radius * hz: continue
                val_z = self._scalar_roma_kernel(dz_val, hz)
                if val_z == 0: continue

                # Cell height
                if dz_grid_np is not None: dz_cell = dz_grid_np[iz]
                elif iz < nz_g - 1: dz_cell = z_grid_np[iz+1] - z_grid_np[iz]
                else: dz_cell = z_grid_np[iz] - z_grid_np[iz-1]

                for iy in range(iy_min, iy_max):
                    dy_val = y_grid_np[iy] - yl
                    if abs(dy_val) > support_radius * hy: continue
                    val_y = self._scalar_roma_kernel(dy_val, hy)
                    if val_y == 0: continue

                    for ix in range(ix_min, ix_max):
                        dx_val = x_grid_np[ix] - xl
                        if abs(dx_val) > support_radius * hx: continue
                        val_x = self._scalar_roma_kernel(dx_val, hx)
                        if val_x == 0: continue
                        
                        # Kernel value (scalar)
                        phi = val_x * val_y * val_z
                        
                        # Volume Integration
                        vol = self.dx * self.dy * dz_cell

                        # Scaled Basis P_scaled
                        sx, sy, sz = dx_val/hx, dy_val/hy, dz_val/hz
                        if use_linear:
                            P_scaled = np.array([1.0, sx, sy, sz])
                        else:
                            P_scaled = np.array([
                                1.0,
                                sx, sy, sz,
                                sx*sy, sx*sz, sy*sz,
                                sx**2, sy**2, sz**2
                            ])
                        
                        # Accumulate M
                        M += np.outer(P_scaled, P_scaled) * phi * vol
                        
                        neighbors.append({
                            'ix': ix, 'iy': iy, 'iz': iz,
                            'dx': dx_val, 'dy': dy_val, 'dz': dz_val,
                            'vol': vol,
                            'phi': phi
                        })
            
            # Solve M * c = [1, 0, ...]^T
            rhs = np.zeros(n_basis)
            rhs[0] = 1.0
            
            # Regularize M if needed
            if np.linalg.cond(M) > 1e12:
                M += np.eye(n_basis) * 1e-8
            
            try:
                c = np.linalg.solve(M, rhs)
            except np.linalg.LinAlgError:
                c = np.zeros(n_basis); c[0] = 1.0
                print(f"Warning: Singular M at lag point {i}")

            # Transform back to b: b = c * H_inv
            b = c * H_inv
            
            # Store data for GPU
            ix_t = torch.tensor([n['ix'] for n in neighbors], dtype=torch.long, device=self.device)
            iy_t = torch.tensor([n['iy'] for n in neighbors], dtype=torch.long, device=self.device)
            iz_t = torch.tensor([n['iz'] for n in neighbors], dtype=torch.long, device=self.device)
            vol_t = torch.tensor([n['vol'] for n in neighbors], dtype=torch.float64, device=self.device)
            b_t = torch.tensor(b, dtype=torch.float64, device=self.device)
            
            # Compute final weights on GPU
            xn, yn, zn = x_grid_t[ix_t], y_grid_t[iy_t], z_grid_t[iz_t]
            dx, dy, dz = xn - self.x_lag_t[i], yn - self.y_lag_t[i], zn - self.z_lag_t[i]
            
            kx = self.roma_kernel(dx, hx)
            ky = self.roma_kernel(dy, hy)
            kz = self.roma_kernel(dz, hz)
            phi_t = kx * ky * kz
            
            # Unscaled Basis P
            ones = torch.ones_like(dx)
            if use_linear:
                P = torch.stack([ones, dx, dy, dz], dim=1)
            else:
                P = torch.stack([
                    ones,
                    dx, dy, dz,
                    dx*dy, dx*dz, dy*dz,
                    dx**2, dy**2, dz**2
                ], dim=1)
            
            # Corrected weight: w = (P . b) * phi
            correction = torch.matmul(P, b_t)
            wdt = correction * phi_t
            
            support_list.append({
                'ix': ix_t, 'iy': iy_t, 'iz': iz_t,
                'wdt': wdt,
                'vol': vol_t
            })
            
        return support_list

    def _load_rkpm_coefficients(self, filepath):
        """Load RKPM coefficients from NPZ file (precomputed)."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"RKPM coefficients file not found: {filepath}\n"
                "Please run: python compute_rkpm_coefficients.py config.yaml"
            )

        try:
            data = np.load(filepath, allow_pickle=False)

            # Check n_lag matches
            if int(data['n_lag']) != self.n_lag:
                raise ValueError(
                    f"Number of Lagrangian points mismatch!\n"
                    f"  Precomputed file: {data['n_lag']}\n"
                    f"  Current config: {self.n_lag}\n"
                    "Please recompute coefficients with current config."
                )

            # Unflatten support lists
            def unflatten_support(prefix):
                """Reconstruct list of dicts from flat arrays."""
                ix_flat = data[f'{prefix}_ix']
                iy_flat = data[f'{prefix}_iy']
                iz_flat = data[f'{prefix}_iz']
                wdt_flat = data[f'{prefix}_wdt']
                vol_flat = data[f'{prefix}_vol']
                n_neighbors = data[f'{prefix}_n_neighbors']

                support_list = []
                offset = 0
                for n in n_neighbors:
                    support_list.append({
                        'ix': torch.tensor(ix_flat[offset:offset+n], dtype=torch.long, device=self.device),
                        'iy': torch.tensor(iy_flat[offset:offset+n], dtype=torch.long, device=self.device),
                        'iz': torch.tensor(iz_flat[offset:offset+n], dtype=torch.long, device=self.device),
                        'wdt': torch.tensor(wdt_flat[offset:offset+n], dtype=torch.float64, device=self.device),
                        'vol': torch.tensor(vol_flat[offset:offset+n], dtype=torch.float64, device=self.device)
                    })
                    offset += n

                return support_list

            support_u = unflatten_support('u')
            support_v = unflatten_support('v')
            support_w = unflatten_support('w')

            file_size = os.path.getsize(filepath) / 1e6
            print(f"  File size: {file_size:.1f} MB", flush=True)

            return support_u, support_v, support_w

        except Exception as e:
            raise RuntimeError(f"Error loading RKPM coefficients: {e}")

    def _load_epsilon(self, filepath):
        """Load epsilon values from NPZ file (precomputed)."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Epsilon file not found: {filepath}\n"
                "Please run: python compute_rkpm_coefficients.py config.yaml"
            )

        try:
            data = np.load(filepath, allow_pickle=False)

            # Check n_lag matches
            if int(data['n_lag']) != self.n_lag:
                raise ValueError(
                    f"Number of Lagrangian points mismatch!\n"
                    f"  Precomputed file: {data['n_lag']}\n"
                    f"  Current config: {self.n_lag}\n"
                    "Please recompute epsilon with current config."
                )

            epsilon_u = torch.tensor(data['epsilon_u'], dtype=torch.float64, device=self.device)
            epsilon_v = torch.tensor(data['epsilon_v'], dtype=torch.float64, device=self.device)
            epsilon_w = torch.tensor(data['epsilon_w'], dtype=torch.float64, device=self.device)

            file_size = os.path.getsize(filepath) / 1e6
            print(f"  File size: {file_size:.1f} MB", flush=True)

            return epsilon_u, epsilon_v, epsilon_w

        except Exception as e:
            raise RuntimeError(f"Error loading epsilon: {e}")

    def compute_epsilon_direct(self, support):
        """
        Direct solver for epsilon: A * eps = 1
        A_ij = sum_k (w_ik * w_jk * vol_k)
        """
        # Flatten indices
        if not support:
            print("WARNING: Support list is empty!")
            return torch.ones(self.n_lag, device=self.device, dtype=torch.float64)

        # Check for empty supports
        for i, s in enumerate(support):
            if s['ix'].numel() == 0:
                print(f"WARNING: Lagrangian point {i} has NO neighbors!")
                # This will cause max() to fail below.
                # We should probably skip this point or assign a dummy neighbor?
                # Or better, fix the support generation.
                # For now, let's just ensure we don't crash, but this indicates a physics problem.
                pass

        try:
            max_ix = max(torch.max(s['ix']).item() for s in support if s['ix'].numel() > 0) + 1
            max_iy = max(torch.max(s['iy']).item() for s in support if s['iy'].numel() > 0) + 1
            max_iz = max(torch.max(s['iz']).item() for s in support if s['iz'].numel() > 0) + 1
        except ValueError:
             print("ERROR: All supports are empty!")
             return torch.ones(self.n_lag, device=self.device, dtype=torch.float64)

        
        stride_y = max_ix
        stride_z = max_ix * max_iy
        N_grid = max_ix * max_iy * max_iz
        
        indices_list = []
        values_list = []
        vol_list = [] # We need volume for the inner product
        
        for i in range(self.n_lag):
            data = support[i]
            grid_idx = data['ix'] + data['iy'] * stride_y + data['iz'] * stride_z
            lag_idx = torch.full_like(grid_idx, i)
            
            indices = torch.stack([lag_idx, grid_idx], dim=0)
            indices_list.append(indices)
            values_list.append(data['wdt'])
            vol_list.append(data['vol']) # Store volume associated with each weight
            
        all_indices = torch.cat(indices_list, dim=1)
        all_values = torch.cat(values_list)
        all_vols = torch.cat(vol_list)
        
        # Construct W (N_lag x N_grid)
        W = torch.sparse_coo_tensor(all_indices, all_values, (self.n_lag, N_grid), device=self.device)

        # CRITICAL: According to Pinelli et al. (2010) partition of unity:
        # Interpolation: u_lag_i = sum_k w_ik * ΔV_k * u_eul_k
        # Spreading: f_eul_k = sum_i w_ik * epsilon_i * Δs_i * f_lag_i
        # Partition of unity I(S(1)) = 1 gives:
        #   1 = sum_k w_ik * ΔV_k * (sum_j w_jk * epsilon_j * Δs_j)
        #     = sum_j (Δs_j * sum_k w_ik * w_jk * ΔV_k) * epsilon_j
        # Therefore: A_ij = Δs_j * sum_k (w_ik * w_jk * ΔV_k)
        # See rkpm_explained.md line 172-174

        W_vol = torch.sparse_coo_tensor(all_indices, all_values * all_vols,
                                        (self.n_lag, N_grid), device=self.device)

        # A_base = W @ (W * ΔV).T gives sum_k (w_ik * w_jk * ΔV_k)
        A_base = torch.matmul(W, W_vol.t()).to_dense()

        # Multiply by Δs_j for each column j
        # A_ij = Δs_j * (sum_k w_ik * w_jk * ΔV_k)
        A = A_base * self.dS_t.unsqueeze(0)  # Broadcast dS across rows
        
        # Solve A * eps = 1
        rhs = torch.ones(self.n_lag, device=self.device, dtype=torch.float64)

        # Add regularization for ill-conditioned systems
        # This helps when some Lagrangian points have poor support (corners, edges)
        reg_strength = 1e-6  # Increased regularization
        A.diagonal().add_(reg_strength)

        epsilon = torch.linalg.solve(A, rhs)

        # Report epsilon statistics (no clamping - clamping breaks partition of unity)
        print(f"  Epsilon range: [{epsilon.min():.2f}, {epsilon.max():.2f}], mean={epsilon.mean():.2f}")
        n_extreme = (torch.abs(epsilon) > 10.0).sum().item()
        if n_extreme > 0:
            print(f"  WARNING: {n_extreme}/{self.n_lag} points have |epsilon| > 10")
            print(f"  This indicates ill-conditioning. Use low relaxation (0.05-0.1) and long ramp (50-100 steps)")

        return epsilon

    def vectorize_support(self, support_list):
        """
        Convert list of support dictionaries to flattened tensors for GPU vectorization.

        Returns dict with:
            - ix, iy, iz: flattened grid indices (total_support_points,)
            - wdt: flattened weights (total_support_points,)
            - vol: flattened volumes (total_support_points,)
            - lag_idx: Lagrangian point index for each support point (total_support_points,)
            - offsets: Start index for each Lagrangian point (n_lag+1,)
        """
        all_ix = []
        all_iy = []
        all_iz = []
        all_wdt = []
        all_vol = []
        all_lag_idx = []
        offsets = [0]

        for i, s in enumerate(support_list):
            n_support = len(s['ix'])
            all_ix.append(s['ix'])
            all_iy.append(s['iy'])
            all_iz.append(s['iz'])
            all_wdt.append(s['wdt'])
            all_vol.append(s['vol'])
            all_lag_idx.append(torch.full((n_support,), i, dtype=torch.long, device=self.device))
            offsets.append(offsets[-1] + n_support)

        return {
            'ix': torch.cat(all_ix),
            'iy': torch.cat(all_iy),
            'iz': torch.cat(all_iz),
            'wdt': torch.cat(all_wdt),
            'vol': torch.cat(all_vol),
            'lag_idx': torch.cat(all_lag_idx),
            'offsets': torch.tensor(offsets, dtype=torch.long, device=self.device)
        }

    def interpolate(self, field, component):
        """
        Interpolate velocity to Lagrangian points (GPU-optimized).

        Vectorized implementation that processes all Lagrangian points in parallel.
        """
        if component == 'u':
            support = self.support_u
        elif component == 'v':
            support = self.support_v
        elif component == 'w':
            support = self.support_w
        else:
            raise ValueError(f"Unknown component: {component}")

        # Gather all field values at support points (vectorized)
        field_vals = field[support['ix'], support['iy'], support['iz']]

        # Weight by kernel and volume (vectorized)
        weighted = field_vals * support['wdt'] * support['vol']

        # Sum contributions for each Lagrangian point using scatter_add
        lag_field = torch.zeros(self.n_lag, device=self.device, dtype=torch.float64)
        lag_field.scatter_add_(0, support['lag_idx'], weighted)

        return lag_field

    def spread(self, f_lag, component):
        """
        Spread Lagrangian force to Eulerian grid (GPU-optimized).

        Vectorized implementation using scatter operations for parallel processing.
        """
        # Return field with correct shape (physical domain)
        if component == 'u':
            f_euler = torch.zeros(self.nx+1, self.ny, self.nz, device=self.device, dtype=torch.float64)
            support = self.support_u
            epsilon = self.epsilon_u
        elif component == 'v':
            f_euler = torch.zeros(self.nx, self.ny+1, self.nz, device=self.device, dtype=torch.float64)
            support = self.support_v
            epsilon = self.epsilon_v
        elif component == 'w':
            f_euler = torch.zeros(self.nx, self.ny, self.nz+1, device=self.device, dtype=torch.float64)
            support = self.support_w
            epsilon = self.epsilon_w
        else:
            raise ValueError(f"Unknown component: {component}")

        # CRITICAL: According to Pinelli et al. (2010), spreading formula is:
        # f(x_k) = sum_I (F(X_I) * w_tilde_Ik * epsilon_I * Δs_I)
        # Where Δs_I is the LAGRANGIAN surface element, NOT Eulerian volume!
        # See rkpm_explained.md line 167

        # Scale forces by epsilon and dS for each Lagrangian point (vectorized)
        F_scaled = f_lag[support['lag_idx']] * epsilon[support['lag_idx']] * self.dS_t[support['lag_idx']]

        # Weight by kernel (vectorized)
        terms = F_scaled * support['wdt']

        # Scatter to Eulerian grid using index_put_ with accumulate
        # This accumulates all contributions to each grid point
        f_euler.index_put_((support['ix'], support['iy'], support['iz']), terms, accumulate=True)

        return f_euler
