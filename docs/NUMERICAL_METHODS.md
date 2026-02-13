# TorChannel Numerical Methods

This document describes the governing equations, boundary conditions, and numerical methods used in TorChannel.

---

## Table of Contents

1. [Governing Equations](#governing-equations)
2. [Computational Domain](#computational-domain)
3. [Boundary Conditions](#boundary-conditions)
4. [Spatial Discretization](#spatial-discretization)
5. [Grid Generation](#grid-generation)
6. [Time Integration](#time-integration)
7. [Pressure-Velocity Coupling](#pressure-velocity-coupling)
8. [Bulk Velocity Forcing](#bulk-velocity-forcing)
9. [Turbulence Statistics](#turbulence-statistics)
10. [References](#references)

---

## Governing Equations

TorChannel solves the incompressible Navier-Stokes equations for turbulent channel flow:

### Momentum Equation

$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{f}$$

where:
- $\mathbf{u} = (u, v, w)$: velocity vector
- $p$: pressure (divided by density $\rho$)
- $\nu$: kinematic viscosity
- $\mathbf{f}$: body force (pressure gradient forcing in x-direction)

### Continuity Equation

$$\nabla \cdot \mathbf{u} = 0$$

### Dimensionless Form

The equations are made dimensionless using:
- Length scale: $\delta$ (channel half-height)
- Velocity scale: $U_{\text{bulk}}$ (mean velocity)
- Time scale: $\delta / U_{\text{bulk}}$

This gives:
- Reynolds number: $Re = U_{\text{bulk}} \times L_z / \nu$ (where $L_z = 2\delta$)
- Friction Reynolds number: $Re_\tau = u_\tau \times \delta / \nu$

---

## Computational Domain

### Domain Geometry

```
        z = Lz (top wall, no-slip)
        ┌─────────────────────────┐
        │                         │
        │   Periodic x (length Lx)│
    ┌───│                         │───┐
    │   │   Periodic y (length Ly)│   │
    │   │                         │   │
    └───│                         │───┘
        │                         │
        └─────────────────────────┘
        z = 0 (bottom wall, no-slip)
```

### Dimensions

- **Streamwise (x)**: $0 \leq x \leq L_x$, periodic
- **Spanwise (y)**: $0 \leq y \leq L_y$, periodic
- **Wall-normal (z)**: $0 \leq z \leq L_z$, no-slip walls

Typical domain sizes (normalized by $\delta = L_z/2$):
- $L_x \approx 4\pi\delta$ to $8\pi\delta$
- $L_y \approx 2\pi\delta$ to $4\pi\delta$
- $L_z = 2\delta$ (fixed)

---

## Boundary Conditions

### Velocity Boundary Conditions

**Periodic directions (x, y):**
- $u(x + L_x, y, z) = u(x, y, z)$
- $u(x, y + L_y, z) = u(x, y, z)$

**No-slip walls ($z = 0$, $L_z$):**
- $u(x, y, 0) = 0$
- $u(x, y, L_z) = 0$
- $v(x, y, 0) = 0$
- $v(x, y, L_z) = 0$
- $w(x, y, 0) = 0$
- $w(x, y, L_z) = 0$

### Pressure Boundary Conditions

**Neumann BC at walls:**
- $\left.\frac{\partial p}{\partial z}\right|_{z=0} = 0$
- $\left.\frac{\partial p}{\partial z}\right|_{z=L_z} = 0$

This ensures:
1. Consistency with continuity equation at walls
2. Proper divergence-free projection

**Periodic in x, y:**
- Inherits from velocity periodicity

---

## Spatial Discretization

### Staggered Grid (Arakawa C-Grid)

Velocity components are stored at cell faces:

```
        w(i,j,k+1)
            │
            │
    v(i,j,k)───u(i,j,k)───v(i+1,j,k)
            │
            │ p(i,j,k)
            │
    v(i,j,k)───u(i+1,j,k)───v(i+1,j,k)
            │
            │
        w(i,j,k)
```

- **u**: Face-centered in x (staggered in x)
- **v**: Face-centered in y (staggered in y)
- **w**: Face-centered in z (staggered in z)
- **p**: Cell-centered

### Array Dimensions

Including ghost cells:
- u: (nx+1, ny+2, nz+2)
- v: (nx+2, ny+1, nz+2)
- w: (nx+2, ny+2, nz+1)
- p: (nx+2, ny+2, nz+2)

Ghost cells enforce boundary conditions.

### Finite Difference Operators

**Uniform spacing (x, y directions):**

Derivatives use centered 2nd-order differences:
- $\frac{\partial u}{\partial x} \approx \frac{u_{i+1} - u_{i-1}}{2\Delta x}$
- $\frac{\partial^2 u}{\partial x^2} \approx \frac{u_{i+1} - 2u_i + u_{i-1}}{\Delta x^2}$

**Non-uniform spacing (z direction):**

On stretched grid with variable spacing $\Delta z(k)$:

First derivative:
$$\frac{\partial u}{\partial z} \approx \frac{u_{k+1} - u_{k-1}}{z_{k+1} - z_{k-1}}$$

Second derivative:
$$\frac{\partial^2 u}{\partial z^2} \approx \frac{2}{z_{k+1} - z_{k-1}} \left[ \frac{u_{k+1} - u_k}{z_{k+1} - z_k} - \frac{u_k - u_{k-1}}{z_k - z_{k-1}} \right]$$

### Interpolation

When variables are needed at different locations:

Linear interpolation:
$$u_{\text{center}} = 0.5 \times (u_{\text{left}} + u_{\text{right}})$$

Conservative interpolation for advection:
- Ensures momentum conservation
- Uses face-centered velocities for flux computation

---

## Grid Generation

### Hyperbolic Tangent Stretching

To resolve the near-wall region efficiently, the grid is stretched in z using:

$$\xi(k) = \frac{2k}{n_z} - 1, \quad k = 0, 1, \ldots, n_z$$

$$z_{\text{face}}(k) = \frac{L_z}{2} \left[1 + \frac{\tanh(\gamma\xi)}{\tanh(\gamma)}\right]$$

where $\gamma$ is the stretching parameter.

### Properties

- $\gamma = 0$: Uniform grid
- $\gamma > 0$: Clustering near walls ($z = 0$ and $z = L_z$)
- Typical range: $\gamma = 2.0\text{--}3.5$

### Grid Spacing

Face locations: $z_f = \{z_0, z_1, \ldots, z_{n_z}\}$

Cell centers:
$$z_c(k) = \frac{z_f(k) + z_f(k+1)}{2}$$

Cell spacing:
- $\Delta z_f(k) = z_f(k+1) - z_f(k)$ (distance between faces)
- $\Delta z_c(k) = z_c(k) - z_c(k-1)$ (distance between centers)

### Wall Units

For turbulent channel flow, grid spacing is checked in wall units:

$$\Delta x^+ = \frac{\Delta x \cdot u_\tau}{\nu}, \quad \Delta y^+ = \frac{\Delta y \cdot u_\tau}{\nu}, \quad \Delta z^+ = \frac{\Delta z \cdot u_\tau}{\nu}$$

Requirements:
- $\Delta x^+ < 15$ (streamwise)
- $\Delta y^+ < 10$ (spanwise)
- $\Delta z^+ < 0.5$ near wall (wall-normal)

---

## Time Integration

Three time integration schemes are available:

### 1. IMEX (Recommended)

**Implicit-Explicit** scheme combining stability and efficiency.

**Explicit terms** (Adams-Bashforth 2):
- Advection: $(\mathbf{u} \cdot \nabla)u$
- Horizontal diffusion: $\nu\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)$

**Implicit terms** (Crank-Nicolson):
- Vertical diffusion: $\nu\frac{\partial^2 u}{\partial z^2}$

**Algorithm:**

Explicit prediction:
$$\mathbf{u}^* = \mathbf{u}^n + \Delta t \left[\frac{3}{2} \text{RHS}^n - \frac{1}{2} \text{RHS}^{n-1}\right]$$
where RHS includes advection + xy-diffusion.

Implicit correction:
$$\left[I - \frac{\Delta t}{2} \nu \frac{\partial^2}{\partial z^2}\right] \mathbf{u}^{**} = \mathbf{u}^* + \frac{\Delta t}{2} \nu \frac{\partial^2 \mathbf{u}^n}{\partial z^2}$$

Projection:
$$\mathbf{u}^{n+1} = \mathbf{u}^{**} - \Delta t \nabla \varphi$$
where $\nabla^2 \varphi = \nabla \cdot \mathbf{u}^{**} / \Delta t$.

**Advantages:**
- No timestep restriction from z-diffusion
- Stable for stretched grids with fine wall spacing
- Good accuracy (2nd-order in time)

**Stability:**
- CFL condition for advection and xy-diffusion
- Unconditionally stable for z-diffusion

### 2. Adams-Bashforth 2 (AB2)

Fully explicit 2nd-order scheme.

**Algorithm:**

Explicit step:
$$\mathbf{u}^* = \mathbf{u}^n + \Delta t \left[\frac{3}{2} \text{RHS}^n - \frac{1}{2} \text{RHS}^{n-1}\right]$$
where RHS includes all terms (advection + diffusion).

Projection:
$$\mathbf{u}^{n+1} = \mathbf{u}^* - \Delta t \nabla \varphi$$

**Advantages:**
- Simple implementation
- Fast per-step computation

**Disadvantages:**
- Stricter timestep restriction (includes z-diffusion)
- Requires smaller Δt for stretched grids

**Stability:**
- CFL condition for advection
- Diffusion stability: $\Delta t < 0.5 \frac{\Delta z_{\text{min}}^2}{\nu}$
- For stretched grids: can be very restrictive

### 3. Forward Euler (FE)

Simple 1st-order explicit scheme (for testing only).

**Algorithm:**

Explicit step:
$$\mathbf{u}^* = \mathbf{u}^n + \Delta t \, \text{RHS}^n$$

Projection:
$$\mathbf{u}^{n+1} = \mathbf{u}^* - \Delta t \nabla \varphi$$

**Disadvantages:**
- Low accuracy (1st-order)
- Strict stability limits
- Not recommended for production runs

**Use case:** Basic testing and code validation only.

### Adaptive Timestepping

Timestep $\Delta t$ is adjusted to maintain target CFL:

$$\text{CFL} = \Delta t \times \max\left(\frac{|u|}{\Delta x} + \frac{|v|}{\Delta y} + \frac{|w|}{\Delta z}\right)$$

**Update rule:**

$$\Delta t_{\text{new}} = \Delta t_{\text{old}} \times \frac{\text{CFL}_{\text{target}}}{\text{CFL}_{\text{current}}}$$

Constrained by: $\Delta t_{\text{min}} \leq \Delta t_{\text{new}} \leq \Delta t_{\text{max}}$

**Update frequency:** Every `dt_update_interval` steps

---

## Pressure-Velocity Coupling

### Fractional Step Method

The projection method decouples pressure from velocity:

1. **Momentum step**: Predict velocity u* without pressure
2. **Projection step**: Enforce divergence-free condition

### Projection Algorithm

From predicted velocity $\mathbf{u}^*$, find corrected velocity $\mathbf{u}^{n+1}$:

$$\mathbf{u}^{n+1} = \mathbf{u}^* - \Delta t \nabla \varphi$$

where $\varphi$ is the pressure correction (pseudo-pressure).

Enforce $\nabla \cdot \mathbf{u}^{n+1} = 0$:

$$\nabla \cdot \mathbf{u}^{n+1} = \nabla \cdot \mathbf{u}^* - \Delta t \nabla^2 \varphi = 0$$

This gives the **Poisson equation**:

$$\nabla^2 \varphi = \frac{\nabla \cdot \mathbf{u}^*}{\Delta t}$$

### Boundary Conditions for $\varphi$

**Neumann BC at walls (z = 0, Lz):**
$$\frac{\partial \varphi}{\partial z} = 0$$

**Periodic in x, y** (inherits from velocity).

### FFT-Based Poisson Solver

For periodic directions (x, y) and Neumann BC in z:

1. **FFT in x, y:**
   $$\hat{\varphi}(k_x, k_y, z) = \text{FFT}_{xy}[\varphi(x, y, z)]$$

2. **Tridiagonal system in z** (for each wavenumber pair):
   $$\left[\frac{\partial^2}{\partial z^2} - k_x^2 - k_y^2\right] \hat{\varphi} = \widehat{\text{RHS}}$$

   Using modified wavenumbers:
   $$k_{x,\text{mod}} = \frac{2}{\Delta x} \sin\left(\frac{k_x \Delta x}{2}\right)$$
   $$k_{y,\text{mod}} = \frac{2}{\Delta y} \sin\left(\frac{k_y \Delta y}{2}\right)$$

3. **Solve tridiagonal system** for each $(k_x, k_y)$ mode

4. **Inverse FFT:**
   $$\varphi(x, y, z) = \text{FFT}_{xy}^{-1}[\hat{\varphi}(k_x, k_y, z)]$$

**Advantages:**
- Fast: O(N log N) complexity
- Accurate: 2nd-order with modified wavenumbers
- Scales well on GPU (batched tridiagonal solves)

### Direct Poisson Solver (Fallback)

For validation purposes, a direct sparse matrix solver is available:

1. Discretize ∇²φ = RHS on the full 3D grid
2. Form sparse matrix A
3. Solve Au = b using sparse linear algebra

**Disadvantages:**
- Slower: O(N^1.5) to O(N²) complexity
- Limited to smaller grids
- Not recommended for production

---

## Bulk Velocity Forcing

To maintain constant bulk (mean) velocity $U_{\text{bulk}}$, a forcing term is applied:

$$f_x = \frac{U_{\text{bulk}} - u_{\text{bulk}}}{\Delta t}$$

where:
$$u_{\text{bulk}} = \frac{1}{V} \iiint u \, dV$$

### Implementation

At each timestep:

1. Compute current bulk velocity:
   $$u_{\text{bulk}} = \frac{\sum(u \times \text{cell\_volume})}{\text{total\_volume}}$$

2. Compute forcing:
   $$f_x = \frac{U_{\text{bulk,target}} - u_{\text{bulk}}}{\Delta t}$$

3. Add forcing to momentum equation:
   $$\frac{\partial u}{\partial t} = \ldots + f_x$$

### Physical Interpretation

- Mimics constant pressure gradient driving the flow
- Maintains constant mass flux through channel
- Friction velocity $u_\tau$ adjusts to balance forcing

### Relationship to $u_\tau$

In steady state:
- Forcing balances wall shear: $f_x \approx u_\tau^2 / \delta$
- $u_\tau = \sqrt{f_x \times \delta}$

---

## Turbulence Statistics

### Mean Quantities

Time-averaged mean velocity:

$$U(z) = \langle u(x, y, z, t) \rangle_{x,y,t}$$

Computed as:
$$U(z) = \frac{1}{N} \sum_{n} \left[\text{mean over } x,y \text{ of } u(x,y,z,t_n)\right]$$

### Reynolds Stresses

Fluctuating velocity: $u' = u - U$

Reynolds stresses:
- $\langle u'u' \rangle(z)$: Streamwise variance
- $\langle v'v' \rangle(z)$: Spanwise variance
- $\langle w'w' \rangle(z)$: Wall-normal variance
- $\langle u'w' \rangle(z)$: Reynolds shear stress

### Friction Velocity

Friction velocity $u_\tau$ is computed from:

**Method 1: Wall shear stress**
$$\tau_{\text{wall}} = \nu \left.\frac{dU}{dz}\right|_{\text{wall}} = \rho u_\tau^2$$

**Method 2: Reynolds stress near wall**
$$-\langle u'w' \rangle|_{\text{wall}} \approx u_\tau^2$$

TorChannel uses **Method 1** (velocity gradient at wall) for accuracy.

### 2D Energy Spectra

At height $z^+ \approx 15$ (buffer layer):

$$E_{uu}(k_x, k_y) = \frac{|\hat{u}(k_x, k_y)|^2}{(n_x n_y)^2}$$

where $\hat{u}$ is the 2D FFT of $u'(x, y)$.

**Premultiplied spectra** (for visualization):
$$\Phi_{uu}(k_x, k_y) = k_x \times k_y \times E_{uu}(k_x, k_y)$$

Highlights energy-containing scales.

### Total Stress Balance

In channel flow, total stress varies linearly:

$$\tau_{\text{total}}(z) = \tau_{\text{wall}} \times \left(1 - \frac{z}{\delta}\right)$$

Decomposition:
$$\tau_{\text{total}} = \underbrace{-\langle u'w' \rangle}_{\text{Reynolds}} + \underbrace{\nu\frac{dU}{dz}}_{\text{Viscous}}$$

Near wall: viscous dominates
Far from wall: Reynolds stress dominates

---

## References

### Key Papers

1. **Kim, J., Moin, P., & Moser, R. (1987)**
   "Turbulence statistics in fully developed channel flow at low Reynolds number"
   *Journal of Fluid Mechanics*, 177, 133-166.
   - Seminal DNS study at Re_τ ≈ 180
   - Established benchmark for turbulent channel flow
   - Detailed statistics and validation

2. **Moser, R. D., Kim, J., & Mansour, N. N. (1999)**
   "Direct numerical simulation of turbulent channel flow up to Re_τ = 590"
   *Physics of Fluids*, 11(4), 943-945.
   - Extended DNS to higher Reynolds numbers
   - Comprehensive database: http://turbulence.pha.jhu.edu/

3. **Verzicco, R., & Orlandi, P. (1996)**
   "A finite-difference scheme for three-dimensional incompressible flows in cylindrical coordinates"
   *Journal of Computational Physics*, 123(2), 402-414.
   - Finite differences on non-uniform grids
   - Projection method implementation

### Textbooks

4. **Pope, S. B. (2000)**
   *Turbulent Flows*
   Cambridge University Press.
   - Comprehensive turbulence theory
   - Wall turbulence and DNS methodology

5. **Peyret, R. (2002)**
   *Spectral Methods for Incompressible Viscous Flow*
   Springer.
   - Spectral and spectral-element methods
   - FFT-based solvers

6. **Ferziger, J. H., & Perić, M. (2002)**
   *Computational Methods for Fluid Dynamics*
   Springer.
   - Finite difference and finite volume methods
   - Staggered grids and projection methods

### DNS Databases

- **Johns Hopkins Turbulence Database**: http://turbulence.pha.jhu.edu/
- **AGARD Test Cases**: Historical validation cases

---

## Validation Criteria

TorChannel results should satisfy:

1. **Mean velocity profile:**
   - Viscous sublayer: $U^+ = z^+$ for $z^+ < 5$
   - Log layer: $U^+ = \frac{1}{\kappa} \ln(z^+) + B$ for $z^+ > 30$
   - $\kappa \approx 0.41$, $B \approx 5.2$

2. **Friction velocity:**
   - Converges to target: $\frac{|u_\tau - u_{\tau,\text{target}}|}{u_{\tau,\text{target}}} < 1\%$

3. **Reynolds stresses:**
   - Peak locations and magnitudes match literature
   - $\langle u'u' \rangle$: peak $\approx 3.0 \, u_\tau^2$ near wall
   - $-\langle u'w' \rangle$: linear decrease from wall

4. **Total stress balance:**
   - $$\tau_{\text{total}} = -\langle u'w' \rangle + \nu \frac{dU}{dz}$$
   - matches linear profile

5. **Energy spectra:**
   - Show $-5/3$ slope in inertial range (at higher Re)
   - Peak wavelengths match expected structure sizes

---

For implementation details and code architecture, see [Implementation Documentation](IMPLEMENTATION.md).
