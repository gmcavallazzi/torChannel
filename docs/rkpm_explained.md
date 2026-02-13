### Technical Deep Dive: Why RKPM?

The core innovation in Pinelli et al. (2010) is replacing the standard fixed "cosine" delta functions (common in Peskin’s IBM) with a **Reproducing Kernel Particle Method (RKPM)** approach.

**1. The Problem with Standard Delta Functions**
Standard IBM uses a fixed discrete delta function, e.g., $\delta_h(r) = \frac{1}{4}(1 + \cos(\frac{\pi r}{2}))$.
* **On Uniform Grids:** This works because the sum of weights naturally equals 1 ($\sum \delta = 1$) and first moments cancel out ($\sum \delta \cdot x = x_{interface}$).
* **On Stretched/Curvilinear Grids:** These symmetries break. The sum of weights may not equal 1, and the interpolation acts like a "leaky" filter, reducing the order of accuracy of your FD scheme.

**2. The RKPM Solution (Moment Conditions)**
Pinelli’s method dynamically calculates weights $w_{lk}$ connecting a Lagrangian marker $\mathbf{X}_l$ to an Eulerian fluid node $\mathbf{x}_k$ such that they **exactly reproduce** polynomials up to a certain degree $m$.

To achieve this, the weight is not just a base kernel function $\phi$; it is a base kernel corrected by a polynomial function $P$:
$$\tilde{\delta}(\mathbf{x}_k - \mathbf{X}_l) = \underbrace{\mathbf{p}^T(\mathbf{x}_k - \mathbf{X}_l) \mathbf{b}(\mathbf{X}_l)}_{\text{Correction}} \cdot \underbrace{\phi(\mathbf{x}_k - \mathbf{X}_l)}_{\text{Base Window}}$$

* $\mathbf{p} = [1, x, y, z, \dots]^T$ is the basis vector.
* $\mathbf{b}$ is a coefficient vector solved locally to ensure the moments are satisfied.
* $\phi$ is usually a simple spline or top-hat function to ensure compact support (locality).

**3. Conservation via Adjointness**
For the method to remain stable (energy conserving), the **Spreading** operator (Force $\to$ Fluid) must be the adjoint (transpose) of the **Interpolation** operator (Fluid $\to$ Velocity), scaled by the grid cell volumes. This guarantees that the work done by the fluid on the interface equals the work done by the interface force on the fluid.

---

### Algorithm Document: RKPM Finite Difference IBM

**Scope:** Incompressible Navier-Stokes on a Collocated or Staggered Finite Difference Grid.
**Objective:** Enforce velocity boundary condition $\mathbf{U}_l^{wall}$ on interface $\Gamma$.

#### Part 1: Pre-Computation (Weight Generation)
*Execute this once for static boundaries. For moving boundaries, execute every time the boundary moves more than half a cell width.*

**Step 1.1: Identify Support**
For each Lagrangian marker $\mathbf{X}_l$:
1.  Identify the index of the nearest grid node $(i,j,k)$.
2.  Select a support block of nodes $\Omega_l$ around it.
    * *Recommendation:* A $3 \times 3$ (2D) or $3 \times 3 \times 3$ (3D) stencil is usually sufficient for linear reproducibility.
    * Let $N_{sup}$ be the number of nodes in this support.

**Step 1.2: Construct the Moment Matrix**
We seek weights $w_{lk}$ such that:
$$\sum_{k \in \Omega_l} w_{lk} (\mathbf{x}_k - \mathbf{X}_l)^\alpha = \delta_{\alpha,0}$$
For 2D Linear Reproducibility (Basis: $1, x, y$), solving for coefficients $\mathbf{b} = [b_0, b_1, b_2]^T$:
Construct the matrix $\mathbf{M}$ and vector $\mathbf{m}$:
$$\mathbf{M} = \sum_{k \in \Omega_l} \phi_k \cdot \begin{bmatrix} 1 & \Delta x_k & \Delta y_k \\ \Delta x_k & \Delta x_k^2 & \Delta x_k \Delta y_k \\ \Delta y_k & \Delta x_k \Delta y_k & \Delta y_k^2 \end{bmatrix}, \quad \mathbf{m} = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}$$
where $\Delta x_k = x_k - X_l$ and $\phi_k$ is a standard window function (e.g., cubic spline) based on distance.

**Step 1.3: Solve for Coefficients**
Solve the linear system (size $3\times3$ or $4\times4$) for $\mathbf{b}$:
$$\mathbf{M} \mathbf{b} = \mathbf{m}$$

**Step 1.4: Compute Final Weights**
Calculate and store the weights for each node $k$ in the support:
$$w_{lk} = \left( b_0 + b_1 \Delta x_k + b_2 \Delta y_k \right) \cdot \phi(\mathbf{x}_k - \mathbf{X}_l)$$
*Check:* Ensure $\sum w_{lk} = 1$ (machine precision).

---

#### Part 2: Time Integration (Fractional Step)
*Assuming a standard projection method (Predictor-Corrector).*

**Step 2.1: Velocity Prediction (No Forcing)**
Advance momentum equation ignoring the body force to get intermediate velocity $\mathbf{u}^*$:
$$\frac{\mathbf{u}^* - \mathbf{u}^n}{\Delta t} = -(\mathbf{u} \cdot \nabla \mathbf{u})^n + \nu \nabla^2 \mathbf{u}^n - \nabla p^n$$

**Step 2.2: Interpolation (Eulerian $\to$ Lagrangian)**
Interpolate the prediction $\mathbf{u}^*$ to the marker locations $\mathbf{X}_l$:
$$\mathbf{U}^*_l = \sum_{k \in \Omega_l} w_{lk} \mathbf{u}^*_k$$

**Step 2.3: Force Calculation (Lagrangian)**
Compute the restoring force needed to impose the BC $\mathbf{U}_l^{wall}$:
$$\mathbf{F}_l = \frac{\mathbf{U}^{wall}_l - \mathbf{U}^*_l}{\Delta t}$$
*Note: $\mathbf{F}_l$ is a force density per unit volume (or area in 2D).*

**Step 2.4: Spreading (Lagrangian $\to$ Eulerian)**
Distribute the force back to the grid. **Crucial:** Use the volume ratio scaling.
$$\mathbf{f}_k = \sum_{l \in Markers} \mathbf{F}_l \cdot w_{lk} \cdot \frac{\Delta V_l}{\Delta V_k}$$
* $\Delta V_k$: Volume of grid cell $k$ (finite difference cell size $dx dy dz$).
* $\Delta V_l$: "Volume" associated with marker $l$ (usually $ds^2$ or surface area segment).

**Step 2.5: Velocity Correction**
Update the predictor velocity with the IBM force:
$$\mathbf{u}^{**} = \mathbf{u}^* + \Delta t \mathbf{f}$$

**Step 2.6: Pressure Projection**
Standard pressure correction step to enforce continuity $\nabla \cdot \mathbf{u} = 0$:
1.  Solve $\nabla^2 \Phi = \frac{\nabla \cdot \mathbf{u}^{**}}{\Delta t}$
2.  $\mathbf{u}^{n+1} = \mathbf{u}^{**} - \Delta t \nabla \Phi$
3.  $p^{n+1} = p^n + \Phi$

---

### Implementation Cheat Sheet

| Component | Standard IBM | Pinelli (RKPM) IBM |
| :--- | :--- | :--- |
| **Grid Requirement** | Uniform Cartesian | **Generalized / Stretched** |
| **Weight Formula** | Explicit Cosine $\delta(r)$ | **Implicit Linear System** |
| **Cost** | Cheap (Direct calculation) | **Moderate (Small matrix solve per point)** |
| **Support Size** | Fixed (e.g. 4 points) | **Adaptive (based on condition)** |
| **Accuracy** | 1st order near boundary | **2nd order (if moments satisfied)** |

Based on the detailed formulation in Pinelli et al. (2010), particularly Sections 2.2 and 2.3, here is a rigorous explanation of the algorithm for a **3D case with a non-uniform grid**.

The method distinguishes itself by not assuming a uniform grid topology. Instead of a fixed shape function (like the cosine delta), it calculates a **corrected shape function** locally for every Lagrangian marker. This correction forces the interpolation to respect the specific geometry of your stretched grid cells.

### 1. The Core Concept: Correcting the Window Function
On a non-uniform grid, a standard bell-shaped curve (window function) distorts the data it interpolates because the grid points are not symmetrically spaced.

Pinelli et al. use the **Reproducing Kernel Particle Method (RKPM)** to fix this.
* **Standard Kernel:** $w(\mathbf{r})$ (e.g., a spline).
* **Corrected Kernel:** $\tilde{w}(\mathbf{r}, \mathbf{X}_l)$.
    The corrected kernel is the standard kernel multiplied by a **polynomial correction function** $P(\mathbf{r})$:
    $$\tilde{w}(\mathbf{x} - \mathbf{X}_l) = P(\mathbf{x} - \mathbf{X}_l) \cdot w(\mathbf{x} - \mathbf{X}_l)$$
    
[cite_start]The coefficients of this polynomial are calculated uniquely for every marker $\mathbf{X}_l$ so that the kernel **exactly reproduces** polynomials up to a specific order (usually quadratic) on the local stretched grid[cite: 7, 8].

---

### 2. Step-by-Step Algorithm for 3D Stretched Grids

#### Step 1: Defining the "Support Cage" ($\Omega_I$)
Unlike uniform grids where you simply take "3 points left, 3 points right," the support on a stretched grid is defined by physical dimensions ensuring enough nodes are included for stability.

1.  **Locate the Center:** For a marker at $\mathbf{X}_I = (X_I, Y_I, Z_I)$, find the nearest grid node $(x_i, y_j, z_k)$.
2.  **Determine Dilation Parameters ($\delta, \eta, \sigma$):** You need to define the physical size of the kernel support. Pinelli defines local dilation parameters based on the local grid spacing $h$.
    * For the non-uniform direction $z$, calculate the local spacing $h_z^+$ and $h_z^-$ (distances to neighbors).
    * Set the dilation parameter $\sigma_I$ (for Z-direction) roughly as:
        $$\sigma_I \approx h_z(Z_I)$$
    * [cite_start]The support size is typically $3\sigma_I$ in the $z$ direction[cite: 358].
3.  **The Cage:** The support $\Omega_I$ is the rectangular volume centered at $\mathbf{X}_I$ with dimensions $3\delta \times 3\eta \times 3\sigma$.
    * [cite_start]**Crucial Rule:** The cage must contain at least **27 nodes** (3x3x3) in 3D to solve for the quadratic moments without singularity[cite: 366].

#### Step 2: The Moment Matrix Assembly
This is the "brain" of the method. You are solving for the polynomial coefficients that "fix" the kernel distortion caused by the stretched grid.

**The Polynomial Basis (3D Quadratic):**
We use a basis vector $\mathbf{p}$ with 10 terms (for 2nd order accuracy):
$$\mathbf{p}^T = [1, x, y, z, xy, yz, zx, x^2, y^2, z^2]$$
*(Note: Coordinates here are relative to the marker, i.e., $x = x_{node} - X_I$)*.

**The System to Solve:**
For each marker $I$, you solve a $10 \times 10$ symmetric linear system for the coefficient vector $\mathbf{b}$:
$$\mathbf{M}^I \mathbf{b}^I = \mathbf{e}_1$$
where $\mathbf{e}_1 = [1, 0, 0, \dots, 0]^T$. [cite_start]This forces the zeroth moment to be 1 (conservation of mass/force) and higher moments to be 0 (canceling interpolation errors)[cite: 327].

**Computing Matrix $\mathbf{M}$ (Discrete Integral):**
The element $m_{\alpha \beta}$ of matrix $\mathbf{M}$ is computed by summing over the grid nodes in the support cage $\Omega_I$:
$$m_{\alpha \beta} = \sum_{k \in \Omega_I} (\mathbf{p}_\alpha \mathbf{p}_\beta) \cdot w(\mathbf{x}_k - \mathbf{X}_I) \cdot \Delta V_k$$
* **Non-Uniform Grid Detail:** $\Delta V_k$ is the **volume of the grid cell** at node $k$. In your stretched $z$ grid, $\Delta V_k = \Delta x \Delta y \Delta z_k$. [cite_start]You *must* include this local volume term in the sum for the method to handle the stretching correctly[cite: 330, 369].
* **Ill-Conditioning:** Because $\Delta x^2$ is much smaller than $1$, this matrix is ill-conditioned. [cite_start]Pinelli prescribes using a scaling matrix $\mathbf{H}$ to normalize distances by the dilation parameters ($\delta, \eta, \sigma$) before solving[cite: 376].

#### Step 3: Interpolation (Fluid $\to$ Marker)
Once you have the vector $\mathbf{b}^I$ for marker $I$, you can construct the corrected weight for any neighbor node $k$:
$$\tilde{w}_{Ik} = w(\mathbf{r}_{Ik}) \cdot (\mathbf{b}^I \cdot \mathbf{p}(\mathbf{r}_{Ik}))$$
The velocity at the marker is then the standard summation:
$$\mathbf{U}(X_I) = \sum_{k \in \Omega_I} \mathbf{u}_k \cdot \tilde{w}_{Ik} \cdot \Delta V_k$$

#### Step 4: Spreading (Marker $\to$ Fluid) & The Partition of Unity
This is the most technically distinct part of the Pinelli algorithm.
Most methods just use the transpose of the interpolation weights. [cite_start]Pinelli notes that to strictly conserve moments on arbitrary grids, the spreading weights must satisfy a **Partition of Unity** condition on the Lagrangian (marker) set[cite: 422].

**The Problem:** If markers are clustered or sparse relative to the grid, simple spreading overlaps too much or too little force.

**The Solution (Global/Local System for $\epsilon$):**
The force spreading equation is:
$$\mathbf{f}(x_k) = \sum_{I} \mathbf{F}(X_I) \cdot \tilde{w}_{Ik} \cdot \epsilon_I \Delta s_I$$
* $\Delta s_I$: Area/Volume associated with the marker.
* $\epsilon_I$: A computed scaling factor.

[cite_start]To find $\epsilon_I$, you must solve a system (Equation 47 in the paper) that ensures that if you spread a "unit" field from the markers, the fluid grid sees exactly "1" everywhere[cite: 419]:
$$\mathbf{A} \vec{\epsilon} = \vec{1}$$
Where $\mathbf{A}$ is a matrix representing the interaction between markers.
* [cite_start]**Simplification:** Pinelli notes that if your Lagrangian marker spacing $\Delta s$ is close to the local Eulerian grid spacing $\Delta x$ (ratio $\approx 1$), then $\vec{\epsilon}$ is smooth and close to 1. In this case, you can often approximate $\epsilon \approx 1$ or solve this system iteratively in very few steps[cite: 445].

---

### 3. Summary of Technical Implementation Details
1.  **Pre-computation:**
    * For every marker, identify neighbors in the $3\sigma$ cage.
    * Compute distances using **stretched** $z$-coordinates.
    * Assemble $10 \times 10$ Moment Matrix $\mathbf{M}$ using local cell volumes $\Delta V_k$.
    * Scale $\mathbf{M}$ using $\mathbf{H}$ to fix condition number.
    * Solve $\mathbf{M}\mathbf{b} = [1,0,\dots]^T$ to get polynomial coefficients $\mathbf{b}$.
    * Store the final corrected weights $\tilde{w}_{Ik}$.

2.  **Time Step Loop:**
    * **Interpolate:** $\mathbf{U}_{IB} = \sum \mathbf{u}_{grid} \tilde{w} \Delta V$.
    * **Force:** $\mathbf{F}_{IB} = (\mathbf{U}_{wall} - \mathbf{U}_{IB}) / \Delta t$.
    * **Spread:** $\mathbf{f}_{grid} = \sum \mathbf{F}_{IB} \tilde{w} \epsilon \Delta s$. .
    * **Project:** Solve pressure Poisson and update velocity.

### Why this works for your Stretched Grid
Standard methods assume the error terms cancel out due to symmetry (e.g., $x$ on the left cancels $-x$ on the right). On a stretched grid, $x_{left} \neq -x_{right}$.
The RKPM method explicitly calculates the error term $\sum w \cdot x$ and finds a coefficient $b_1$ to subtract it out exactly. [cite_start]This guarantees that your boundary layer profile (which is linear/quadratic near the wall) is preserved even if the grid is highly distorted[cite: 60].