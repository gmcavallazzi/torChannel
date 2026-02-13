# Simulation Parameters from Monti et al. (2022)
## θ = 0° Case (Wall-Normally Mounted Canopy)

### Geometric Parameters

- **Filament length**: l = 0.25H
- **Canopy height**: h = l⊥ = 0.25H
- **Filament spacing**: ΔS = πH/24 ≈ 0.131H
- **Solidity**: λ = 0.35
- **Number of filaments**: 48 × 36 (streamwise × spanwise)
- **Filament diameter**: d ≈ 2.2Δx
- **Filament shape**: Cylindrical
- **Filament type**: Rigid

### Flow Parameters

- **Bulk Reynolds number**: Re_b = U_b H/ν = 6000
- **Friction Reynolds number**: Re_τ = u_τ H/ν = 1157.5
  - Based on total shear stress at canopy tip

### Computational Domain

- **Domain size**:
  - L_x/H = 2π
  - L_y/H = 1
  - L_z/H = 1.5π

- **Grid resolution**:
  - N_x = 576
  - N_y = 300
  - N_z = 432

- **Resolution in wall units** (at canopy tip):
  - Δx⁺ = 12.63
  - Δy_h⁺ = 0.35
  - Δz⁺ = 12.63

### Boundary Conditions

- **Streamwise (x)**: Periodic
- **Spanwise (z)**: Periodic
- **Bottom wall (y = 0)**: No-slip
- **Top surface (y = H)**: Free-slip (open channel)
- **Forcing**: Constant flow rate maintained by adjusted pressure gradient

### Numerical Method

- **Simulation type**: Large-Eddy Simulation (LES)
- **LES model**: Integral Length-Scale Approximation (ILSA)
- **Canopy representation**: Immersed Boundary Method (IBM)
- **Spatial discretization**: Second-order finite volume
- **Time advancement**: Semi-implicit fractional-step method
- **Temporal scheme**: Crank-Nicolson (implicit for wall-normal diffusion) + Adams-Bashforth (explicit for other terms)

### Reference

Monti, A., Nicholas, S., Omidyeganeh, M., Pinelli, A., & Rosti, M.E. (2022).
"On the solidity parameter in canopy flows."
*Journal of Fluid Mechanics* (Under consideration).
arXiv:2205.08050v2
