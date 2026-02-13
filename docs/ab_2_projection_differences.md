# Differences Between Standard and Incremental Projection Methods (Including Forcing Term)

## Forcing Term Often Used in Practice
A commonly used feedback-type forcing for flow‑rate or bulk‑velocity control is:
\[
\mathbf{f} = \frac{\mathbf{u}_{\text{target}} - \mathbf{u}_{\text{bulk}}}{\Delta t}.
\]
This term is added *explicitly* in the predictor equation of both the standard and incremental projection schemes.

In an AB2 predictor, this becomes:
\[
\frac{u^* - u^n}{\Delta t} + \left(1.5\,N(u^n) - 0.5\,N(u^{n-1})\right)
= -\nabla p^n + \nu\nabla^2 u^* + f^{n+1},
\]
with
\[
 f^{n+1} = \frac{u_{\text{target}}^{n+1} - u_{\text{bulk}}^{n}}{\Delta t}.
\]
The projection step automatically removes any divergence introduced by the forcing:
\[
\nabla^2 \phi = \frac{1}{\Delta t}\nabla\cdot u^*,
\]
ensuring the corrected velocity is divergence‑free.

---

# Differences Between Standard and Incremental Projection Methods (Points 3 and 4)

This brief note summarizes the essential differences between the **standard projection** method and the **incremental (pressure-correction)** projection method when used with an Adams–Bashforth (AB2) explicit treatment of the nonlinear term.

---

## 1. Treatment of Pressure in the Predictor Step
**Standard Projection:**
- Predictor velocity \(u^*\) is computed *without* a pressure term:
  \
  \[
  \frac{u^* - u^n}{\Delta t} + \left(1.5\,N(u^n) - 0.5\,N(u^{n-1})\right) = \nu\nabla^2 u^* + f^{n+1}.
  \]

**Incremental Projection:**
- Predictor includes the old pressure gradient:
  \
  \[
  \frac{u^* - u^n}{\Delta t} + \left(1.5\,N(u^n) - 0.5\,N(u^{n-1})\right) = -\nabla p^n + \nu\nabla^2 u^* + f^{n+1}.
  \]

---

## 2. Nature of the Pressure Solve
**Standard Projection:**
- Solve Poisson for the *new pressure*:
  \
  \[
  \nabla^2 p^{n+1} = \frac{1}{\Delta t} \nabla\cdot u^*.
  \]
- Velocity correction:
  \
  \[
  u^{n+1} = u^* - \Delta t\,\nabla p^{n+1}.
  \]

**Incremental Projection:**
- Solve Poisson for the *pressure increment*:
  \
  \[
  \nabla^2 \phi = \frac{1}{\Delta t} \nabla\cdot u^*, \qquad \phi = p^{n+1} - p^n.
  \]
- Update velocity and pressure:
  \
  \[
  u^{n+1} = u^* - \Delta t\,\nabla \phi,
  \]
  \
  \[
  p^{n+1} = p^n + \phi.
  \]

---

## 3. Accuracy
**Standard Projection:**
- Pressure typically first order in time due to the lack of pressure in the predictor.
- Velocity sometimes reaches second order but the splitting error can degrade accuracy.

**Incremental Projection:**
- Second-order accuracy in both velocity and pressure.
- Reduced splitting error because pressure is included in prediction.

---

## 4. Boundary Conditions
**Standard Projection:**
- Boundary conditions apply directly to \(p^{n+1}\), e.g. Neumann for walls:
  \
  \[
  \frac{\partial p^{n+1}}{\partial n} = \frac{1}{\Delta t} n\cdot u^*.
  \]
- More sensitive to inconsistencies.

**Incremental Projection:**
- Boundary conditions applied to the increment \(\phi\):
  \
  \[
  \frac{\partial \phi}{\partial n} = \frac{1}{\Delta t} n\cdot u^*.
  \]
- Ensures corrected velocity satisfies the imposed physical boundary conditions more cleanly.

---

## 5. Practical Recommendation
- The **incremental projection** method is preferred for:
  - Better pressure accuracy,
  - More stable boundary-condition treatment,
  - Lower splitting error.

The **standard projection** method is simpler but less accurate, especially for unsteady flows.

