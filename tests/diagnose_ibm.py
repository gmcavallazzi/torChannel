"""
Diagnostic script to check IBM force magnitudes and stability issues.
"""
import torch
import yaml
import numpy as np
from solver import ChannelFlow

torch.set_default_dtype(torch.float64)

def diagnose_ibm(config_file='config.yaml'):
    """Run diagnostic checks on IBM implementation."""

    # Load solver
    solver = ChannelFlow(config_file=config_file)

    if not solver.ibm_enabled:
        print("ERROR: IBM is not enabled in config!")
        return

    print("\n" + "="*80)
    print("IBM DIAGNOSTICS")
    print("="*80)

    # 1. Check epsilon values (partition of unity regularization)
    print("\n1. EPSILON VALUES (Partition of Unity):")
    print(f"   epsilon_u: min={solver.ibm.epsilon_u.min():.6f}, max={solver.ibm.epsilon_u.max():.6f}, mean={solver.ibm.epsilon_u.mean():.6f}")
    print(f"   epsilon_v: min={solver.ibm.epsilon_v.min():.6f}, max={solver.ibm.epsilon_v.max():.6f}, mean={solver.ibm.epsilon_v.mean():.6f}")
    print(f"   epsilon_w: min={solver.ibm.epsilon_w.min():.6f}, max={solver.ibm.epsilon_w.max():.6f}, mean={solver.ibm.epsilon_w.mean():.6f}")

    # Check for anomalous epsilon values
    if (solver.ibm.epsilon_u.max() > 100 or solver.ibm.epsilon_v.max() > 100 or
        solver.ibm.epsilon_w.max() > 100):
        print("   WARNING: Epsilon values > 100 detected! This suggests poor conditioning.")

    if (solver.ibm.epsilon_u.min() < 0 or solver.ibm.epsilon_v.min() < 0 or
        solver.ibm.epsilon_w.min() < 0):
        print("   ERROR: Negative epsilon values! Partition of unity is violated.")

    # 2. Check Lagrangian point distribution
    print("\n2. LAGRANGIAN POINTS:")
    print(f"   Number of points: {solver.ibm.n_lag}")
    print(f"   x range: [{solver.ibm.x_lag_t.min():.4f}, {solver.ibm.x_lag_t.max():.4f}]")
    print(f"   y range: [{solver.ibm.y_lag_t.min():.4f}, {solver.ibm.y_lag_t.max():.4f}]")
    print(f"   z range: [{solver.ibm.z_lag_t.min():.4f}, {solver.ibm.z_lag_t.max():.4f}]")
    print(f"   dS: min={solver.ibm.dS_t.min():.6e}, max={solver.ibm.dS_t.max():.6e}")

    # 3. Perform one predictor step and check forces
    print("\n3. FORCE MAGNITUDE AT FIRST STEP:")
    solver.apply_bc_uvw()

    # Compute predictor
    u_pred = solver.u.clone()
    v_pred = solver.v.clone()
    w_pred = solver.w.clone()

    # Interpolate to Lagrangian points
    u_lag = solver.ibm.interpolate(u_pred[:, 1:-1, 1:-1], 'u')
    v_lag = solver.ibm.interpolate(v_pred[1:-1, :, 1:-1], 'v')
    w_lag = solver.ibm.interpolate(w_pred[1:-1, 1:-1, :], 'w')

    print(f"   u_lag: min={u_lag.min():.6f}, max={u_lag.max():.6f}, mean={u_lag.mean():.6f}")
    print(f"   v_lag: min={v_lag.min():.6f}, max={v_lag.max():.6f}, mean={v_lag.mean():.6f}")
    print(f"   w_lag: min={w_lag.min():.6f}, max={w_lag.max():.6f}, mean={w_lag.mean():.6f}")

    # Compute forces (WITHOUT relaxation)
    dt = solver.dt
    f_u_lag = (0.0 - u_lag) / dt
    f_v_lag = (0.0 - v_lag) / dt
    f_w_lag = (0.0 - w_lag) / dt

    print(f"\n   Raw IBM forces (no relaxation):")
    print(f"   f_u_lag: min={f_u_lag.min():.3e}, max={f_u_lag.max():.3e}, rms={torch.sqrt(torch.mean(f_u_lag**2)):.3e}")
    print(f"   f_v_lag: min={f_v_lag.min():.3e}, max={f_v_lag.max():.3e}, rms={torch.sqrt(torch.mean(f_v_lag**2)):.3e}")
    print(f"   f_w_lag: min={f_w_lag.min():.3e}, max={f_w_lag.max():.3e}, rms={torch.sqrt(torch.mean(f_w_lag**2)):.3e}")

    # Spread to Eulerian grid
    f_u_euler = solver.ibm.spread(f_u_lag, 'u')
    f_v_euler = solver.ibm.spread(f_v_lag, 'v')
    f_w_euler = solver.ibm.spread(f_w_lag, 'w')

    print(f"\n   Spread IBM forces (Eulerian):")
    print(f"   f_u_euler: min={f_u_euler.min():.3e}, max={f_u_euler.max():.3e}, rms={torch.sqrt(torch.mean(f_u_euler**2)):.3e}")
    print(f"   f_v_euler: min={f_v_euler.min():.3e}, max={f_v_euler.max():.3e}, rms={torch.sqrt(torch.mean(f_v_euler**2)):.3e}")
    print(f"   f_w_euler: min={f_w_euler.min():.3e}, max={f_w_euler.max():.3e}, rms={torch.sqrt(torch.mean(f_w_euler**2)):.3e}")

    # 4. Estimate maximum stable time step
    print("\n4. STABILITY ESTIMATE:")
    max_force = max(f_u_euler.abs().max(), f_v_euler.abs().max(), f_w_euler.abs().max())
    print(f"   Maximum force magnitude: {max_force:.3e}")

    # For explicit Euler: dt * |f| < ~0.1 for stability
    # This is a rough estimate
    if max_force > 0:
        dt_stable_estimate = 0.1 / max_force.item()
        print(f"   Estimated stable dt (explicit): ~{dt_stable_estimate:.6f}")
        print(f"   Current dt: {dt:.6f}")
        if dt > dt_stable_estimate:
            print(f"   WARNING: Current dt may be too large! Ratio: {dt/dt_stable_estimate:.2f}x")
        else:
            print(f"   Current dt is {dt_stable_estimate/dt:.2f}x smaller than estimate (conservative)")

    # 5. Check velocity near obstacle
    print("\n5. VELOCITY NEAR OBSTACLE:")
    # Find grid points near cube center
    center = solver.ibm.center
    ix = int((center[0] - solver.dx/2) / solver.dx)
    iy = int((center[1] - solver.dy/2) / solver.dy)
    iz = int((center[2] - solver.z_c[1:-1].cpu().numpy()[0]) /
             (solver.z_c[1:-1].cpu().numpy()[1] - solver.z_c[1:-1].cpu().numpy()[0]))

    # Clamp to valid range
    ix = max(0, min(ix, solver.nx-1))
    iy = max(0, min(iy, solver.ny-1))
    iz = max(0, min(iz, solver.nz-1))

    print(f"   Near center (i={ix}, j={iy}, k={iz}):")
    print(f"   u = {solver.u[ix, iy, iz]:.6f}")
    print(f"   v = {solver.v[ix, iy, iz]:.6f}")
    print(f"   w = {solver.w[ix, iy, iz]:.6f}")

    # 6. Test Partition of Unity
    print("\n6. PARTITION OF UNITY TEST:")
    print("   Testing I(S(f)) = f property...")

    # Create a test field with constant value
    test_value = 10.0
    f_lag_test = torch.full((solver.ibm.n_lag,), test_value, device=solver.device, dtype=torch.float64)

    # Spread then interpolate
    f_u_spread = solver.ibm.spread(f_lag_test, 'u')
    f_v_spread = solver.ibm.spread(f_lag_test, 'v')
    f_w_spread = solver.ibm.spread(f_lag_test, 'w')

    # Interpolate back
    f_u_interp = solver.ibm.interpolate(f_u_spread, 'u')
    f_v_interp = solver.ibm.interpolate(f_v_spread, 'v')
    f_w_interp = solver.ibm.interpolate(f_w_spread, 'w')

    # Compute error
    error_u = (f_u_interp - f_lag_test).abs()
    error_v = (f_v_interp - f_lag_test).abs()
    error_w = (f_w_interp - f_lag_test).abs()

    print(f"   Input: f_lag = {test_value:.1f} (constant)")
    print(f"   After I(S(f)):")
    print(f"     u: min={f_u_interp.min():.6f}, max={f_u_interp.max():.6f}, mean_error={error_u.mean():.6e}")
    print(f"     v: min={f_v_interp.min():.6f}, max={f_v_interp.max():.6f}, mean_error={error_v.mean():.6e}")
    print(f"     w: min={f_w_interp.min():.6f}, max={f_w_interp.max():.6f}, mean_error={error_w.mean():.6e}")

    max_error = max(error_u.max().item(), error_v.max().item(), error_w.max().item())
    relative_error = max_error / test_value

    if relative_error < 0.01:
        print(f"   ✓ PASS: Partition of unity satisfied (error {relative_error:.2%})")
    elif relative_error < 0.1:
        print(f"   ⚠ MARGINAL: Partition of unity has {relative_error:.2%} error")
    else:
        print(f"   ✗ FAIL: Partition of unity violated! Error = {relative_error:.2%}")
        print(f"   This will cause incorrect IBM forces!")

    # 7. Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)

    # Based on force magnitude, suggest relaxation
    typical_velocity = 1.0  # U_bulk
    force_ratio = max_force.item() * dt / typical_velocity if max_force > 0 else 0

    if force_ratio > 0.5:
        alpha_suggested = 0.5 / force_ratio
        print(f"1. Add relaxation to IBM forcing: alpha ~ {alpha_suggested:.3f}")
        print(f"   This will reduce force by factor of {1.0/alpha_suggested:.2f}")
    else:
        print("1. Force magnitude seems reasonable")

    print(f"2. Consider force ramping over first ~10 time steps")
    print(f"3. Current dt={dt:.6f} - consider testing with smaller/larger values")

    if max(solver.ibm.epsilon_u.max(), solver.ibm.epsilon_v.max(), solver.ibm.epsilon_w.max()) > 10:
        print("4. High epsilon values suggest you may need more Lagrangian points")

    print("="*80 + "\n")

if __name__ == "__main__":
    diagnose_ibm('config.yaml')
