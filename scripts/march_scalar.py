"""Parabolized (space-marching) STEADY scalar driver -- fast developing-duct mixing.

The duct flow is steady and the scalar is passive, so we solve the velocity ONCE to
steady, FREEZE it, and then obtain the steady scalar field in a SINGLE downstream sweep
(scalar.march_scalar_steady) instead of time-marching it to steady state. Dropping the
streamwise-diffusion term (parabolic approximation, exact at high Pe) turns the steady
balance into an x-marching problem -- O(nx) cross-plane solves rather than O(Pe)
pseudo-time steps -- which is what makes high-Sc / high-N developing-duct sweeps tractable.

This is an ADDITIVE option: it reuses the same ChannelFlow setup, velocity solve, Koch
inlet, immersed mask and Mx(x) diagnostic as scripts/mixing_campaign.py and
scripts/run_frozen_scalar.py, and writes a `_final.npz` in the same layout so the
plot_campaign_* scripts work unchanged. Only the scalar solve is replaced by the marcher.

Validity: developing duct (bc_x='inout'), u>0 in the fluid, no-flux z walls. The
streamwise term is 2nd-order upwind (low false diffusion); transverse advection (for the
surface_baffle's secondary flow) is included by default (--no-cross to drop it for v=w=0).

Examples
--------
  # validate against the existing Sc=10 N=2 campaign result (should match Mx closely)
  python scripts/march_scalar.py --mode baffle --Sc 10 --N 2 \
         --compare results/campaign/baffle_Sc10_N2_final.npz

  # the open high-Sc, high-N test (resolved cross-section)
  python scripts/march_scalar.py --mode baffle --Sc 100 --N 4 --nx 256 --ny 256 --nz 256
"""
import os, sys, time, argparse, tempfile, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from solver import ChannelFlow
from scalar import scalar_stats
from mixing_campaign import base_config, Mx_profile


def _lmix(Mx, Lx, thr):
    """First x where Mx drops below thr (linear interp); nan if it never does."""
    x = np.linspace(0.0, Lx, len(Mx))
    below = np.where(Mx < thr)[0]
    if len(below) == 0 or below[0] == 0:
        return float('nan')
    i = below[0]
    x0, x1, m0, m1 = x[i - 1], x[i], Mx[i - 1], Mx[i]
    return float(x0 + (thr - m0) * (x1 - x0) / (m1 - m0)) if m1 != m0 else float(x1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='baffle', choices=['baffle', 'surface_baffle'])
    ap.add_argument('--Sc', type=float, default=10.0)
    ap.add_argument('--N', type=int, default=0)
    ap.add_argument('--nx', type=int, default=64)
    ap.add_argument('--ny', type=int, default=96)
    ap.add_argument('--nz', type=int, default=96)
    ap.add_argument('--dt_vel', type=float, default=5e-4, help="velocity solve dt (CFL-safe)")
    ap.add_argument('--vel_steps', type=int, default=40000, help="max velocity steps to steady")
    ap.add_argument('--vel_tol', type=float, default=1e-6, help="velocity steadiness (max|du|/500 steps)")
    ap.add_argument('--no-cross', dest='cross', action='store_false',
                    help="drop transverse advection (v=w=0 ducts, e.g. smooth baffle)")
    ap.add_argument('--n_inner', type=int, default=30, help="max Gauss-Seidel sweeps per plane")
    ap.add_argument('--tol', type=float, default=1e-9, help="per-plane GS convergence tol")
    ap.add_argument('--thr', type=float, default=0.70, help="M threshold for the reported L_mix")
    ap.add_argument('--compare', default=None, help="path to an existing *_final.npz to validate Mx against")
    ap.add_argument('--outdir', default='results/campaign')
    ap.add_argument('--tag', default=None)
    a = ap.parse_args()

    cfg = base_config(a.mode, a.Sc, a.dt_vel, nx=a.nx, ny=a.ny, nz=a.nz)
    cfg['scalar']['N'] = a.N
    if a.mode == 'surface_baffle':
        cfg['immersed']['N'] = a.N
    Lx = cfg['domain']['Lx']
    f = tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False); yaml.safe_dump(cfg, f); f.close()
    sim = ChannelFlow(f.name)
    nx, ny, nz = sim.nx, sim.ny, sim.nz
    chi = sim.chi_c if sim.immersed_enabled else None

    # ---- phase 1: solve velocity to steady, scalar disabled (Koch inlet kept aside) ----
    sim.scalar_enabled = False
    print(f"=== [{a.mode} Sc={a.Sc} N={a.N} grid={nx}x{ny}x{nz}] velocity -> steady ===", flush=True)
    t0 = time.time(); u_prev = sim.u.detach().clone()
    for step in range(1, a.vel_steps + 1):
        sim.step_imex(a.dt_vel)
        if step % 500 == 0:
            du = float((sim.u - u_prev).abs().max()); u_prev = sim.u.detach().clone()
            if step % 2000 == 0:
                print(f"  [vel] step {step:>6}  max|du|={du:.2e}  ({time.time()-t0:5.0f}s)", flush=True)
            if du < a.vel_tol:
                print(f"  [vel] steady at step {step} (max|du|={du:.2e})", flush=True); break
    sim.scalar_enabled = True

    # marching needs forward flow; flag any reversed fluid cells (recirculation)
    if chi is not None:
        u_fluid = sim.u[1:nx+1, 1:ny+1, 1:nz+1][chi[1:nx+1, 1:ny+1, 1:nz+1] < 0.5]
    else:
        u_fluid = sim.u[1:nx+1, 1:ny+1, 1:nz+1]
    umin = float(u_fluid.min())
    if umin < -1e-6:
        print(f"  WARNING: min fluid u={umin:.2e} < 0 -- streamwise recirculation; "
              f"the parabolic march clamps u>=0 and may be inaccurate there.", flush=True)

    # ---- phase 2: single downstream parabolic sweep ----
    print(f"=== parabolic march: cross_adv={a.cross} n_inner={a.n_inner} ===", flush=True)
    t0 = time.time()
    sim.solve_scalar_parabolic(cross_adv=a.cross, n_inner=a.n_inner, tol=a.tol, verbose=True)
    march_s = time.time() - t0

    Mx = Mx_profile(sim)
    st = scalar_stats(sim.scalar, nx, ny, nz, sim.dz_f, chi)
    cc = sim.scalar[1:nx+1, 1:ny+1, 1:nz+1]
    cmin, cmax = float(cc.min()), float(cc.max())
    lm = _lmix(Mx, Lx, a.thr)
    print(f"  swept {nx} planes in {march_s:.1f}s | M(out)={Mx[-1]:.4f} "
          f"c in[{cmin:.3f},{cmax:.3f}] | L_mix(M={a.thr})={lm:.3f}", flush=True)

    if a.compare:
        ref = np.load(a.compare, allow_pickle=True)
        Mx_ref = ref['Mx']; Lx_ref = float(ref['Lx'])
        n = min(len(Mx), len(Mx_ref))
        dmax = float(np.max(np.abs(Mx[:n] - Mx_ref[:n])))
        lm_ref = _lmix(Mx_ref, Lx_ref, a.thr)
        print(f"  [compare vs {os.path.basename(a.compare)}] max|dMx|={dmax:.4f} | "
              f"L_mix march={lm:.3f} ref={lm_ref:.3f} (ratio {lm/lm_ref:.3f})", flush=True)

    os.makedirs(a.outdir, exist_ok=True)
    tag = a.tag or f"{a.mode}_Sc{int(a.Sc)}_N{a.N}_march"
    final_path = os.path.join(a.outdir, f"{tag}_final.npz")
    np.savez(final_path, mode=a.mode, Sc=a.Sc, N=a.N, status='steady', method='parabolic',
             Lx=Lx, scalar=cc.detach().cpu().numpy(),
             u=sim.u[1:nx+1, 1:ny+1, 1:nz+1].detach().cpu().numpy(),
             chi_c=(chi[1:nx+1, 1:ny+1, 1:nz+1].detach().cpu().numpy() if chi is not None else None),
             Mx=Mx, x=np.linspace(0, Lx, len(Mx)))
    print(f"=== done -> {tag}_final.npz ===", flush=True)


if __name__ == "__main__":
    main()
