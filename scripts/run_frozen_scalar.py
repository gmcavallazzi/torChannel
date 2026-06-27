"""Frozen-velocity passive-scalar driver (high-Sc duct mixing).

The duct flow is steady and the scalar is passive, so we solve the velocity ONCE to
steady state, FREEZE it, and then time-march only the scalar -- skipping the momentum +
pressure-Poisson solve every step (the dominant cost) and lifting the dt ceiling off the
velocity's viscous-diffusion limit onto the scalar's (much looser at high Sc) advection limit.

Two scalar time schemes (`--scheme`):
  ssprk3 : fully-explicit SSP-RK3 (z-diffusion explicit). Strong-stability-preserving ->
           keeps TVD boundedness up to CFL~1. Best at HIGH Sc (advection-limited, big dt).
  ab2    : the IMEX AB2 + implicit-z scheme (ChannelFlow.advance_scalar). Best at LOW Sc,
           where z-diffusion would otherwise cap an explicit dt.

The converged steady scalar field is scheme-independent, so ssprk3 and ab2 agree at steady
state (validation); ssprk3 just gets there in far fewer, cheaper steps at high Sc.

Output matches scripts/mixing_campaign.py (history/snaps/final with Mx(x)) so the same
plot_campaign_* scripts work. Velocity is geometry-only (Sc- and, for `baffle`, N-independent).
"""
import os, sys, time, argparse, tempfile, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from solver import ChannelFlow
from scalar import scalar_stats
from utils import compute_divergence
from mixing_campaign import base_config, Mx_profile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='baffle', choices=['baffle', 'surface_baffle'])
    ap.add_argument('--Sc', type=float, default=100.0)
    ap.add_argument('--N', type=int, default=0)
    ap.add_argument('--nx', type=int, default=192)
    ap.add_argument('--ny', type=int, default=96)
    ap.add_argument('--nz', type=int, default=96)
    ap.add_argument('--scheme', default='ssprk3', choices=['ssprk3', 'ab2'])
    ap.add_argument('--dt_vel', type=float, default=5e-4, help="velocity solve dt (CFL-safe)")
    ap.add_argument('--dt_scalar', type=float, default=5e-3, help="scalar march dt (big at high Sc)")
    ap.add_argument('--vel_steps', type=int, default=40000, help="max velocity steps to steady")
    ap.add_argument('--vel_tol', type=float, default=1e-6, help="velocity steadiness (max|du| per check)")
    ap.add_argument('--max_steps', type=int, default=40000, help="scalar march steps")
    ap.add_argument('--check', type=int, default=200)
    ap.add_argument('--snap', type=int, default=2000)
    ap.add_argument('--drift_tol', type=float, default=2e-5)
    ap.add_argument('--min_steps', type=int, default=2000)
    ap.add_argument('--outdir', default='results/campaign')
    ap.add_argument('--tag', default=None)
    a = ap.parse_args()

    cfg = base_config(a.mode, a.Sc, a.dt_scalar, nx=a.nx, ny=a.ny, nz=a.nz)
    cfg['scalar']['N'] = a.N
    if a.mode == 'surface_baffle':
        cfg['immersed']['N'] = a.N
    Lx = cfg['domain']['Lx']
    f = tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False); yaml.safe_dump(cfg, f); f.close()
    sim = ChannelFlow(f.name)
    nx, ny, nz = sim.nx, sim.ny, sim.nz
    chi = sim.chi_c if sim.immersed_enabled else None

    # ---- phase 1: solve velocity to steady, scalar disabled (Koch IC kept aside) ----
    c_ic = sim.scalar.clone()                       # the generation-N Koch inlet IC
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
    sim.scalar.copy_(c_ic); sim._apply_scalar_bc()  # restore the fresh Koch IC on the steady flow
    sim.rhs_c_prev = None                            # fresh AB2 history for the scalar march

    # ---- phase 2: frozen-velocity scalar march ----
    os.makedirs(a.outdir, exist_ok=True)
    tag = a.tag or f"{a.mode}_Sc{int(a.Sc)}_N{a.N}_frozen_{a.scheme}"
    hist_path = os.path.join(a.outdir, f"{tag}_history.npz")
    snap_path = os.path.join(a.outdir, f"{tag}_snaps.npz")
    final_path = os.path.join(a.outdir, f"{tag}_final.npz")
    advance = sim.advance_scalar_ssprk3 if a.scheme == 'ssprk3' else sim.advance_scalar
    print(f"=== scalar march: scheme={a.scheme} dt={a.dt_scalar:.1e} ===", flush=True)

    t_hist, M_hist, drift_hist, mm_hist = [], [], [], []
    snaps = []; c_prev = None; t0 = time.time(); status = 'max_steps'

    def flush():
        np.savez(hist_path, mode=a.mode, Sc=a.Sc, N=a.N, dt=a.dt_scalar, scheme=a.scheme,
                 status=status, steps=step, Lx=Lx, t=np.array(t_hist), M=np.array(M_hist),
                 drift=np.array(drift_hist), minmax=np.array(mm_hist))

    for step in range(a.max_steps + 1):
        if step > 0:
            advance(a.dt_scalar)
        if step % a.check == 0:
            st = scalar_stats(sim.scalar, nx, ny, nz, sim.dz_f, chi)
            M = st['M']
            cc = sim.scalar[1:nx+1, 1:ny+1, 1:nz+1]
            cmin, cmax = float(cc.min()), float(cc.max())
            c_now = cc.detach().clone()
            drift = float((c_now - c_prev).abs().max()) if c_prev is not None else float('inf')
            c_prev = c_now
            t_hist.append(step * a.dt_scalar); M_hist.append(M)
            drift_hist.append(drift); mm_hist.append([cmin, cmax])
            print(f"  [N={a.N}] step {step:>6} t={step*a.dt_scalar:8.3f}  M={M:.4f}  "
                  f"drift={drift:.2e}  c in[{cmin:.3f},{cmax:.3f}]  ({time.time()-t0:5.0f}s)", flush=True)
            flush()
            if step % a.snap == 0:
                snaps.append({'t': step * a.dt_scalar, 'step': step, 'M': M, 'Mx': Mx_profile(sim),
                              'c_xy': sim.scalar[1:nx+1, 1:ny+1, nz // 2].detach().cpu().numpy()})
                np.savez(snap_path, snaps=np.array(snaps, dtype=object), Lx=Lx, dt=a.dt_scalar, mode=a.mode, N=a.N, Sc=a.Sc)
            if not (np.isfinite(M) and cmax < 1e3):
                status = 'DIVERGED'; print("  ABORT: not finite", flush=True); break
            if step >= a.min_steps and drift < a.drift_tol:
                status = 'steady'; break

    flush()
    final = dict(mode=a.mode, Sc=a.Sc, N=a.N, dt=a.dt_scalar, scheme=a.scheme, status=status,
                 steps=step, Lx=Lx, scalar=sim.scalar[1:nx+1, 1:ny+1, 1:nz+1].detach().cpu().numpy(),
                 u=sim.u[1:nx+1, 1:ny+1, 1:nz+1].detach().cpu().numpy(), chi_c=(chi[1:nx+1, 1:ny+1, 1:nz+1].detach().cpu().numpy() if chi is not None else None))
    final['Mx'] = Mx_profile(sim); final['x'] = np.linspace(0, Lx, len(final['Mx']))
    np.savez(final_path, **final)
    print(f"=== done: status={status} steps={step} -> {tag}_{{history,snaps,final}}.npz ===", flush=True)


if __name__ == "__main__":
    main()
