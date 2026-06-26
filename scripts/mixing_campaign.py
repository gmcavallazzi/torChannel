"""Sc-campaign driver: fractal-mixing N-sweep at fixed dt with convergence early-stop.

Two modes:
  baffle          : SHORT developing duct (inflow/outflow), SMOOTH circular wall + Koch
                    BAFFLE interface prescribed at the inlet (no wall surface). Mixing is
                    the steady M(x) profile; the time-march to steady state is the only
                    transient. Early-stop at steady state.
  surface_baffle  : same short developing duct but with the circular fractal WALL surface
                    at the inlet (in addition to the Koch baffle). Differs from `baffle`
                    ONLY by the wall fractal -> clean controlled comparison. Steady M(x),
                    early-stop at steady state.

Both are developing ducts (steady M(x)); the only physical transient is the laminar
fill-in to steady state, which the drift-based early-stop detects.

Design choices requested:
  * FIXED dt (time.dt_update_interval = 0) for performance and reproducibility.
  * EARLY STOP on convergence: every CHECK steps we measure the global mixedness M and the
    field drift (max |c(t)-c(t-CHECK)|). Stop when the scalar stops changing (drift < tol;
    steady state OR fully mixed) or M < M_stop, capped by max_steps.
  * DIVERGENCE GUARD: every CHECK we check max|div| in the fluid and finiteness; abort the
    case (and report) if it blows up — so a bad dt fails loudly instead of silently.

Saves one npz per N (t, M, and for surface mode the final M(x)); a separate plot script
renders figures (keeps heavy compute decoupled from usetex/plotting).
"""
import os, sys, copy, tempfile, yaml, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_JIT", "0")
import numpy as np, torch
from solver import ChannelFlow
from scalar import scalar_stats
from utils import compute_divergence


def base_config(mode, Sc, dt):
    # Short developing duct, circular cross-section. The N-dependence lives near the inlet
    # (Koch folds + the surface's secondary flow), so Lx=6 resolves that region rather than
    # the long diffusive tail. `baffle` uses a SMOOTH circular wall (kind='pipe'); the wall
    # fractal is added only in `surface_baffle` (kind='pipe_koch') -> the two differ ONLY by
    # the wall, with the same R=0.42 available cross-section.
    if mode == 'baffle':
        immersed = {'enabled': True, 'kind': 'pipe', 'eta': 1.0e-4,
                    'pipe_R': 0.42, 'pipe_yc': 0.5, 'pipe_zc': 0.5}
    elif mode == 'surface_baffle':
        immersed = {'enabled': True, 'kind': 'pipe_koch', 'eta': 1.0e-4,
                    'pipe_R': 0.42, 'pipe_yc': 0.5, 'pipe_zc': 0.5,
                    'N': 0, 'r': 3.0, 'koch_amp': 0.15, 'n_lobes': 6, 'inlet_len': 1.0}
    else:
        raise ValueError(mode)
    return dict(
        grid={'nx': 64, 'ny': 96, 'nz': 96},
        domain={'Lx': 6.0, 'Ly': 1.0, 'Lz': 1.0, 'bc_y': 'wall', 'bc_x': 'inout'},
        flow={'Re': 40.0, 'Re_tau': 10.0, 'U_bulk': 1.0, 'gamma': 1.0},
        boundary_conditions={'top_wall': {'type': 'dirichlet'}},
        initialization={'type': 'parabolic', 'perturbation_intensity': 0.0},
        solver={'type': 'fft'},
        time={'dt': dt, 'n_steps': 1, 't_max': 1e9, 'CFL_target': 0.25,
              'dt_update_interval': 0, 'dt_max': dt, 'dt_min': dt, 'scheme': 'IMEX'},
        compute={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
        output={'results_folder': '/tmp/torchannel_campaign', 'n_out': 10**9, 'n_save': 10**9},
        statistics={'enabled': False, 'n_stats': 0},
        immersed=immersed,
        scalar={'enabled': True, 'Sc': Sc, 'wall_bc': 'neumann', 'scheme': 'tvd',
                'init_type': 'koch', 'N': 0, 'r': 3.0, 'eps_cells': 1.0, 'theta': 0.5})


def Mx_profile(sim):
    """Fluid-masked segregation per streamwise slice (developing duct)."""
    nx, ny, nz = sim.nx, sim.ny, sim.nz
    c = sim.scalar[1:nx+1, 1:ny+1, 1:nz+1]
    fl = (1.0 - sim.chi_c[1:nx+1, 1:ny+1, 1:nz+1]) if sim.immersed_enabled else torch.ones_like(c)
    wz = sim.dz_f[0:nz].view(1, 1, -1)
    vol = (fl * wz).clamp(min=0)
    A = vol.sum(dim=(1, 2)).clamp(min=1e-30)
    mean = (c * vol).sum(dim=(1, 2)) / A
    var = (((c - mean.view(-1, 1, 1))**2) * vol).sum(dim=(1, 2)) / A
    std = torch.sqrt(var.clamp(min=0))
    std_max = torch.sqrt((mean * (1 - mean)).clamp(min=1e-30))
    return (std / std_max).detach().cpu().numpy()


def run_case(mode, Sc, N, dt, max_steps, check=500, snap=4000, M_stop=0.02,
             drift_tol=2e-5, min_steps=2000, outdir='results/campaign', warm=None):
    cfg = base_config(mode, Sc, dt)
    cfg['scalar']['N'] = N
    if mode == 'surface_baffle':
        cfg['immersed']['N'] = N
    Lx = cfg['domain']['Lx']
    f = tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False); yaml.safe_dump(cfg, f); f.close()
    sim = ChannelFlow(f.name)
    nx, ny, nz = sim.nx, sim.ny, sim.nz
    chi = sim.chi_c if sim.immersed_enabled else None

    # WARM START: seed velocity + scalar from a previous converged case (same grid). The
    # steady state is unique, so this only shortens the path to it -- the new inlet/wall BCs
    # for THIS N are re-imposed below, and the drift early-stop still verifies steadiness.
    # For the baffle the wall (hence flow) is identical across N, so the flow starts already
    # developed; only the near-inlet scalar fold has to re-adjust.
    if warm is not None:
        with torch.no_grad():
            sim.u.copy_(warm['u']); sim.v.copy_(warm['v']); sim.w.copy_(warm['w'])
            sim.scalar.copy_(warm['scalar'])
        sim.apply_bc_uvw()
        sim._apply_scalar_bc()
        sim.rhs_u_prev = sim.rhs_v_prev = sim.rhs_w_prev = None   # drop stale AB2 history
        sim.rhs_c_prev = None
        print(f"  [N={N}] warm-started from previous converged case", flush=True)
    os.makedirs(outdir, exist_ok=True)
    tag = f"{mode}_Sc{int(Sc)}_N{N}"
    hist_path = os.path.join(outdir, f"{tag}_history.npz")
    snap_path = os.path.join(outdir, f"{tag}_snaps.npz")
    final_path = os.path.join(outdir, f"{tag}_final.npz")

    # convergence/time-series history (saved every `check`) and field snapshots
    # (saved every `snap`) are both flushed to disk incrementally, so a long job that
    # is killed still leaves usable, post-processable data behind.
    t_hist, M_hist, drift_hist, div_hist = [], [], [], []
    snaps = []                                  # list of dicts -> object array on save
    c_prev = None
    t0 = time.time()
    status = 'max_steps'

    def take_snapshot(stp, M):
        s = {'t': stp * dt, 'step': stp, 'M': M}
        s['Mx'] = Mx_profile(sim)
        s['c_xy'] = sim.scalar[1:nx+1, 1:ny+1, nz // 2].detach().cpu().numpy()        # mid-z (x,y)
        s['c_yz_in'] = sim.scalar[max(1, nx // 12), 1:ny+1, 1:nz+1].detach().cpu().numpy()  # near inlet
        snaps.append(s)
        np.savez(snap_path, snaps=np.array(snaps, dtype=object), Lx=Lx, dt=dt, mode=mode, N=N, Sc=Sc)

    def flush_history(stp):
        np.savez(hist_path, mode=mode, Sc=Sc, N=N, dt=dt, status=status, steps=stp, Lx=Lx,
                 t=np.array(t_hist), M=np.array(M_hist),
                 drift=np.array(drift_hist), divmax=np.array(div_hist))

    for step in range(max_steps + 1):
        if step > 0:
            sim.step_imex(sim.dt)
        if step % check == 0:
            st = scalar_stats(sim.scalar, nx, ny, nz, sim.dz_f, chi)
            M = st['M']
            d = compute_divergence(sim.u, sim.v, sim.w, nx, ny, nz, sim.dx, sim.dy, sim.dz_f)
            da = torch.abs(d)
            if chi is not None:
                da = da[(chi[1:nx+1, 1:ny+1, 1:nz+1] < 0.5)]
            divmax = float(da.max())
            finite = bool(torch.isfinite(sim.scalar).all() and torch.isfinite(sim.u).all())
            c_now = sim.scalar[1:nx+1, 1:ny+1, 1:nz+1].detach().clone()
            drift = float((c_now - c_prev).abs().max()) if c_prev is not None else float('inf')
            c_prev = c_now
            t_hist.append(step * dt); M_hist.append(M)
            drift_hist.append(drift); div_hist.append(divmax)
            print(f"  [N={N}] step {step:>7} t={step*dt:8.3f}  M={M:.4f}  "
                  f"drift={drift:.2e}  fluid|div|={divmax:.1e}  ({time.time()-t0:5.0f}s)", flush=True)
            flush_history(step)
            if step % snap == 0:
                take_snapshot(step, M)
            if not finite or divmax > 1e3:
                status = 'DIVERGED'; flush_history(step)
                print(f"  [N={N}] ABORT: not finite / div blew up", flush=True); break
            if M < M_stop:
                status = 'mixed'; break
            if step >= min_steps and drift < drift_tol:
                status = 'steady'; break

    flush_history(step)
    take_snapshot(step, M_hist[-1] if M_hist else float('nan'))      # always snapshot final state
    final = dict(mode=mode, Sc=Sc, N=N, dt=dt, status=status, steps=step, Lx=Lx,
                 scalar=sim.scalar[1:nx+1, 1:ny+1, 1:nz+1].detach().cpu().numpy().astype(np.float32),
                 u=sim.u[1:nx+1, 1:ny+1, 1:nz+1].detach().cpu().numpy().astype(np.float32))
    final['Mx'] = Mx_profile(sim)
    final['x'] = np.linspace(0, Lx, len(final['Mx']))
    final['chi_c'] = sim.chi_c[1:nx+1, 1:ny+1, 1:nz+1].detach().cpu().numpy().astype(np.float32)
    np.savez(final_path, **final)
    print(f"  [N={N}] DONE status={status} steps={step}  -> {tag}_{{history,snaps,final}}.npz", flush=True)
    # full ghosted state (same grid across N) -> warm start for the next case
    state = {k: getattr(sim, k).detach().clone() for k in ('u', 'v', 'w', 'scalar')}
    return final, state


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['baffle', 'surface_baffle'])
    ap.add_argument('--Sc', type=float, default=10.0)
    ap.add_argument('--Ns', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--dt', type=float, required=True)
    ap.add_argument('--max_steps', type=int, default=2_000_000)
    ap.add_argument('--check', type=int, default=500)
    ap.add_argument('--snap', type=int, default=4000)
    ap.add_argument('--M_stop', type=float, default=0.02)
    ap.add_argument('--drift_tol', type=float, default=2e-5)
    ap.add_argument('--min_steps', type=int, default=2000)
    ap.add_argument('--outdir', default='results/campaign')
    a = ap.parse_args()
    print(f"=== campaign mode={a.mode} Sc={a.Sc} dt={a.dt} Ns={a.Ns} ===", flush=True)
    warm = None
    for N in a.Ns:
        _, warm = run_case(a.mode, a.Sc, N, a.dt, a.max_steps, check=a.check, snap=a.snap,
                           M_stop=a.M_stop, drift_tol=a.drift_tol, min_steps=a.min_steps,
                           outdir=a.outdir, warm=warm)
    print("=== campaign complete ===", flush=True)
