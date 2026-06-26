"""Sc-campaign driver: fractal-mixing N-sweep at fixed dt with convergence early-stop.

Two modes:
  baffle          : temporal periodic box, smooth square duct, Koch BAFFLE interface IC
                    (the proposal's diffusive-limit test; no wall surface). x-homogeneous,
                    so a thin streamwise box suffices. Mixing measured as global M(t),
                    with t == downstream distance x = U*t. Early-stop when fully mixed.
  surface_baffle  : developing duct (inflow/outflow), circular fractal WALL surface at the
                    inlet + Koch baffle. Mixing measured as the steady M(x) profile.
                    Early-stop when the field reaches steady state.

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
    if mode == 'baffle':
        return dict(
            grid={'nx': 16, 'ny': 128, 'nz': 128},
            domain={'Lx': 2.0, 'Ly': 1.0, 'Lz': 1.0, 'bc_y': 'wall', 'bc_x': 'periodic'},
            flow={'Re': 40.0, 'Re_tau': 10.0, 'U_bulk': 1.0, 'gamma': 1.0},
            boundary_conditions={'top_wall': {'type': 'dirichlet'}},
            initialization={'type': 'parabolic', 'perturbation_intensity': 0.0},
            solver={'type': 'fft'},
            time={'dt': dt, 'n_steps': 1, 't_max': 1e9, 'CFL_target': 0.25,
                  'dt_update_interval': 0, 'dt_max': dt, 'dt_min': dt, 'scheme': 'IMEX'},
            compute={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            output={'results_folder': '/tmp/torchannel_campaign', 'n_out': 10**9, 'n_save': 10**9},
            statistics={'enabled': False, 'n_stats': 0},
            scalar={'enabled': True, 'Sc': Sc, 'wall_bc': 'neumann', 'scheme': 'tvd',
                    'init_type': 'koch', 'N': 0, 'r': 3.0, 'eps_cells': 1.0, 'theta': 0.5})
    if mode == 'surface_baffle':
        # Short developing duct: the N-dependence lives near the inlet (Koch folds +
        # the surface's secondary flow), so we resolve that region well rather than the
        # long diffusive tail. Lx=6 captures the inlet + early M(x) decay where N differs.
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
            immersed={'enabled': True, 'kind': 'pipe_koch', 'eta': 1.0e-4,
                      'pipe_R': 0.42, 'pipe_yc': 0.5, 'pipe_zc': 0.5,
                      'N': 0, 'r': 3.0, 'koch_amp': 0.15, 'n_lobes': 6, 'inlet_len': 1.0},
            scalar={'enabled': True, 'Sc': Sc, 'wall_bc': 'neumann', 'scheme': 'tvd',
                    'init_type': 'koch', 'N': 0, 'r': 3.0, 'eps_cells': 1.0, 'theta': 0.5})
    raise ValueError(mode)


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
             drift_tol=2e-5, min_steps=2000, outdir='results/campaign'):
    cfg = base_config(mode, Sc, dt)
    cfg['scalar']['N'] = N
    if mode == 'surface_baffle':
        cfg['immersed']['N'] = N
    Lx = cfg['domain']['Lx']
    f = tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False); yaml.safe_dump(cfg, f); f.close()
    sim = ChannelFlow(f.name)
    nx, ny, nz = sim.nx, sim.ny, sim.nz
    chi = sim.chi_c if sim.immersed_enabled else None
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
        if mode == 'surface_baffle':
            s['Mx'] = Mx_profile(sim)
            s['c_xy'] = sim.scalar[1:nx+1, 1:ny+1, nz // 2].detach().cpu().numpy()    # mid-z (x,y)
            s['c_yz_in'] = sim.scalar[max(1, nx // 12), 1:ny+1, 1:nz+1].detach().cpu().numpy()  # near inlet
        else:
            s['c_yz'] = sim.scalar[nx // 2, 1:ny+1, 1:nz+1].detach().cpu().numpy()    # cross-section
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
    if mode == 'surface_baffle':
        final['Mx'] = Mx_profile(sim)
        final['x'] = np.linspace(0, Lx, len(final['Mx']))
        final['chi_c'] = sim.chi_c[1:nx+1, 1:ny+1, 1:nz+1].detach().cpu().numpy().astype(np.float32)
    np.savez(final_path, **final)
    print(f"  [N={N}] DONE status={status} steps={step}  -> {tag}_{{history,snaps,final}}.npz", flush=True)
    return final


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
    for N in a.Ns:
        run_case(a.mode, a.Sc, N, a.dt, a.max_steps, check=a.check, snap=a.snap,
                 M_stop=a.M_stop, drift_tol=a.drift_tol, min_steps=a.min_steps, outdir=a.outdir)
    print("=== campaign complete ===", flush=True)
