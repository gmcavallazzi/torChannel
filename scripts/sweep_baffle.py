"""Fast Sc-sweep over Koch generation N for the BAFFLE (smooth pipe), high Sc.

Two facts make this cheap:
  1. The baffle wall is N-INDEPENDENT, so the steady velocity is solved ONCE and reused
     for every N (and cached to disk).
  2. The pipe flow is SMOOTH, so the velocity is well-resolved on a COARSE cross-section;
     only the scalar needs the fine grid (to resolve the Koch striations without false
     diffusion). We solve velocity coarse, interpolate it onto the fine scalar grid, and
     get each N's steady scalar in a single parabolic sweep (scalar.march_scalar_steady).

So the whole N-sweep costs ~ one coarse velocity solve + N cheap scalar sweeps, instead of
one fine velocity solve PER N plus a high-Sc time-march per N.

Baffle only (v=w=0): the marcher runs with cross_adv=False and a cell-centred streamwise
velocity (u_cc) interpolated from the coarse solve. Output matches the campaign layout
(`{mode}_Sc{Sc}_N{N}_march_final.npz`) so plot_campaign_* / compare scripts work.

Example (the planned run):
  python scripts/sweep_baffle.py --Sc 100 --Ns 0 1 2 3 4 \
         --nx 192 --ny 256 --nz 256 --vel_ny 128 --vel_nz 128 --dt_vel 2.5e-4
"""
import os, sys, time, argparse, tempfile, yaml, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from solver import ChannelFlow
from scalar import march_scalar_steady, scalar_stats
from mixing_campaign import base_config, Mx_profile


def _interp1d(arr, src, dst, axis):
    """Linear interpolation of `arr` along `axis` from coords `src` to `dst` (edge-clamped)."""
    idx = torch.searchsorted(src.contiguous(), dst.contiguous()).clamp(1, len(src) - 1)
    lo, hi = idx - 1, idx
    w = ((dst - src[lo]) / (src[hi] - src[lo])).clamp(0.0, 1.0)
    shape = [1] * arr.dim(); shape[axis] = -1
    w = w.view(shape)
    return arr.index_select(axis, lo) * (1.0 - w) + arr.index_select(axis, hi) * w


def _cell_axes(sim):
    """Physical cell-centre coordinates (x, y, z) of a ChannelFlow's interior grid."""
    dev = sim.u.device
    x = (torch.arange(sim.nx, device=dev, dtype=torch.float64) + 0.5) * sim.dx
    y = (torch.arange(sim.ny, device=dev, dtype=torch.float64) + 0.5) * sim.dy
    z = sim.z_c[1:sim.nz + 1].to(dev)
    return x, y, z


def _build(mode, Sc, dt, nx, ny, nz, N, Lx=6.0):
    cfg = base_config(mode, Sc, dt, nx=nx, ny=ny, nz=nz, Lx=Lx)
    cfg['scalar']['N'] = N
    f = tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False); yaml.safe_dump(cfg, f); f.close()
    return ChannelFlow(f.name)


def _lmix(Mx, Lx, thr):
    x = np.linspace(0.0, Lx, len(Mx)); below = np.where(Mx < thr)[0]
    if len(below) == 0 or below[0] == 0:
        return float('nan')
    i = below[0]; m0, m1 = Mx[i - 1], Mx[i]
    return float(x[i - 1] + (thr - m0) * (x[i] - x[i - 1]) / (m1 - m0)) if m1 != m0 else float(x[i])


def solve_coarse_velocity(mode, nx, vny, vnz, dt_vel, vel_steps, vel_tol, cache):
    """Solve the (N-independent) baffle velocity on a coarse cross-section; cache to disk.
    Returns (u_cc_coarse interior (nx,vny,vnz), coarse axes xc,yc,zc)."""
    if cache and os.path.exists(cache):
        d = np.load(cache)
        if (int(d['nx']), int(d['ny']), int(d['nz'])) == (nx, vny, vnz):
            print(f"  [vel] loaded cache {cache} ({nx}x{vny}x{vnz})", flush=True)
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
            t = lambda a: torch.tensor(a, dtype=torch.float64, device=dev)
            return t(d['u_cc']), t(d['xc']), t(d['yc']), t(d['zc'])
    sim = _build(mode, 1.0, dt_vel, nx, vny, vnz, 0)
    sim.scalar_enabled = False
    print(f"  [vel] solving coarse {nx}x{vny}x{vnz} dt={dt_vel:.1e} ...", flush=True)
    t0 = time.time(); u_prev = sim.u.detach().clone()
    for step in range(1, vel_steps + 1):
        sim.step_imex(dt_vel)
        if step % 500 == 0:
            du = float((sim.u - u_prev).abs().max()); u_prev = sim.u.detach().clone()
            if step % 4000 == 0:
                print(f"    step {step:>6}  max|du|/500={du:.2e}  ({time.time()-t0:5.0f}s)", flush=True)
            if du < vel_tol:
                print(f"    steady at step {step} (max|du|/500={du:.2e}, {time.time()-t0:.0f}s)", flush=True)
                break
    nxx, ny, nz = sim.nx, sim.ny, sim.nz
    u_cc = 0.5 * (sim.u[0:nxx, 1:ny+1, 1:nz+1] + sim.u[1:nxx+1, 1:ny+1, 1:nz+1])  # cell-centred (nx,ny,nz)
    xc, yc, zc = _cell_axes(sim)
    if cache:
        np.savez(cache, u_cc=u_cc.cpu().numpy(), xc=xc.cpu().numpy(), yc=yc.cpu().numpy(),
                 zc=zc.cpu().numpy(), nx=nxx, ny=ny, nz=nz)
        print(f"  [vel] cached -> {cache}", flush=True)
    return u_cc, xc, yc, zc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='baffle', choices=['baffle'])
    ap.add_argument('--Sc', type=float, default=100.0)
    ap.add_argument('--Ns', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--nx', type=int, default=192, help="scalar streamwise planes (over --Lx)")
    ap.add_argument('--ny', type=int, default=256)
    ap.add_argument('--nz', type=int, default=256)
    ap.add_argument('--Lx', type=float, default=6.0, help="scalar domain length (mixing length grows ~Pe)")
    ap.add_argument('--vel_nx', type=int, default=192, help="velocity streamwise planes (always over Lx=6, the developing region)")
    ap.add_argument('--vel_ny', type=int, default=128, help="coarse velocity cross-section (y)")
    ap.add_argument('--vel_nz', type=int, default=128, help="coarse velocity cross-section (z)")
    ap.add_argument('--dt_vel', type=float, default=2.5e-4)
    ap.add_argument('--vel_steps', type=int, default=60000)
    ap.add_argument('--vel_tol', type=float, default=1e-5)
    ap.add_argument('--n_inner', type=int, default=20)
    ap.add_argument('--tol', type=float, default=1e-10)
    ap.add_argument('--thr', type=float, default=0.90)
    ap.add_argument('--vel_cache', default=None, help="npz path to cache/load the coarse velocity")
    ap.add_argument('--outdir', default='results/campaign')
    a = ap.parse_args()

    cache = a.vel_cache or os.path.join(a.outdir, f"velcache_{a.mode}_{a.vel_nx}x{a.vel_ny}x{a.vel_nz}.npz")
    os.makedirs(a.outdir, exist_ok=True)

    # ---- velocity: solve ONCE on the coarse cross-section (N-independent), cache ----
    # Always solved over the developing region Lx=6 (the flow is fully developed by x~3); for a
    # longer scalar domain the x-interpolation edge-clamps to this developed profile (exact).
    u_cc_c, xc, yc, zc = solve_coarse_velocity(a.mode, a.vel_nx, a.vel_ny, a.vel_nz,
                                               a.dt_vel, a.vel_steps, a.vel_tol, cache)

    # ---- interpolate coarse velocity onto the fine scalar grid ONCE (N-independent) ----
    probe = _build(a.mode, a.Sc, 1e-3, a.nx, a.ny, a.nz, 0, Lx=a.Lx)
    xf, yf, zf = _cell_axes(probe)
    nx, ny, nz = probe.nx, probe.ny, probe.nz
    u_int = _interp1d(_interp1d(_interp1d(u_cc_c, xc, xf, 0), yc, yf, 1), zc, zf, 2)  # (nx,ny,nz)
    chi_f = probe.chi_c[1:nx+1, 1:ny+1, 1:nz+1].clone()       # detach from probe so it can be freed
    u_int = u_int * (1.0 - chi_f)                              # zero inside the fine solid mask
    u_cc = torch.zeros(nx+2, ny+2, nz+2, dtype=torch.float64, device=u_int.device)
    u_cc[1:nx+1, 1:ny+1, 1:nz+1] = u_int
    print(f"  [vel] interpolated {a.vel_nx}x{a.vel_ny}x{a.vel_nz} (Lx=6) -> {nx}x{ny}x{nz} (Lx={a.Lx:g}); "
          f"max u_cc={float(u_int.max()):.3f}", flush=True)
    dummy = torch.zeros_like(u_cc)

    # ---- per-N parabolic scalar sweep on the fine grid ----
    Lx = float(probe.Lx)
    # Free the interpolation solver + temporaries: only ONE fine ChannelFlow may be resident at
    # a time (each is ~50 GB at 576x768x768), or the sweep OOMs at the N->N+1 transition.
    del probe, u_int, u_cc_c, xf, yf, zf
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"=== sweep {a.mode} Sc={a.Sc} N={a.Ns} grid={nx}x{ny}x{nz} ===", flush=True)
    summary = []
    for N in a.Ns:
        sim = _build(a.mode, a.Sc, 1e-3, a.nx, a.ny, a.nz, N, Lx=a.Lx)
        D = sim.scalar_D
        t0 = time.time()
        sim.scalar = march_scalar_steady(sim.c_inlet, dummy, dummy, dummy, nx, ny, nz,
                                         sim.dx, sim.dy, sim.dz_c, sim.dz_f, D,
                                         bc_y=sim.bc_y, wall_bc=sim.scalar_wall_bc,
                                         cross_adv=False, n_inner=a.n_inner, tol=a.tol,
                                         verbose=False, u_cc=u_cc)
        sim._apply_scalar_bc()
        Mx = Mx_profile(sim); cc = sim.scalar[1:nx+1, 1:ny+1, 1:nz+1]
        cmin, cmax = float(cc.min()), float(cc.max())
        lm = _lmix(Mx, Lx, a.thr)
        print(f"  N={N}: {time.time()-t0:4.1f}s  M(out)={Mx[-1]:.4f}  c in[{cmin:.3f},{cmax:.3f}]  "
              f"L_mix(M={a.thr})={lm:.3f}", flush=True)
        # lightweight output: Mx(x) (the result) + a mid-z (x,y) slice and a handful of
        # (y,z) cross-sections for visualisation -- NOT the full 3D field (256^3 ~ 0.2 GB/N).
        tag = f"{a.mode}_Sc{int(a.Sc)}_N{N}_march"
        xs_idx = np.linspace(0, nx - 1, 6).astype(int)
        np.savez(os.path.join(a.outdir, f"{tag}.npz"),
                 mode=a.mode, Sc=a.Sc, N=N, status='steady', method='parabolic', Lx=Lx,
                 Mx=Mx, x=np.linspace(0, Lx, len(Mx)),
                 c_xy=cc[:, :, nz // 2].detach().cpu().numpy(),               # (nx, ny) mid-z slice
                 c_yz=cc[xs_idx].detach().cpu().numpy(),                      # (6, ny, nz)
                 chi_yz=chi_f[xs_idx[0]].detach().cpu().numpy(),              # (ny, nz) wall mask
                 xs=np.linspace(0, Lx, nx)[xs_idx])
        summary.append((N, lm))
        # release this N's fine solver + scalar field before building the next one
        del sim, cc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n=== L_mix(N)/L_mix(0) summary (M={:.2f}) ===".format(a.thr), flush=True)
    l0 = summary[0][1] if summary and np.isfinite(summary[0][1]) else float('nan')
    for N, lm in summary:
        print(f"  N={N}: L_mix={lm:.3f}  ratio={lm/l0:.3f}", flush=True)


if __name__ == "__main__":
    main()
