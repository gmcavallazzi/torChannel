"""Plain-duct Koch mixing as pure cross-plane diffusion (the diffusive limit).

In a straight duct the laminar flow is unidirectional (v=w=0) and the Koch
interface is homogeneous in x, so the scalar advection u.grad c is identically
zero: the mixing is EXACTLY 2D diffusion in the (y,z) cross-section. This is the
proposal's diffusive limit, where L_mix(N)/L_mix(0) is provably Sc-INDEPENDENT.

This driver freezes the velocity at zero and advances only the passive scalar
(reusing the validated transport + TVD path via ChannelFlow.advance_scalar), so it
is stable at large dt and decoupled from the (here irrelevant) momentum solve. It
sweeps Koch generation N and Schmidt number Sc, and reports L_mix(N)/L_mix(0) and
chi(N)/chi(0) against the proposal's prediction r^{-Df N}.

Usage:
    python scripts/run_duct_diffusion.py configs/duct_koch.yaml \
        --Ns 0 1 2 --Scs 1 16 --dt 0.005
"""
import argparse, copy, os, sys, tempfile
import numpy as np
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
torch.set_default_dtype(torch.float64)
from solver import ChannelFlow
from scalar import scalar_stats, scalar_dissipation, apply_scalar_bc


def run_one(base_cfg, N, Sc, dt, thresh, nsteps_max, sample_every):
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault('scalar', {})
    cfg['scalar'].update({'enabled': True, 'init_type': 'koch', 'N': N, 'Sc': Sc})
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
        import yaml; yaml.safe_dump(cfg, f); cfg_path = f.name
    flow = ChannelFlow(cfg_path)
    flow.u.zero_(); flow.v.zero_(); flow.w.zero_()   # freeze velocity: pure diffusion
    nx, ny, nz = flow.nx, flow.ny, flow.nz
    bc_y = getattr(flow, 'bc_y', 'periodic'); wbc = flow.scalar_wall_bc

    def M_of():
        return scalar_stats(flow.scalar, nx, ny, nz, flow.dz_f)['M']

    def chi_of():
        apply_scalar_bc(flow.scalar, wbc, bc_y)
        return scalar_dissipation(flow.scalar, nx, ny, nz, flow.dx, flow.dy, flow.dz_f)

    ts, Ms, chis = [0.0], [M_of()], [chi_of()]
    t = 0.0; t_mix = float('nan')
    for n in range(1, nsteps_max + 1):
        flow.advance_scalar(dt); t += dt
        if n % sample_every == 0:
            M = M_of(); ts.append(t); Ms.append(M); chis.append(chi_of())
            if M < thresh:
                # linear interpolation for t_mix
                t_mix = ts[-2] + (Ms[-2]-thresh)/(Ms[-2]-Ms[-1])*(ts[-1]-ts[-2]); break
    cmin = float(flow.scalar[1:nx+1,1:ny+1,1:nz+1].min())
    cmax = float(flow.scalar[1:nx+1,1:ny+1,1:nz+1].max())
    chi_e = chis[1] if len(chis) > 1 else chis[0]
    return dict(N=N, Sc=Sc, t_mix=t_mix, chi_early=chi_e, cmin=cmin, cmax=cmax,
                ts=np.array(ts), Ms=np.array(Ms), chis=np.array(chis))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--Ns', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--Scs', type=float, nargs='+', default=[1.0, 16.0])
    ap.add_argument('--dt', type=float, default=0.005)
    ap.add_argument('--thresh', type=float, default=0.05)
    ap.add_argument('--nsteps_max', type=int, default=80000)
    ap.add_argument('--sample_every', type=int, default=40)
    ap.add_argument('--out', default='results/duct_diffusion')
    args = ap.parse_args()
    with open(args.config) as f:
        import yaml; base = yaml.safe_load(f)
    base.setdefault('scalar', {}); base['scalar']['scheme'] = base['scalar'].get('scheme', 'tvd')
    os.makedirs(args.out, exist_ok=True)
    r = base['scalar'].get('r', 3.0); Df = np.log(4.0)/np.log(r)

    rows = {}
    for Sc in args.Scs:
        for N in args.Ns:
            print(f"--- Sc={Sc:g} N={N} ---", flush=True)
            rows[(Sc, N)] = run_one(base, N, Sc, args.dt, args.thresh,
                                    args.nsteps_max, args.sample_every)

    print(f"\n=== DUCT DIFFUSIVE-LIMIT N-SWEEP (r={r:g}, Df={Df:.3f}) ===")
    print(f"prediction r^-Df*N: " + "  ".join(f"N{N}={r**(-Df*N):.3f}" for N in args.Ns))
    for Sc in args.Scs:
        L0 = rows[(Sc, args.Ns[0])]['t_mix']
        chi0 = rows[(Sc, args.Ns[0])]['chi_early']
        print(f"\n Sc={Sc:g}:")
        print(f"   {'N':>2} {'t_mix=L_mix':>11} {'L_mix(N)/L(0)':>13} {'chi_early':>11} "
              f"{'chi(N)/chi(0)':>13} {'c-range':>16}")
        for N in args.Ns:
            d = rows[(Sc, N)]
            lr = d['t_mix']/L0 if np.isfinite(d['t_mix']) and np.isfinite(L0) else float('nan')
            print(f"   {N:>2} {d['t_mix']:>11.3f} {lr:>13.3f} {d['chi_early']:>11.3e} "
                  f"{d['chi_early']/chi0:>13.3f}  [{d['cmin']:.2e},{d['cmax']:.3f}]")
    np.savez(os.path.join(args.out, 'duct_diffusion.npz'),
             **{f"Sc{Sc:g}_N{N}_t": rows[(Sc,N)]['ts'] for Sc in args.Scs for N in args.Ns},
             **{f"Sc{Sc:g}_N{N}_M": rows[(Sc,N)]['Ms'] for Sc in args.Scs for N in args.Ns},
             **{f"Sc{Sc:g}_N{N}_chi": rows[(Sc,N)]['chis'] for Sc in args.Scs for N in args.Ns},
             r=r, Df=Df)
    print(f"\nwrote {args.out}/duct_diffusion.npz", flush=True)


if __name__ == "__main__":
    main()
