"""Fractal inlet-SURFACE proxy: sweep the WALL Koch generation (koch_herringbone)
with a FLAT scalar interface, and measure L_mix. Tests whether a multi-scale
(fractal) corrugated wall folds a flat interface faster than the smooth (N=0) groove."""
import os, sys, copy, tempfile, yaml
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_JIT", "0")
from solver import ChannelFlow
from scalar import scalar_stats, scalar_dissipation, apply_scalar_bc

base = yaml.safe_load(open("configs/herringbone_duct_strong.yaml"))
base['immersed']['kind'] = 'koch_herringbone'
base['immersed']['r'] = 3.0
base['immersed']['koch_amp'] = 1.0
base['scalar']['init_type'] = 'interface_y'   # FLAT two-stream interface (wall does the folding)
base['scalar']['interface_pos'] = 0.5
dt, nsteps, sample_every, thresh = 0.0025, 16000, 80, 0.05

def run(wallN):
    c = copy.deepcopy(base); c['immersed']['N'] = wallN
    f = tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False); yaml.safe_dump(c, f); f.close()
    sim = ChannelFlow(f.name); nx, ny, nz = sim.nx, sim.ny, sim.nz
    chi_c, bc_y, wbc = sim.chi_c, sim.bc_y, sim.scalar_wall_bc
    M = lambda: scalar_stats(sim.scalar, nx, ny, nz, sim.dz_f, chi_c=chi_c)['M']
    def CHI():
        apply_scalar_bc(sim.scalar, wbc, bc_y)
        return scalar_dissipation(sim.scalar, nx, ny, nz, sim.dx, sim.dy, sim.dz_f, chi_c=chi_c)
    ts, Ms, chis = [0.0], [M()], [CHI()]; t = 0.0; tmix = float('nan')
    for n in range(1, nsteps+1):
        sim.step_imex(sim.dt); t += sim.dt
        if n % sample_every == 0:
            ts.append(t); Ms.append(M()); chis.append(CHI())
            if Ms[-1] < thresh:
                tmix = ts[-2] + (Ms[-2]-thresh)/(Ms[-2]-Ms[-1])*(ts[-1]-ts[-2]); break
    chi_e = chis[1] if len(chis) > 1 else chis[0]
    print(f"  wallN={wallN}: t_mix=L_mix={tmix:.3f}  chi_early={chi_e:.3e}", flush=True)
    return dict(N=wallN, tmix=tmix, chi=chi_e, ts=np.array(ts), Ms=np.array(Ms), chis=np.array(chis))

rows = {}
for wn in [0, 1, 2]:
    print(f"=== wall N={wn} ===", flush=True)
    rows[wn] = run(wn)
r = base['immersed']['r']; Df = np.log(4)/np.log(r)
L0 = rows[0]['tmix']; chi0 = rows[0]['chi']
print(f"\n=== FRACTAL-WALL (surface proxy) SWEEP: flat interface, Sc=16, Df={Df:.3f} ===")
print(f"{'wallN':>5} {'L_mix':>9} {'L(N)/L(0)':>10} {'pred r^-DfN':>12} {'chi(N)/chi(0)':>13}")
for wn in [0,1,2]:
    d = rows[wn]
    print(f"{wn:>5} {d['tmix']:>9.3f} {d['tmix']/L0:>10.3f} {r**(-Df*wn):>12.3f} {d['chi']/chi0:>13.3f}")
os.makedirs("results/fractal_wall", exist_ok=True)
np.savez("results/fractal_wall/wall_sweep.npz",
         **{f"N{wn}_t": rows[wn]['ts'] for wn in [0,1,2]},
         **{f"N{wn}_M": rows[wn]['Ms'] for wn in [0,1,2]},
         **{f"N{wn}_chi": rows[wn]['chis'] for wn in [0,1,2]}, r=r, Df=Df)
print("wrote results/fractal_wall/wall_sweep.npz")
