"""Inflow/outflow developing-duct regression: bc_x='inout' + bc_y='wall'.

A duct with a prescribed inlet profile and a convective/mass-corrected outflow must
(a) stay divergence-free after projection, (b) conserve mass (integral u_in = u_out),
and (c) remain stable while the flow develops downstream."""
import os, sys, tempfile
os.environ.setdefault("PYTORCH_JIT", "0")
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_default_dtype(torch.float64)
from solver import ChannelFlow
from utils import compute_divergence

CFG = """
grid:   {nx: 24, ny: 24, nz: 24}
domain: {Lx: 4.0, Ly: 1.0, Lz: 1.0, bc_y: wall, bc_x: inout}
flow:   {Re: 100.0, Re_tau: 10.0, U_bulk: 1.0, gamma: 1.0}
boundary_conditions: {top_wall: {type: "dirichlet"}}
initialization: {type: "parabolic", perturbation_intensity: 0.0}
solver: {type: "fft"}
time:   {dt: 0.005, n_steps: 1, t_max: 1e9, CFL_target: 0.25, dt_update_interval: 0,
         dt_max: 0.01, dt_min: 0.0001, scheme: "IMEX"}
compute: {device: "cpu"}
output: {results_folder: /tmp/torchannel_inout_test, n_out: 1000000, n_save: 100000000}
statistics: {enabled: false, n_stats: 0}
scalar: {enabled: false}
"""

if __name__ == "__main__":
    f = tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False); f.write(CFG); f.close()
    sim = ChannelFlow(f.name)
    nx, ny, nz = sim.nx, sim.ny, sim.nz
    afl = sim._inout_fluid_area_w
    maxdiv = 0.0
    for _ in range(300):
        sim.step_imex(sim.dt)
        d = compute_divergence(sim.u, sim.v, sim.w, nx, ny, nz, sim.dx, sim.dy, sim.dz_f)
        maxdiv = max(maxdiv, float(torch.max(torch.abs(d))))
    Qin = float((sim.u[0, 1:ny+1, 1:nz+1] * afl).sum())
    Qout = float((sim.u[nx, 1:ny+1, 1:nz+1] * afl).sum())
    finite = bool(torch.isfinite(sim.u).all())
    print(f"max|div| over 300 steps = {maxdiv:.3e}")
    print(f"Q_in = {Qin:.6f}   Q_out = {Qout:.6f}   |dQ| = {abs(Qin-Qout):.2e}")
    print(f"u finite = {finite}")
    ok = (maxdiv < 1e-10) and (abs(Qin - Qout) < 1e-9) and finite
    print("INFLOW/OUTFLOW DUCT TEST PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)
