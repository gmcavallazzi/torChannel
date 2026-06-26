"""Verify the pipe_koch wall is a genuine Koch fractal border (Df=log4/log3).
Plots the analytic wall boundary R(theta) for N=0..3 at high resolution (the TRUE
intended geometry, independent of the simulation grid), plus a zoom on one arc."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from immersed import _koch_zigzag_disp

usetex = os.environ.get("TORCHANNEL_USETEX", "1") == "1"
plt.rcParams.update({"text.usetex": usetex, "font.family": "serif", "font.size": 11})

R, cy, cz = 0.42, 0.5, 0.5
amp = 0.15 * R           # same as the run
r = 3.0
n_lobes = 1
th = np.linspace(0, 2*np.pi, 40001)
tn = ((th + np.pi) / (2*np.pi) * n_lobes) % 1.0


def wall(N):
    d = _koch_zigzag_disp(N, r, tn)
    m = np.max(np.abs(d))
    dhat = d/m if m > 0 else d
    Rw = R + amp * dhat
    return cy + Rw*np.cos(th), cz + Rw*np.sin(th)


fig, axs = plt.subplots(1, 2, figsize=(9, 4.4))
colors = ['0.5', 'C0', 'C1', 'C3']
for N in range(4):
    y, z = wall(N)
    axs[0].plot(y, z, color=colors[N], lw=1.4, label=r"$N=%d$" % N)
axs[0].set_aspect('equal'); axs[0].set_xlabel(r"$y$"); axs[0].set_ylabel(r"$z$")
axs[0].legend(loc='upper right', fontsize=9); axs[0].set_title(r"Koch wall, $D_f=\log 4/\log 3$")

# zoom on a 90-deg arc to expose the self-similar folds
for N in range(4):
    y, z = wall(N)
    axs[1].plot(y, z, color=colors[N], lw=1.6)
axs[1].set_aspect('equal'); axs[1].set_xlim(0.5, 0.96); axs[1].set_ylim(0.5, 0.96)
axs[1].set_xlabel(r"$y$"); axs[1].set_ylabel(r"$z$"); axs[1].set_title(r"Zoom")
fig.tight_layout()
os.makedirs("results/figures", exist_ok=True)
fig.savefig("results/figures/fractal_wall_check.png", dpi=150, bbox_inches='tight')
print("wrote results/figures/fractal_wall_check.png")
