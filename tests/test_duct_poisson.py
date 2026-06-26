"""Duct (bc_y='wall') pressure-Poisson regression: manufactured solution.

Apply the SAME discrete 7-point Laplacian the solver inverts (periodic x; periodic
or Neumann-wall y; Neumann stretched z) to a random field p0, get div = L p0, then
solve and confirm recovery of p0 up to an additive constant — for both bc_y modes.
This is the primary correctness check for the DCT-in-y Poisson path.
"""
import os, sys
os.environ.setdefault("PYTORCH_JIT", "0")
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_default_dtype(torch.float64)
from utils import generate_grid
from projection_fft import initialize_fft_solver, solve_poisson_fft


def lap(p0, dx, dy, coeff_left, coeff_right, bc_y):
    nx, ny, nz = p0.shape
    xm = torch.roll(p0, 1, 0); xp = torch.roll(p0, -1, 0)
    Lx = (xp - 2 * p0 + xm) / dx**2
    if bc_y == 'wall':
        ym = torch.cat([p0[:, :1, :], p0[:, :-1, :]], 1)
        yp = torch.cat([p0[:, 1:, :], p0[:, -1:, :]], 1)
    else:
        ym = torch.roll(p0, 1, 1); yp = torch.roll(p0, -1, 1)
    Ly = (yp - 2 * p0 + ym) / dy**2
    cl = coeff_left.view(1, 1, nz); cr = coeff_right.view(1, 1, nz)
    zm = torch.cat([p0[:, :, :1], p0[:, :, :-1]], 2)
    zp = torch.cat([p0[:, :, 1:], p0[:, :, -1:]], 2)
    Lz = cl * zm - (cl + cr) * p0 + cr * zp
    return Lx + Ly + Lz


def run(bc_y, nx=16, ny=24, nz=20, Lx=2.0, Ly=1.0, Lz=1.0, gamma=1.2):
    dx, dy = Lx / nx, Ly / ny
    z_f, z_c, dz_f, dz_c = generate_grid(gamma, nz, Lz, device='cpu')
    coeff_left = 1.0 / (dz_c[:-1] * dz_f)
    coeff_right = 1.0 / (dz_c[1:] * dz_f)
    fft = initialize_fft_solver(nx, ny, nz, dx, dy, dz_c, dz_f,
                                top_wall_bc_type='dirichlet', bc_y=bc_y)
    torch.manual_seed(0)
    p0 = torch.randn(nx, ny, nz); p0 = p0 - p0.mean()
    div = lap(p0, dx, dy, coeff_left, coeff_right, bc_y)
    p = solve_poisson_fft(div, fft)
    ps = p[1:nx+1, 1:ny+1, 1:nz+1]; ps = ps - ps.mean()
    return (ps - p0).abs().max().item()


if __name__ == "__main__":
    e_per = run('periodic')
    e_wall = run('wall')
    print(f"bc_y=periodic  max|p_solved - p0| = {e_per:.3e}")
    print(f"bc_y=wall      max|p_solved - p0| = {e_wall:.3e}")
    ok = max(e_per, e_wall) < 1e-9
    print("DUCT POISSON TEST PASSED" if ok else "DUCT POISSON TEST FAILED")
    sys.exit(0 if ok else 1)
