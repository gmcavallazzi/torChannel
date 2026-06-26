"""Inflow/outflow Poisson regression: bc_x='wall' (Neumann pressure in x, via DCT) +
bc_y='wall' (duct). Manufactured solution: apply the discrete 7-point Laplacian with
Neumann in x,y and Neumann stretched z, solve, recover the field up to a constant."""
import os, sys
os.environ.setdefault("PYTORCH_JIT", "0")
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_default_dtype(torch.float64)
from utils import generate_grid
from projection_fft import initialize_fft_solver, solve_poisson_fft


def lap(p0, dx, dy, cl, cr):
    nx, ny, nz = p0.shape
    xm = torch.cat([p0[:1], p0[:-1]], 0); xp = torch.cat([p0[1:], p0[-1:]], 0)
    Lx = (xp - 2*p0 + xm) / dx**2
    ym = torch.cat([p0[:, :1], p0[:, :-1]], 1); yp = torch.cat([p0[:, 1:], p0[:, -1:]], 1)
    Ly = (yp - 2*p0 + ym) / dy**2
    cl = cl.view(1, 1, nz); cr = cr.view(1, 1, nz)
    zm = torch.cat([p0[:, :, :1], p0[:, :, :-1]], 2); zp = torch.cat([p0[:, :, 1:], p0[:, :, -1:]], 2)
    Lz = cl*zm - (cl+cr)*p0 + cr*zp
    return Lx + Ly + Lz


def run():
    nx, ny, nz = 20, 24, 18
    Lx, Ly, Lz, gamma = 3.0, 1.0, 1.0, 1.2
    dx, dy = Lx/nx, Ly/ny
    z_f, z_c, dz_f, dz_c = generate_grid(gamma, nz, Lz, device='cpu')
    cl = 1.0/(dz_c[:-1]*dz_f); cr = 1.0/(dz_c[1:]*dz_f)
    fft = initialize_fft_solver(nx, ny, nz, dx, dy, dz_c, dz_f,
                                top_wall_bc_type='dirichlet', bc_y='wall', bc_x='wall')
    torch.manual_seed(0)
    p0 = torch.randn(nx, ny, nz); p0 = p0 - p0.mean()
    div = lap(p0, dx, dy, cl, cr)
    p = solve_poisson_fft(div, fft)
    ps = p[1:nx+1, 1:ny+1, 1:nz+1]; ps = ps - ps.mean()
    return (ps - p0).abs().max().item()


if __name__ == "__main__":
    err = run()
    print(f"bc_x=wall, bc_y=wall  max|p_solved - p0| = {err:.3e}")
    print("INFLOW/OUTFLOW POISSON TEST PASSED" if err < 1e-9 else "FAILED")
    sys.exit(0 if err < 1e-9 else 1)
