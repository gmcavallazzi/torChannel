import torch
import sys
sys.path.append('/Users/giorgio.cavallazzi/Library/CloudStorage/OneDrive-City,UniversityofLondon/python_DNS_playground/DNS_homemade')

# Original loop-based implementations
def compute_divergence_loop(u, v, w, nx, ny, nz, dx, dy, dz_f):
    div = torch.zeros(nx, ny, nz)
    
    for i in range(1, nx+1):
        for j in range(1, ny+1):
            for k in range(1, nz+1):
                du_dx = (u[i, j, k] - u[i-1, j, k]) / dx
                dv_dy = (v[i, j, k] - v[i, j-1, k]) / dy
                dw_dz = (w[i, j, k] - w[i, j, k-1]) / dz_f[k-1]
                div[i-1, j-1, k-1] = du_dx + dv_dy + dw_dz
    
    return div

def project_velocity_loop(u, v, w, p, nx, ny, nz, dx, dy, dz_c, dz_f, dt):
    u_out = u.clone()
    v_out = v.clone()
    w_out = w.clone()
    
    # Correct u (x-faces)
    for i in range(1, nx+1):
        for j in range(1, ny+1):
            for k in range(1, nz+1):
                dp_dx = (p[i+1, j, k] - p[i, j, k]) / dx
                u_out[i, j, k] -= dt * dp_dx

    # Correct v (y-faces)
    for i in range(1, nx+1):
        for j in range(1, ny+1):
            for k in range(1, nz+1):
                dp_dy = (p[i, j+1, k] - p[i, j, k]) / dy
                v_out[i, j, k] -= dt * dp_dy

    # Correct w (z-faces)
    for i in range(1, nx+1):
        for j in range(1, ny+1):
            for k in range(1, nz):
                dp_dz = (p[i, j, k+1] - p[i, j, k]) / dz_c[k]
                w_out[i, j, k] -= dt * dp_dz

    return u_out, v_out, w_out

# Vectorized implementations
def compute_divergence_vectorized(u, v, w, nx, ny, nz, dx, dy, dz_f):
    du_dx = (u[1:nx+1, 1:ny+1, 1:nz+1] - u[0:nx, 1:ny+1, 1:nz+1]) / dx
    dv_dy = (v[1:nx+1, 1:ny+1, 1:nz+1] - v[1:nx+1, 0:ny, 1:nz+1]) / dy
    dw_dz = (w[1:nx+1, 1:ny+1, 1:nz+1] - w[1:nx+1, 1:ny+1, 0:nz]) / dz_f[0:nz].view(1, 1, -1)
    
    div = du_dx + dv_dy + dw_dz
    return div

def project_velocity_vectorized(u, v, w, p, nx, ny, nz, dx, dy, dz_c, dz_f, dt):
    u_out = u.clone()
    v_out = v.clone()
    w_out = w.clone()
    
    # Vectorized correction for u (x-faces)
    dp_dx = (p[2:nx+2, 1:ny+1, 1:nz+1] - p[1:nx+1, 1:ny+1, 1:nz+1]) / dx
    u_out[1:nx+1, 1:ny+1, 1:nz+1] -= dt * dp_dx

    # Vectorized correction for v (y-faces)
    dp_dy = (p[1:nx+1, 2:ny+2, 1:nz+1] - p[1:nx+1, 1:ny+1, 1:nz+1]) / dy
    v_out[1:nx+1, 1:ny+1, 1:nz+1] -= dt * dp_dy

    # Vectorized correction for w (z-faces)
    dp_dz = (p[1:nx+1, 1:ny+1, 2:nz+1] - p[1:nx+1, 1:ny+1, 1:nz]) / dz_c[1:nz].view(1, 1, -1)
    w_out[1:nx+1, 1:ny+1, 1:nz] -= dt * dp_dz

    return u_out, v_out, w_out

# Test setup
torch.set_default_dtype(torch.float64)
print("="*80)
print("Testing Vectorization Implementation")
print("="*80)

nx, ny, nz = 8, 8, 16
dx, dy = 0.5, 0.5
dt = 0.001

# Create stretched grid in z
from utils import generate_grid
z_f, z_c, dz_f, dz_c = generate_grid(gamma=1.0, nz=nz, Lz=2.0)

# Create random test fields
torch.manual_seed(42)
u = torch.randn(nx+1, ny+2, nz+2)
v = torch.randn(nx+2, ny+1, nz+2)
w = torch.randn(nx+2, ny+2, nz+1)
p = torch.randn(nx+2, ny+2, nz+2)

print("\n1. Testing compute_divergence")
print("-" * 80)

div_loop = compute_divergence_loop(u, v, w, nx, ny, nz, dx, dy, dz_f)
div_vec = compute_divergence_vectorized(u, v, w, nx, ny, nz, dx, dy, dz_f)

diff_div = torch.abs(div_loop - div_vec)
max_diff_div = torch.max(diff_div)
mean_diff_div = torch.mean(diff_div)

print(f"  Loop result shape: {div_loop.shape}")
print(f"  Vectorized result shape: {div_vec.shape}")
print(f"  Max difference: {max_diff_div:.2e}")
print(f"  Mean difference: {mean_diff_div:.2e}")
print(f"  Loop result sample: {div_loop[0, 0, :3]}")
print(f"  Vectorized result sample: {div_vec[0, 0, :3]}")

if max_diff_div < 1e-12:
    print("  ✓ PASS: compute_divergence vectorization is correct")
else:
    print(f"  ✗ FAIL: Significant differences detected!")
    print(f"  Location of max diff: {torch.where(diff_div == max_diff_div)}")

print("\n2. Testing project_velocity")
print("-" * 80)

u_loop, v_loop, w_loop = project_velocity_loop(u.clone(), v.clone(), w.clone(), 
                                                 p, nx, ny, nz, dx, dy, dz_c, dz_f, dt)
u_vec, v_vec, w_vec = project_velocity_vectorized(u.clone(), v.clone(), w.clone(), 
                                                    p, nx, ny, nz, dx, dy, dz_c, dz_f, dt)

diff_u = torch.abs(u_loop - u_vec)
diff_v = torch.abs(v_loop - v_vec)
diff_w = torch.abs(w_loop - w_vec)

max_diff_u = torch.max(diff_u)
max_diff_v = torch.max(diff_v)
max_diff_w = torch.max(diff_w)

print(f"  u max difference: {max_diff_u:.2e}")
print(f"  v max difference: {max_diff_v:.2e}")
print(f"  w max difference: {max_diff_w:.2e}")

if max_diff_u < 1e-12 and max_diff_v < 1e-12 and max_diff_w < 1e-12:
    print("  ✓ PASS: project_velocity vectorization is correct")
else:
    print(f"  ✗ FAIL: Significant differences detected!")
    if max_diff_u > 1e-12:
        print(f"    u: Location of max diff: {torch.where(diff_u == max_diff_u)}")
        idx = torch.where(diff_u == max_diff_u)
        i, j, k = idx[0][0].item(), idx[1][0].item(), idx[2][0].item()
        print(f"    u[{i},{j},{k}]: loop={u_loop[i,j,k]:.6e}, vec={u_vec[i,j,k]:.6e}")
    if max_diff_v > 1e-12:
        print(f"    v: Location of max diff: {torch.where(diff_v == max_diff_v)}")
        idx = torch.where(diff_v == max_diff_v)
        i, j, k = idx[0][0].item(), idx[1][0].item(), idx[2][0].item()
        print(f"    v[{i},{j},{k}]: loop={v_loop[i,j,k]:.6e}, vec={v_vec[i,j,k]:.6e}")
    if max_diff_w > 1e-12:
        print(f"    w: Location of max diff: {torch.where(diff_w == max_diff_w)}")
        idx = torch.where(diff_w == max_diff_w)
        i, j, k = idx[0][0].item(), idx[1][0].item(), idx[2][0].item()
        print(f"    w[{i},{j},{k}]: loop={w_loop[i,j,k]:.6e}, vec={w_vec[i,j,k]:.6e}")

print("\n" + "="*80)
print("Test Complete")
print("="*80)
