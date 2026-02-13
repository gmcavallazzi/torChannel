import numpy as np
import matplotlib.pyplot as plt

def compute_dUdz_old(U_mean, z_c):
    """Old 1st order centered difference implementation."""
    nz = len(U_mean)
    dUdz = np.zeros(nz)
    dz_f = z_c[1:] - z_c[:-1]
    
    if nz > 2:
        for k in range(1, nz-1):
            dz_avg = dz_f[k] + dz_f[k-1]
            dUdz[k] = (U_mean[k+1] - U_mean[k-1]) / dz_avg
            
    dUdz[0] = (U_mean[1] - U_mean[0]) / dz_f[0]
    dUdz[-1] = (U_mean[-1] - U_mean[-2]) / dz_f[-1]
    return dUdz

def compute_dUdz_new(U_mean, z_c):
    """New 2nd order implementation."""
    nz = len(U_mean)
    dUdz = np.zeros(nz)
    
    if nz > 2:
        h_minus = z_c[1:-1] - z_c[0:-2]
        h_plus = z_c[2:] - z_c[1:-1]
        
        c_minus = -h_plus / (h_minus * (h_minus + h_plus))
        c_center = (h_plus - h_minus) / (h_minus * h_plus)
        c_plus = h_minus / (h_plus * (h_minus + h_plus))
        
        dUdz[1:-1] = (c_minus * U_mean[0:-2] + 
                      c_center * U_mean[1:-1] + 
                      c_plus * U_mean[2:])

    h0 = z_c[1] - z_c[0]
    h1 = z_c[2] - z_c[1]
    c0 = -(2*h0 + h1) / (h0 * (h0 + h1))
    c1 = (h0 + h1) / (h0 * h1)
    c2 = -h0 / (h1 * (h0 + h1))
    dUdz[0] = c0 * U_mean[0] + c1 * U_mean[1] + c2 * U_mean[2]

    h_last = z_c[-1] - z_c[-2]
    h_prev = z_c[-2] - z_c[-3]
    c_last = (2*h_last + h_prev) / (h_last * (h_last + h_prev))
    c_prev = -(h_last + h_prev) / (h_last * h_prev)
    c_prev2 = h_last / (h_prev * (h_last + h_prev))
    dUdz[-1] = c_last * U_mean[-1] + c_prev * U_mean[-2] + c_prev2 * U_mean[-3]

    return dUdz

# Test setup
nz = 64
Lz = 2.0
gamma = 2.5 # Strong stretching

# Generate stretched grid
k = np.linspace(0, nz, nz+1)
xi = (2 * k / nz) - 1
z_f = 0.5 * Lz * (1 + np.tanh(gamma * xi) / np.tanh(gamma))
z_c = 0.5 * (z_f[:-1] + z_f[1:]) # Interior points

# Test function: u(z) = sin(pi * z)
# Analytical derivative: u'(z) = pi * cos(pi * z)
u = np.sin(np.pi * z_c)
du_exact = np.pi * np.cos(np.pi * z_c)

# Compute numerical derivatives
du_old = compute_dUdz_old(u, z_c)
du_new = compute_dUdz_new(u, z_c)

# Compute errors
err_old = np.abs(du_old - du_exact)
err_new = np.abs(du_new - du_exact)

print(f"Max error (Old): {err_old.max():.6e}")
print(f"Max error (New): {err_new.max():.6e}")
print(f"Mean error (Old): {err_old.mean():.6e}")
print(f"Mean error (New): {err_new.mean():.6e}")
print(f"Improvement factor (Max): {err_old.max()/err_new.max():.2f}x")

# Plot errors
plt.figure(figsize=(10, 6))
plt.semilogy(z_c, err_old, label='Old (1st Order)')
plt.semilogy(z_c, err_new, label='New (2nd Order)')
plt.xlabel('z')
plt.ylabel('Absolute Error')
plt.title('Derivative Approximation Error on Stretched Grid')
plt.legend()
plt.grid(True, which="both", ls="-")
plt.savefig('derivative_test.png')
print("Saved error plot to derivative_test.png")
