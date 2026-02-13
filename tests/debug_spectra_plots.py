
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import load_flow_fields

# LaTeX formatting
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'font.size': 10
})

def compute_spectra_variations(field_file, output_prefix='spectra_debug'):
    print(f"Loading field from {field_file}...")
    device = torch.device('cpu') # Use CPU for safety/simplicity
    fields = load_flow_fields(field_file, device=device)
    
    u = fields['u']
    v = fields['v']
    w = fields['w']
    z_c = fields['z_c']
    Lx = fields['Lx']
    Ly = fields['Ly']
    
    nx = u.shape[0] - 1
    ny = u.shape[1] - 2
    nz = w.shape[2] - 1
    Lz = z_c[-1].item()
    
    # Target z+ = 15
    nu = 1.0 / 10000.0 # Approx
    Re_tau_target = 535.0 # Approx
    z_plus_target = 15.0
    delta = Lz / 2.0
    u_tau_target = Re_tau_target * nu / delta
    z_target_phys = z_plus_target * nu / u_tau_target
    
    z_c_interior = z_c[1:nz+1]
    idx_bot = torch.argmin(torch.abs(z_c_interior - z_target_phys)).item()
    k_bot = idx_bot + 1
    
    print(f"Using bottom wall at k={k_bot}, z={z_c[k_bot]:.6f}")
    
    # Extract fluctuations (simplified, assuming mean flow U(z) is roughly correct)
    # Just take the plane and subtract mean
    u_plane = u[0:nx+1, 1:ny+1, k_bot]
    # Handle staggered grid for u: average to cell centers in x
    u_plane = 0.5 * (u_plane[:-1, :] + u_plane[1:, :])
    u_plane = u_plane - torch.mean(u_plane)
    
    # v: staggered in y
    v_plane = v[1:nx+1, 0:ny+1, k_bot]
    v_plane = 0.5 * (v_plane[:, :-1] + v_plane[:, 1:])
    v_plane = v_plane - torch.mean(v_plane)
    
    # w: centered in x,y
    w_plane = w[1:nx+1, 1:ny+1, k_bot]
    w_plane = w_plane - torch.mean(w_plane)
    
    # Compute 2D FFT
    u_fft = torch.fft.rfft2(u_plane)
    v_fft = torch.fft.rfft2(v_plane)
    w_fft = torch.fft.rfft2(w_plane)
    
    # Energy
    E_uu = torch.abs(u_fft)**2 / (nx*ny)**2
    E_vv = torch.abs(v_fft)**2 / (nx*ny)**2
    E_ww = torch.abs(w_fft)**2 / (nx*ny)**2
    E_uw = (u_fft * torch.conj(w_fft)).real / (nx*ny)**2
    
    # Wavenumbers
    kx = torch.arange(nx, dtype=torch.float32).view(-1, 1) * (2 * np.pi / Lx)
    ky = torch.arange(ny//2+1, dtype=torch.float32).view(1, -1) * (2 * np.pi / Ly)
    
    # Premultiply
    premult_uu = kx * ky * E_uu
    
    nkx = nx // 2
    nky = ny // 2
    
    # 1. Original Logic (Incorrect folding)
    # premult_uu_sym = premult_uu[1:nkx+1, 1:nky+1] + torch.flip(premult_uu[1:nkx+1, 1:nky+1], dims=[0])
    slice_orig = premult_uu[1:nkx+1, 1:nky+1]
    slice_flip = torch.flip(premult_uu[1:nkx+1, 1:nky+1], dims=[0])
    spec_original = slice_orig + slice_flip
    
    # 2. Current Fix (Summing positive and negative kx)
    E_pos = premult_uu[1:nkx+1, 1:nky+1]
    E_neg = premult_uu[nx-nkx:nx, 1:nky+1]
    E_neg_flipped = torch.flip(E_neg, dims=[0])
    spec_fixed = E_pos + E_neg_flipped
    
    # 3. Full Shifted Spectrum (No folding, centered at k=0)
    # We need full FFT for this to be easiest
    u_fft_full = torch.fft.fft2(u_plane)
    E_uu_full = torch.abs(u_fft_full)**2 / (nx*ny)**2
    
    # Construct full kx, ky grids
    kx_full = torch.fft.fftfreq(nx, d=Lx/(2*np.pi*nx)) * (2*np.pi) # -pi/dx to pi/dx
    ky_full = torch.fft.fftfreq(ny, d=Ly/(2*np.pi*ny)) * (2*np.pi)
    KX, KY = torch.meshgrid(kx_full, ky_full, indexing='ij')
    
    # Premultiply (using abs(k) because k can be negative)
    # Standard definition: kx * ky * E
    # But for visualization of "premultiplied", we usually plot kx*ky*E vs log(kx), log(ky).
    # Here we just want to see the map.
    premult_full = torch.abs(KX * KY) * E_uu_full
    
    # Shift
    spec_shifted = torch.fft.fftshift(premult_full)
    KX_shifted = torch.fft.fftshift(KX)
    KY_shifted = torch.fft.fftshift(KY)
    
    # Plotting
    def plot_spec(data, title, filename, extent=None, log=True):
        plt.figure(figsize=(6, 5))
        if log:
            # Log scale for value
            vmax = data.max()
            vmin = vmax * 1e-4
            plt.imshow(data.T, origin='lower', cmap='hot', norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), extent=extent)
        else:
            plt.imshow(data.T, origin='lower', cmap='hot', extent=extent)
        plt.colorbar()
        plt.title(title)
        plt.xlabel(r'$k_x$ (index)' if extent is None else r'$k_x$')
        plt.ylabel(r'$k_y$ (index)' if extent is None else r'$k_y$')
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"Saved {filename}")

    # Plot 1: Original Logic
    plot_spec(spec_original.numpy(), r'Original Logic (Bottom Wall Only)', f'{output_prefix}_original.png')
    
    # Plot 2: Current Fix
    plot_spec(spec_fixed.numpy(), r'Current Fix (Bottom Wall Only)', f'{output_prefix}_fixed.png')
    
    # Plot 3: Full Shifted
    # Extent for shifted: min kx to max kx
    extent = [KX_shifted.min().item(), KX_shifted.max().item(), KY_shifted.min().item(), KY_shifted.max().item()]
    plot_spec(spec_shifted.numpy(), r'Full Shifted Spectrum (No Folding)', f'{output_prefix}_shifted.png', extent=extent)

    # 5. Averaged Spectra (Top + Bottom, No Flips)
    print("Computing averaged spectra (no flips)...")
    
    # Extract top wall
    idx_top = torch.argmin(torch.abs(z_c_interior - (Lz - z_target_phys))).item()
    k_top = idx_top + 1
    print(f"Using top wall at k={k_top}, z={z_c[k_top]:.6f}")
    
    u_top = u[0:nx+1, 1:ny+1, k_top]
    u_top = 0.5 * (u_top[:-1, :] + u_top[1:, :])
    u_top = u_top - torch.mean(u_top)
    
    # Compute Top Spectrum
    u_fft_top = torch.fft.rfft2(u_top)
    E_uu_top = torch.abs(u_fft_top)**2 / (nx*ny)**2
    premult_uu_top = kx * ky * E_uu_top
    
    # Average Spectra
    premult_uu_avg = 0.5 * (premult_uu + premult_uu_top)
    
    # Apply folding fix to the averaged spectrum
    E_pos_avg = premult_uu_avg[1:nkx+1, 1:nky+1]
    E_neg_avg = premult_uu_avg[nx-nkx:nx, 1:nky+1]
    E_neg_flipped_avg = torch.flip(E_neg_avg, dims=[0])
    spec_averaged = E_pos_avg + E_neg_flipped_avg
    
    plot_spec(spec_averaged.numpy(), r'Averaged Spectra (Bot+Top, No Flips)', f'{output_prefix}_averaged.png')


    # 4. Wavelength Plot (Lambda_x, Lambda_y) - Log-Log
    print("Generating wavelength plot...")
    
    # Calculate wavelengths
    # kx indices 1 to nkx correspond to wavenumbers:
    # kx_vals = 1 * (2pi/Lx), 2 * (2pi/Lx), ...
    # lambda_x = 2pi / kx_vals = Lx / index
    
    kx_idx = np.arange(1, nkx + 1)
    ky_idx = np.arange(1, nky + 1)
    
    lambda_x = Lx / kx_idx
    lambda_y = Ly / ky_idx
    
    LAMBDA_X, LAMBDA_Y = np.meshgrid(lambda_x, lambda_y, indexing='ij')
    
    # Use the fixed spectrum
    data = spec_fixed.numpy()
    
    plt.figure(figsize=(7, 6))
    
    # Levels
    vmin = np.maximum(data[data > 0].min(), 1e-10) if np.any(data > 0) else 1e-10
    vmax = data.max()
    levels = np.logspace(np.log10(vmin), np.log10(vmax), 15)
    
    # 6. Final Check: All Components vs Lambda (Averaged, No Flips)
    print("Generating final check plots (All components vs Lambda)...")
    
    # We already have u_bot, v_bot, w_bot and u_top, v_top (computed above, but let's ensure v, w are there)
    # Re-extract to be safe and clear
    
    # Bottom
    u_bot = u[0:nx+1, 1:ny+1, k_bot]
    u_bot = 0.5 * (u_bot[:-1, :] + u_bot[1:, :]) - torch.mean(u_bot)
    
    v_bot = v[1:nx+1, 0:ny+1, k_bot]
    v_bot = 0.5 * (v_bot[:, :-1] + v_bot[:, 1:]) - torch.mean(v_bot)
    
    w_bot = w[1:nx+1, 1:ny+1, k_bot]
    w_bot = w_bot - torch.mean(w_bot)
    
    # Top
    u_top = u[0:nx+1, 1:ny+1, k_top]
    u_top = 0.5 * (u_top[:-1, :] + u_top[1:, :]) - torch.mean(u_top)
    
    v_top = v[1:nx+1, 0:ny+1, k_top]
    v_top = 0.5 * (v_top[:, :-1] + v_top[:, 1:]) - torch.mean(v_top)
    
    w_top = w[1:nx+1, 1:ny+1, k_top]
    w_top = w_top - torch.mean(w_top)
    
    # FFTs
    u_fft_bot = torch.fft.rfft2(u_bot)
    v_fft_bot = torch.fft.rfft2(v_bot)
    w_fft_bot = torch.fft.rfft2(w_bot)
    
    u_fft_top = torch.fft.rfft2(u_top)
    v_fft_top = torch.fft.rfft2(v_top)
    w_fft_top = torch.fft.rfft2(-w_top) # Negate w at top wall
    
    # Energy Spectra
    def compute_E(uf, vf, wf):
        E_uu = torch.abs(uf)**2 / (nx*ny)**2
        E_vv = torch.abs(vf)**2 / (nx*ny)**2
        E_ww = torch.abs(wf)**2 / (nx*ny)**2
        E_uw = (uf * torch.conj(wf)).real / (nx*ny)**2
        return E_uu, E_vv, E_ww, E_uw

    E_uu_bot, E_vv_bot, E_ww_bot, E_uw_bot = compute_E(u_fft_bot, v_fft_bot, w_fft_bot)
    E_uu_top, E_vv_top, E_ww_top, E_uw_top = compute_E(u_fft_top, v_fft_top, w_fft_top)
    
    # Average Spectra (No Flips)
    E_uu = 0.5 * (E_uu_bot + E_uu_top)
    E_vv = 0.5 * (E_vv_bot + E_vv_top)
    E_ww = 0.5 * (E_ww_bot + E_ww_top)
    E_uw = 0.5 * (E_uw_bot + E_uw_top)
    
    # Premultiply
    premult_uu = kx * ky * E_uu
    premult_vv = kx * ky * E_vv
    premult_ww = kx * ky * E_ww
    premult_uw = kx * ky * E_uw
    
    # Fold
    def fold(E):
        E_pos = E[1:nkx+1, 1:nky+1]
        E_neg = E[nx-nkx:nx, 1:nky+1]
        E_neg_flipped = torch.flip(E_neg, dims=[0])
        return E_pos + E_neg_flipped

    uu_final = fold(premult_uu).numpy()
    vv_final = fold(premult_vv).numpy()
    ww_final = fold(premult_ww).numpy()
    uw_final = fold(premult_uw).numpy()
    
    # Plotting
    kx_idx = np.arange(1, nkx + 1)
    ky_idx = np.arange(1, nky + 1)
    lambda_x = Lx / kx_idx
    lambda_y = Ly / ky_idx
    LAMBDA_X, LAMBDA_Y = np.meshgrid(lambda_x, lambda_y, indexing='ij')
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    def plot_subplot(ax, data, title, cmap='hot'):
        # Levels
        if cmap == 'RdBu_r': # For uw
            vmax = np.abs(data).max()
            levels = np.linspace(-vmax, vmax, 20)
            im = ax.contourf(LAMBDA_X, LAMBDA_Y, data, levels=levels, cmap=cmap)
        else:
            vmin = np.maximum(data[data > 0].min(), 1e-10) if np.any(data > 0) else 1e-10
            vmax = data.max()
            levels = np.logspace(np.log10(vmin), np.log10(vmax), 15)
            im = ax.contourf(LAMBDA_X, LAMBDA_Y, data, levels=levels, norm=mpl.colors.LogNorm(), cmap=cmap)
            
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\lambda_x$')
        ax.set_ylabel(r'$\lambda_y$')
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    plot_subplot(axes[0, 0], uu_final, r'$k_x k_y E_{uu}$')
    plot_subplot(axes[0, 1], vv_final, r'$k_x k_y E_{vv}$')
    plot_subplot(axes[1, 0], ww_final, r'$k_x k_y E_{ww}$')
    plot_subplot(axes[1, 1], uw_final, r'$k_x k_y E_{uw}$', cmap='RdBu_r')
    
    plt.suptitle(r'Premultiplied Spectra (Averaged Bot+Top, No Flips)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_final_check.png', dpi=150)
    print(f"Saved {output_prefix}_final_check.png")

if __name__ == "__main__":
    import sys
    field_file = sys.argv[1] if len(sys.argv) > 1 else 'results/fields.npz'
    compute_spectra_variations(field_file)
