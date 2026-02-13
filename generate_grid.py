import torch
import yaml
import os
import argparse
from utils import generate_grid, plot_grid, save_grid_csv

# Set double precision for stability
torch.set_default_dtype(torch.float64)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate DNS grid and evaluate memory requirements')
    parser.add_argument('--Re', type=float, default=None,
                        help='Reynolds number (1/nu) to override config file value')
    args = parser.parse_args()

    # Load configuration
    config_file = 'config.yaml'
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Device setup
    device_config = config.get('compute', {}).get('device', 'auto')
    if device_config == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_config == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Extract grid parameters
    nx = config['grid']['nx']
    ny = config['grid']['ny']
    nz = config['grid']['nz']

    Lx = config['domain']['Lx']
    Ly = config['domain']['Ly']
    Lz = config['domain']['Lz']

    gamma = config['flow']['gamma']

    # Reynolds number (allow command line override)
    Re_config = config['flow']['Re']
    if args.Re is not None:
        Re = args.Re
        print(f"\nUsing Reynolds number from command line: Re = {Re}")
    else:
        Re = Re_config

    nu = 1.0 / Re

    # Output folder
    output_config = config.get('output', {})
    results_folder = output_config.get('results_folder', 'results')
    os.makedirs(results_folder, exist_ok=True)

    print("="*80)
    print("GRID GENERATION")
    print("="*80)
    print(f"\nConfiguration file: {config_file}")
    print(f"Device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\nGrid dimensions:")
    print(f"  nx = {nx}")
    print(f"  ny = {ny}")
    print(f"  nz = {nz}")
    print(f"  Total points: {nx * ny * nz:,}")

    print(f"\nDomain size:")
    print(f"  Lx = {Lx}")
    print(f"  Ly = {Ly}")
    print(f"  Lz = {Lz}")

    print(f"\nFlow parameters:")
    print(f"  Re = {Re}")
    print(f"  nu = {nu:.6e}")
    if args.Re is not None:
        print(f"  (config file Re = {Re_config}, overridden by command line)")

    print(f"\nGrid spacing:")
    dx = Lx / nx
    dy = Ly / ny
    print(f"  dx = {dx:.6f}")
    print(f"  dy = {dy:.6f}")
    print(f"  dz (variable, stretching factor gamma = {gamma})")

    # Generate stretched grid in z-direction
    print(f"\nGenerating stretched grid in z-direction...")
    z_f, z_c, dz_f, dz_c = generate_grid(gamma, nz, Lz, device=device)

    print(f"  z_f range: [{z_f.min().item():.6f}, {z_f.max().item():.6f}]")
    print(f"  z_c range: [{z_c.min().item():.6f}, {z_c.max().item():.6f}]")
    print(f"  dz_f range: [{dz_f.min().item():.6e}, {dz_f.max().item():.6e}]")
    print(f"  dz_c range: [{dz_c.min().item():.6e}, {dz_c.max().item():.6e}]")
    print(f"  Min/Max dz ratio: {dz_f.max().item() / dz_f.min().item():.2f}")

    # Compute memory estimates
    print(f"\nMemory estimates:")

    # Staggered grid sizes
    u_size = (nx + 1) * ny * nz
    v_size = nx * (ny + 1) * nz
    w_size = nx * ny * (nz + 1)
    p_size = nx * ny * nz

    bytes_per_element = 8  # float64

    total_elements = u_size + v_size + w_size + p_size
    memory_mb = (total_elements * bytes_per_element) / (1024**2)
    memory_gb = memory_mb / 1024

    print(f"  u field: {u_size:,} elements ({u_size * bytes_per_element / 1024**2:.2f} MB)")
    print(f"  v field: {v_size:,} elements ({v_size * bytes_per_element / 1024**2:.2f} MB)")
    print(f"  w field: {w_size:,} elements ({w_size * bytes_per_element / 1024**2:.2f} MB)")
    print(f"  p field: {p_size:,} elements ({p_size * bytes_per_element / 1024**2:.2f} MB)")
    print(f"  Total (u,v,w,p): {memory_mb:.2f} MB ({memory_gb:.3f} GB)")
    print(f"  Estimated total with workspace: {memory_gb * 3:.3f} GB (approximate)")

    # Save grid files
    print(f"\nSaving grid files to: {results_folder}/")
    save_grid_csv(z_f, z_c, dz_f, dz_c, nz, results_folder)
    print(f"  - grid.csv (saved)")

    plot_grid(z_f, z_c, results_folder)
    print(f"  - grid_visualization.png (saved)")

    print("\n" + "="*80)
    print("Grid generation complete!")
    print("="*80)

if __name__ == "__main__":
    main()
