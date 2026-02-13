import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml
import torch
from ibm import IBM_RKPM

def visualize_cube():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Force obstacle type to cube for this test
    config['ibm']['obstacle_type'] = 'cube'
    
    # Mock grid data (needed for IBM init)
    # Match config.yaml domain
    Lx = config['domain']['Lx']
    Ly = config['domain']['Ly']
    Lz = config['domain']['Lz']
    
    nx = 96
    ny = 96
    nz = 128
    
    # Create a dummy grid
    z_c = np.linspace(0, Lz, nz)
    z_f = np.linspace(0, Lz, nz+1)
    dz_c = np.ones_like(z_c) * (Lz/nz)
    dz_f = np.ones_like(z_f) * (Lz/nz)
    
    grid_data = {
        'x_c': np.linspace(0, Lx, nx),
        'x_f': np.linspace(0, Lx, nx),
        'y_c': np.linspace(0, Ly, ny),
        'y_f': np.linspace(0, Ly, ny),
        'z_c': z_c,
        'z_f': z_f,
        'dz_c': dz_c,
        'dz_f': dz_f,
        'dx': Lx/nx,
        'dy': Ly/ny
    }
    
    # Initialize IBM (this will generate points)
    print("Initializing IBM to generate cube points...")
    ibm = IBM_RKPM(config, grid_data)
    
    x = ibm.x_lag
    y = ibm.y_lag
    z = ibm.z_lag
    dS = ibm.dS
    
    print(f"Generated {len(x)} Lagrangian points.")
    print(f"Total Surface Area: {np.sum(dS):.6f}")
    
    # Expected Area
    dims = config['ibm']['cube']['dimensions']
    dx, dy, dz_cube = dims
    expected_area = 2 * (dx*dy + dx*dz_cube + dy*dz_cube)
    print(f"Expected Surface Area: {expected_area:.6f}")
    
    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot
    sc = ax.scatter(x, y, z, c=dS, cmap='viridis', s=5)
    plt.colorbar(sc, label='Area Element dS')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Lagrangian Points for Cube (N={len(x)})')
    
    # Set equal aspect ratio
    # set_box_aspect is available in recent matplotlib
    try:
        ax.set_box_aspect([1,1,1])
    except:
        pass
        
    output_file = 'visualize_cube.png'
    plt.savefig(output_file, dpi=150)
    print(f"Saved visualization to {output_file}")

if __name__ == "__main__":
    visualize_cube()
