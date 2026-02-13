import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_vortices(n_vortices, Ly=0.8, filename='vortex_debug.png'):
    ny = 64
    y = torch.linspace(0, Ly, ny)
    
    # ky calculation from initflow.py
    ky = n_vortices * torch.pi / Ly
    
    # Streamfunction part in y
    psi_y = torch.sin(ky * y)
    
    plt.figure(figsize=(10, 4))
    plt.plot(y.numpy(), psi_y.numpy())
    plt.title(f'Streamfunction y-component for n_vortices={n_vortices}')
    plt.xlabel('y')
    plt.ylabel('sin(ky * y)')
    plt.grid(True)
    plt.savefig(filename)
    print(f"Saved {filename}")

if __name__ == "__main__":
    plot_vortices(2, filename='vortex_n2.png')
    plot_vortices(4, filename='vortex_n4.png')
