
import torch
import numpy as np
from solver import ChannelFlow
from utils import compute_divergence, compute_bulk_velocity
import os

# Set double precision for stability
torch.set_default_dtype(torch.float64)

def test_restart_divergence():
    print("Testing restart divergence issue...")
    
    # 1. Initialize a flow and save it (mocking a previous run)
    # We'll use a small grid for speed
    config_content = """
grid:
  nx: 32
  ny: 32
  nz: 32
domain:
  Lx: 6.28
  Ly: 3.14
  Lz: 2.0
flow:
  Re: 100.0
  Re_tau: 180.0
  U_bulk: 1.0
  gamma: 2.5
initialization:
  type: "parabolic"
  perturbation_intensity: 0.1
time:
  dt: 0.01
  n_steps: 1
  scheme: "AB2"
  CFL_target: 0.5
  dt_update_interval: 0
output:
  results_folder: "test_restart_repro"
  n_out: 1
  n_save: 1
statistics:
  enabled: false
compute:
  device: "cpu"
"""
    
    os.makedirs("test_restart_repro", exist_ok=True)
    with open("test_restart_repro/config_repro.yaml", "w") as f:
        f.write(config_content)
        
    # Run 1: Initialize and save
    print("\n--- Run 1: Initial Run ---")
    sim1 = ChannelFlow("test_restart_repro/config_repro.yaml")
    
    # Ensure initial field is divergence free
    div1 = compute_divergence(sim1.u, sim1.v, sim1.w, sim1.nx, sim1.ny, sim1.nz, sim1.dx, sim1.dy, sim1.dz_f)
    max_div1 = torch.max(torch.abs(div1)).item()
    print(f"Run 1 Initial max divergence: {max_div1:.6e}")
    
    # Check u_bulk
    u_bulk = compute_bulk_velocity(sim1.u, sim1.cell_vol_ratio, sim1.total_volume)
    print(f"Run 1 u_bulk: {u_bulk.item():.6f}")
    print(f"Run 1 u max: {torch.max(sim1.u).item():.6f}")
    
    # Save the field manually to simulate a checkpoint
    # We use the solver's internal save mechanism via step loop or manual save
    # Let's just save manually to be sure
    from utils import save_flow_fields, compute_u_tau
    u_tau = compute_u_tau(sim1.u, sim1.z_c, sim1.nu)
    save_flow_fields(sim1.u, sim1.v, sim1.w, sim1.p, sim1.z_c, sim1.z_f, sim1.Lx, sim1.Ly, 
                     0, 0.0, u_tau, 0.0, "test_restart_repro", "fields_repro.npz")
    
    # 2. Restart from the saved file
    print("\n--- Run 2: Restart Run ---")
    
    # Modify config to point to the saved file
    config_restart_content = config_content.replace(
        'type: "parabolic"', 
        'type: "parabolic"\n  field_file: "test_restart_repro/fields_repro.npz"'
    )
    
    with open("test_restart_repro/config_restart.yaml", "w") as f:
        f.write(config_restart_content)
        
    sim2 = ChannelFlow("test_restart_repro/config_restart.yaml")
    
    # Check divergence immediately after load (and potential rescaling)
    div2 = compute_divergence(sim2.u, sim2.v, sim2.w, sim2.nx, sim2.ny, sim2.nz, sim2.dx, sim2.dy, sim2.dz_f)
    max_div2 = torch.max(torch.abs(div2)).item()
    print(f"Run 2 (Restart) Initial max divergence: {max_div2:.6e}")
    
    if max_div2 > 1e-8 and max_div2 > max_div1 * 100:
        print("\nFAILURE: Divergence increased significantly after restart!")
        print(f"  Original: {max_div1:.6e}")
        print(f"  Restart:  {max_div2:.6e}")
    else:
        print("\nSUCCESS: Divergence remained low after restart.")

if __name__ == "__main__":
    test_restart_divergence()
