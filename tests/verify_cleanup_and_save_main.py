import os
import yaml
import shutil
import torch
from solver import ChannelFlow

def verify_cleanup_and_save():
    # 1. Setup test environment
    test_results_folder = 'test_results_main'
    os.makedirs(test_results_folder, exist_ok=True)
    
    # Create a dummy file to verify deletion
    with open(os.path.join(test_results_folder, 'dummy_file.txt'), 'w') as f:
        f.write('This should be deleted')
    
    # Create a test config
    config = {
        'grid': {'nx': 8, 'ny': 8, 'nz': 8},
        'domain': {'Lx': 1.0, 'Ly': 1.0, 'Lz': 1.0},
        'flow': {'Re': 100.0, 'Re_tau': 100.0, 'U_bulk': 1.0, 'gamma': 1.0},
        'initialization': {'type': 'random', 'perturbation_intensity': 0.0},
        'solver': {'type': 'fft'},
        'time': {
            'dt': 0.001,
            'n_steps': 10,
            'CFL_target': 0.5,
            'scheme': 'AB2'
        },
        'output': {
            'results_folder': test_results_folder,
            'n_out': 1,
            'n_save': 2 # Save every 2 steps
        },
        'compute': {'device': 'cpu'}
    }
    
    with open('test_config_main.yaml', 'w') as f:
        yaml.dump(config, f)
        
    print("Test environment set up.")
    print(f"Created dummy file in {test_results_folder}")
    
    # 2. Initialize solver
    print("Initializing solver...")
    solver = ChannelFlow('test_config_main.yaml')
    
    # 3. Verify cleanup
    files = os.listdir(test_results_folder)
    print(f"Files in {test_results_folder} after init: {files}")
    
    # Check if dummy file is gone
    if 'dummy_file.txt' in files:
        print("FAIL: dummy_file.txt was not deleted.")
    else:
        print("PASS: dummy_file.txt was deleted.")
        
    # Check if grid files were created (solver init does this)
    if 'grid.csv' in files and 'grid.png' in files:
        print("PASS: Grid files were created.")
    else:
        print("FAIL: Grid files were not created.")

    # Check if u_profile_init.png is GONE
    if 'u_profile_init.png' in files:
        print("FAIL: u_profile_init.png was created but should not be.")
    else:
        print("PASS: u_profile_init.png was NOT created.")

    # 4. Run simulation
    print("Running simulation...")
    solver.run_simulation()
    
    # 5. Verify periodic saving
    files_after = os.listdir(test_results_folder)
    print(f"Files in {test_results_folder} after run: {files_after}")
    
    # We expect fields_init.npz (step 0), fields.npz (latest), fields_final.npz, and timeseries.npz
    # Note: fields.npz is overwritten, so we just check it exists.
    expected_files = ['fields_init.npz', 'fields.npz', 'fields_final.npz', 'timeseries.npz']
    
    all_found = True
    for f in expected_files:
        if f not in files_after:
            print(f"FAIL: Expected file {f} not found.")
            all_found = False
        else:
            print(f"PASS: Found expected file {f}")
            
    if all_found:
        print("\nOVERALL: Verification SUCCESSFUL")
    else:
        print("\nOVERALL: Verification FAILED")

    # Cleanup
    # shutil.rmtree(test_results_folder)
    # os.remove('test_config_main.yaml')

if __name__ == "__main__":
    verify_cleanup_and_save()
