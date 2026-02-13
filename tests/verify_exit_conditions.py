import os
import yaml
import shutil
from solver import ChannelFlow

def verify_exit_conditions():
    print("Verifying dual exit conditions...")
    
    # Common config
    base_config = {
        'grid': {'nx': 8, 'ny': 8, 'nz': 8},
        'domain': {'Lx': 1.0, 'Ly': 1.0, 'Lz': 1.0},
        'flow': {'Re': 100.0, 'Re_tau': 100.0, 'U_bulk': 1.0, 'gamma': 1.0},
        'initialization': {'type': 'random', 'perturbation_intensity': 0.0},
        'solver': {'type': 'fft'},
        'compute': {'device': 'cpu'},
        'output': {'results_folder': 'test_exit', 'n_out': 1, 'n_save': 100}
    }
    
    os.makedirs('test_exit', exist_ok=True)

    # Test 1: Exit by n_steps
    print("\nTest 1: Exit by n_steps (n_steps=5, t_max=100.0, dt=0.1)")
    config1 = base_config.copy()
    config1['time'] = {
        'dt': 0.1,
        'n_steps': 5,
        't_max': 100.0,
        'CFL_target': 0.5,
        'scheme': 'AB2',
        'dt_update_interval': 0 # Fixed dt
    }
    
    with open('test_config_exit1.yaml', 'w') as f:
        yaml.dump(config1, f)
        
    solver1 = ChannelFlow('test_config_exit1.yaml')
    solver1.run_simulation()
    
    print(f"Result 1: Steps={solver1.n_steps}, Time={solver1.time:.2f}")
    
    # Note: The loop runs 'while step < n_steps', so it runs exactly n_steps times.
    # Final time should be n_steps * dt = 5 * 0.1 = 0.5
    if abs(solver1.time - 0.5) < 1e-6:
        print("PASS: Test 1 stopped at correct time.")
    else:
        print(f"FAIL: Test 1 stopped at time {solver1.time}, expected 0.5")

    # Test 2: Exit by t_max
    print("\nTest 2: Exit by t_max (n_steps=100, t_max=0.3, dt=0.1)")
    config2 = base_config.copy()
    config2['time'] = {
        'dt': 0.1,
        'n_steps': 100,
        't_max': 0.3,
        'CFL_target': 0.5,
        'scheme': 'AB2',
        'dt_update_interval': 0 # Fixed dt
    }
    
    with open('test_config_exit2.yaml', 'w') as f:
        yaml.dump(config2, f)
        
    solver2 = ChannelFlow('test_config_exit2.yaml')
    solver2.run_simulation()
    
    print(f"Result 2: Time={solver2.time:.2f}")
    
    # Should run for 3 steps: 0.1, 0.2, 0.3.
    # At start of step 4, time is 0.3. Loop condition: time < t_max (0.3 < 0.3 is False).
    # So it should stop at 0.3.
    if abs(solver2.time - 0.3) < 1e-6:
        print("PASS: Test 2 stopped at correct time.")
    else:
        print(f"FAIL: Test 2 stopped at time {solver2.time}, expected 0.3")

    # Cleanup
    # shutil.rmtree('test_exit')
    # os.remove('test_config_exit1.yaml')
    # os.remove('test_config_exit2.yaml')

if __name__ == "__main__":
    verify_exit_conditions()
