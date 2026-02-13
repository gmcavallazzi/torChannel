import torch
import argparse
from solver import ChannelFlow

# Set double precision for stability
torch.set_default_dtype(torch.float64)

def main():
    parser = argparse.ArgumentParser(description='DNS Channel Flow Simulation')
    parser.add_argument('config', 
                       type=str, 
                       nargs='?',  # Makes it optional
                       default='config.yaml',
                       help='Path to configuration file (default: config.yaml)')
    args = parser.parse_args()
    
    solver = ChannelFlow(config_file=args.config)
    solver.run_simulation()

if __name__ == "__main__":
    main()