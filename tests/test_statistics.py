#!/usr/bin/env python
"""
Test script for computing turbulence statistics from a single snapshot.

Usage:
    python test_statistics.py <field_file> [options]

Arguments:
    field_file: Path to .npz field file (required)

Options:
    --config CONFIG_FILE     Path to config.yaml (default: config.yaml)
    --output OUTPUT_FILE     Path to save statistics (default: <field_file>_stats.npz in same directory)
    --Re REYNOLDS_NUMBER     Reynolds number (overrides config, 1/nu)
    --nu VISCOSITY           Kinematic viscosity (overrides config, alternative to --Re)
    --Re_tau RE_TAU          Friction Reynolds number (overrides config)

Examples:
    python test_statistics.py results/fields_final.npz --Re 10000
    python test_statistics.py results/fields_final.npz --Re 5000 --Re_tau 180
    python test_statistics.py results/fields_final.npz --nu 0.0001 --output custom_stats.npz

Note:
    By default, statistics are saved in the same directory as the field file with the name
    <field_filename>_stats.npz (e.g., results/fields_final_stats.npz)
"""

import sys
import os
import argparse
import torch
from statistics import compute_statistics_from_snapshot

# Set double precision for consistency
torch.set_default_dtype(torch.float64)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Compute turbulence statistics from a single snapshot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('field_file', help='Path to .npz field file')
    parser.add_argument('--config', default='config.yaml', help='Path to config.yaml (default: config.yaml)')
    parser.add_argument('--output', default=None, help='Path to save statistics (default: derived from field_file)')
    parser.add_argument('--Re', type=float, help='Reynolds number (1/nu, overrides config)')
    parser.add_argument('--nu', type=float, help='Kinematic viscosity (overrides config, alternative to --Re)')
    parser.add_argument('--Re_tau', type=float, help='Friction Reynolds number (overrides config)')

    args = parser.parse_args()

    # Determine output file path
    if args.output is None:
        # Default: save in same directory as field file
        field_dir = os.path.dirname(args.field_file)
        field_base = os.path.splitext(os.path.basename(args.field_file))[0]

        if field_dir:
            output_file = os.path.join(field_dir, f"{field_base}_stats.npz")
        else:
            output_file = f"{field_base}_stats.npz"
    else:
        output_file = args.output

    # Validate Re/nu options
    if args.Re is not None and args.nu is not None:
        print("Error: Cannot specify both --Re and --nu")
        return 1

    # Prepare override dictionary
    overrides = {}
    if args.Re is not None:
        overrides['Re'] = args.Re
    elif args.nu is not None:
        overrides['Re'] = 1.0 / args.nu

    if args.Re_tau is not None:
        overrides['Re_tau'] = args.Re_tau

    # Compute statistics
    try:
        stats = compute_statistics_from_snapshot(
            args.field_file,
            args.config,
            output_file,
            overrides=overrides
        )
        print("\nTest completed successfully!")
        return 0
    except Exception as e:
        print(f"\nError during computation: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
