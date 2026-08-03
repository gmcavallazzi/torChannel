"""Console entry points: ``torchannel-run`` and ``torchannel-stats``.

These are thin wrappers so that an installed copy is usable from any directory,
without the historical requirement of running from the repository root.
"""

import argparse
import os
import sys


def run(argv=None):
    """Run a simulation from a YAML config (``torchannel-run``)."""
    p = argparse.ArgumentParser(
        prog="torchannel-run",
        description="Run a torChannel DNS from a YAML configuration file.",
    )
    p.add_argument("config", help="path to the YAML configuration file")
    args = p.parse_args(argv)

    if not os.path.exists(args.config):
        p.error(f"config file not found: {args.config}")

    import torch

    # Double precision is the default everywhere; see docs/NUMERICAL_METHODS.md.
    torch.set_default_dtype(torch.float64)

    from torchannel.solver import ChannelFlow

    ChannelFlow(config_file=args.config).run_simulation()
    return 0


def stats(argv=None):
    """Plot turbulence statistics (``torchannel-stats``).

    Delegates to the repository's ``plot_statistics.py``, which owns the CLI.
    """
    try:
        import plot_statistics
    except ImportError:
        sys.exit(
            "torchannel-stats requires plot_statistics.py, which ships with the "
            "repository rather than the installed package. Run it from a clone:\n"
            "    python plot_statistics.py <stats.npz> --config <config.yaml>"
        )
    return plot_statistics.main()


if __name__ == "__main__":
    sys.exit(run())
