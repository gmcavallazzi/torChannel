"""torChannel — GPU-accelerated DNS for incompressible turbulent channel flow.

The solver is a PyTorch program: every field is a ``torch.Tensor`` and every
operator is a tensor op, so a simulation can be driven step-by-step from Python
rather than only as a batch job.

Typical use::

    from torchannel import ChannelFlow
    ChannelFlow(config_file="examples/re180_open/config.yaml").run_simulation()

``ChannelFlow`` is imported lazily so that ``import torchannel`` stays cheap and
does not pull in CUDA initialisation for callers that only want, say,
:func:`torchannel.utils.generate_grid`.
"""

__version__ = "0.1.0"

__all__ = ["ChannelFlow", "__version__"]


def __getattr__(name):
    # PEP 562 lazy attribute access: keeps `import torchannel` free of the
    # solver's (heavy) import chain until ChannelFlow is actually requested.
    if name == "ChannelFlow":
        from torchannel.solver import ChannelFlow

        return ChannelFlow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
