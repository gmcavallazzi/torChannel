"""Back-compat shim: `import projection_fft` resolves to `torchannel.projection_fft`.

The modules moved into the `torchannel` package so the project can be
pip-installed without claiming generic top-level names like `solver` or
`utils`. Rebinding sys.modules (rather than re-exporting) keeps a single
module object, so `isinstance` and module-level state stay consistent between
`import projection_fft` and `import torchannel.projection_fft`.

New code should import from `torchannel` directly.
"""
import sys

from torchannel import projection_fft as _module

sys.modules[__name__] = _module
