"""Back-compat shim: `import tridiag` resolves to `torchannel.tridiag`.

The modules moved into the `torchannel` package so the project can be
pip-installed without claiming generic top-level names like `solver` or
`utils`. Rebinding sys.modules (rather than re-exporting) keeps a single
module object, so `isinstance` and module-level state stay consistent between
`import tridiag` and `import torchannel.tridiag`.

New code should import from `torchannel` directly.
"""
import sys

from torchannel import tridiag as _module

sys.modules[__name__] = _module
