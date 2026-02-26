"""
cuSPARSE tridiagonal solver wrapper using ctypes.

Provides GPU-optimized tridiagonal solves via NVIDIA's cuSPARSE library
(gtsv2StridedBatch). Falls back gracefully if cuSPARSE is not available.

Two solver variants:
- solve_shared_coeffs(a, b, c, d): a,b,c are 1D (shared across batch), d is (batch, n)
- solve_per_system(a, b, c, d): all are (batch, n)
"""

import torch
import ctypes
import ctypes.util

# cuSPARSE status codes
CUSPARSE_STATUS_SUCCESS = 0

# Try to load cuSPARSE library
_cusparse = None
_cusparse_available = False


def _find_cusparse():
    """Find and load the cuSPARSE shared library."""
    global _cusparse, _cusparse_available

    if _cusparse is not None:
        return _cusparse_available

    # Try common library names
    for name in ['cusparse', 'libcusparse.so', 'libcusparse.dylib']:
        path = ctypes.util.find_library(name)
        if path is not None:
            try:
                _cusparse = ctypes.CDLL(path)
                _cusparse_available = True
                return True
            except OSError:
                continue

    # Try loading directly from common CUDA paths
    import os
    cuda_paths = [
        os.environ.get('CUDA_HOME', '/usr/local/cuda'),
        '/usr/local/cuda',
        '/usr/lib/x86_64-linux-gnu',
    ]
    for cuda_path in cuda_paths:
        for lib_name in ['lib64/libcusparse.so', 'lib/libcusparse.so',
                         'lib/libcusparse.dylib', 'libcusparse.so']:
            full_path = os.path.join(cuda_path, lib_name)
            if os.path.exists(full_path):
                try:
                    _cusparse = ctypes.CDLL(full_path)
                    _cusparse_available = True
                    return True
                except OSError:
                    continue

    _cusparse_available = False
    return False


def is_available():
    """Check if cuSPARSE is available."""
    if not torch.cuda.is_available():
        return False
    return _find_cusparse()


class CuSparseTridiagonalSolver:
    """
    GPU-optimized batched tridiagonal solver using cuSPARSE gtsv2StridedBatch.

    Usage:
        solver = CuSparseTridiagonalSolver()
        x = solver.solve_shared_coeffs(a, b, c, d)  # a,b,c: (n,), d: (batch, n)
        x = solver.solve_per_system(a, b, c, d)      # all: (batch, n)
    """

    def __init__(self):
        if not is_available():
            raise RuntimeError("cuSPARSE is not available")

        self._handle = ctypes.c_void_p()
        status = _cusparse.cusparseCreate(ctypes.byref(self._handle))
        if status != CUSPARSE_STATUS_SUCCESS:
            raise RuntimeError(f"cusparseCreate failed with status {status}")

        # Cache for workspace buffers (keyed by (batch_size, n, dtype))
        self._workspace_cache = {}
        # Cache for expanded coefficient buffers
        self._coeff_cache = {}

    def __del__(self):
        if hasattr(self, '_handle') and self._handle:
            try:
                _cusparse.cusparseDestroy(self._handle)
            except Exception:
                pass

    def _get_workspace(self, batch_size, n, dtype):
        """Get or allocate workspace buffer for gtsv2StridedBatch."""
        key = (batch_size, n, dtype)
        if key not in self._workspace_cache:
            # Query workspace size
            buf_size = ctypes.c_size_t(0)
            # Create temporary contiguous buffers for the query
            dummy = torch.zeros(batch_size * n, dtype=dtype, device='cuda')
            element_size = dummy.element_size()

            status = _cusparse.cusparseDgtsv2StridedBatch_bufferSizeExt(
                self._handle,
                ctypes.c_int(n),
                dummy.data_ptr(),  # dl
                dummy.data_ptr(),  # d
                dummy.data_ptr(),  # du
                dummy.data_ptr(),  # x
                ctypes.c_int(batch_size),
                ctypes.c_int(n),   # stride
                ctypes.byref(buf_size)
            )
            if status != CUSPARSE_STATUS_SUCCESS:
                raise RuntimeError(f"gtsv2StridedBatch_bufferSizeExt failed: {status}")

            workspace = torch.empty(buf_size.value, dtype=torch.uint8, device='cuda')
            self._workspace_cache[key] = workspace

        return self._workspace_cache[key]

    def _get_coeff_buffer(self, batch_size, n, name):
        """Get or allocate a buffer for expanding shared coefficients."""
        key = (batch_size, n, name)
        if key not in self._coeff_cache:
            self._coeff_cache[key] = torch.empty(batch_size * n, dtype=torch.float64, device='cuda')
        return self._coeff_cache[key]

    def solve_shared_coeffs(self, a, b, c, d):
        """
        Solve batched tridiagonal systems where a, b, c are shared across batch.

        Args:
            a: Lower diagonal (n,) - same for all systems
            b: Main diagonal (n,) - same for all systems
            c: Upper diagonal (n,) - same for all systems
            d: RHS (batch, n) - different for each system

        Returns:
            x: Solution (batch, n)
        """
        batch_size, n = d.shape

        # Expand shared coefficients to strided format: (batch * n,) with stride n
        # gtsv2StridedBatch expects interleaved layout:
        # [system0_elem0, system1_elem0, ..., system0_elem1, system1_elem1, ...]
        # But with stride=n, it's: [sys0_e0, sys0_e1, ..., sys0_en, sys1_e0, ...]
        # which is (batch, n) contiguous layout

        # Expand a, b, c from (n,) to (batch, n) contiguous
        dl = self._get_coeff_buffer(batch_size, n, 'dl')
        dm = self._get_coeff_buffer(batch_size, n, 'dm')
        du = self._get_coeff_buffer(batch_size, n, 'du')

        # Use expand + contiguous to broadcast
        dl.view(batch_size, n)[:] = a.unsqueeze(0)
        dm.view(batch_size, n)[:] = b.unsqueeze(0)
        du.view(batch_size, n)[:] = c.unsqueeze(0)

        # x is both input (RHS) and output (solution) for gtsv2StridedBatch
        x = d.contiguous().clone()

        workspace = self._get_workspace(batch_size, n, d.dtype)

        status = _cusparse.cusparseDgtsv2StridedBatch(
            self._handle,
            ctypes.c_int(n),
            dl.data_ptr(),
            dm.data_ptr(),
            du.data_ptr(),
            x.data_ptr(),
            ctypes.c_int(batch_size),
            ctypes.c_int(n),  # batchStride
            workspace.data_ptr()
        )
        if status != CUSPARSE_STATUS_SUCCESS:
            raise RuntimeError(f"cusparseDgtsv2StridedBatch failed: {status}")

        return x

    def solve_per_system(self, a, b, c, d):
        """
        Solve batched tridiagonal systems where a, b, c, d all have batch dimension.

        Args:
            a: Lower diagonal (batch, n)
            b: Main diagonal (batch, n)
            c: Upper diagonal (batch, n)
            d: RHS (batch, n)

        Returns:
            x: Solution (batch, n)
        """
        batch_size, n = d.shape

        # Ensure contiguous layout
        dl = a.contiguous()
        dm = b.contiguous()
        du = c.contiguous()
        x = d.contiguous().clone()

        workspace = self._get_workspace(batch_size, n, d.dtype)

        status = _cusparse.cusparseDgtsv2StridedBatch(
            self._handle,
            ctypes.c_int(n),
            dl.data_ptr(),
            dm.data_ptr(),
            du.data_ptr(),
            x.data_ptr(),
            ctypes.c_int(batch_size),
            ctypes.c_int(n),  # batchStride
            workspace.data_ptr()
        )
        if status != CUSPARSE_STATUS_SUCCESS:
            raise RuntimeError(f"cusparseDgtsv2StridedBatch failed: {status}")

        return x
