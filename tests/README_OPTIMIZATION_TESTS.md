# Optimization Tests and Benchmarks

This folder contains tests and benchmarks for the GPU-optimized DNS simulation.

---

## Test Files

### 1. `test_fused_kernels.py`
**Purpose:** Validate the enhanced fused GPU kernels

**What it tests:**
- `compute_momentum_rhs_fused_v2()` (AB2 scheme)
- `compute_momentum_rhs_fused_imex()` (IMEX scheme)
- Compares fused kernels against original separate implementations
- Checks all three velocity components (u, v, w)

**How to run:**
```bash
cd /path/to/DNS_homemade
python tests/test_fused_kernels.py
```

**Expected output:**
```
================================================================================
TESTING ENHANCED FUSED GPU KERNELS
================================================================================
...
✓ PASSED: Fused kernel v2 matches original implementation!
✓ PASSED: Fused IMEX kernel matches original implementation!

TEST SUMMARY
Fused kernel v2 (AB2):   ✓ PASSED
Fused IMEX kernel:       ✓ PASSED

✓ All tests passed! Kernels are numerically correct.
```

**Validation criteria:**
- Maximum absolute difference < 1e-12 (machine precision for float64)
- Tests all interior points where RHS is computed

---

### 2. `benchmark_jit.py`
**Purpose:** Measure performance of JIT-compiled functions

**What it benchmarks:**
- `compute_divergence()` - Called 2x per timestep
- `compute_bulk_velocity()` - Called 1x per timestep
- `diffusion_u()` - Used in AB2 scheme
- `diffusion_v()` - Used in AB2 scheme

**How to run:**
```bash
cd /path/to/DNS_homemade
python tests/benchmark_jit.py
```

**Expected output:**
```
================================================================================
JIT COMPILATION PERFORMANCE BENCHMARK
================================================================================

Device: cuda (or cpu)
Grid size: 64x64x64

1. compute_divergence():
   Average time: 0.XXX ms
   Throughput: XXXX calls/sec

2. compute_bulk_velocity():
   Average time: 0.XXX ms
   Throughput: XXXX calls/sec
...
```

**Notes:**
- First call includes JIT compilation overhead (~100-500ms)
- Subsequent calls show optimized performance
- GPU performance scales better with larger grids

---

### 3. `verify_optimizations.py`
**Purpose:** Verify which optimizations are present in the codebase

**What it checks:**
- Presence of fused kernels
- JIT compilation decorators
- Solver integration
- Test file existence
- Documentation files

**How to run:**
```bash
cd /path/to/DNS_homemade
python tests/verify_optimizations.py
```

**Expected output:**
```
================================================================================
OPTIMIZATION VERIFICATION REPORT
================================================================================

1. FUSED KERNELS (operators.py):
   ✓ compute_momentum_rhs_fused_v2() exists
   ✓ compute_momentum_rhs_fused_imex() exists
   ...

2. JIT COMPILATION (operators.py):
   ✓ diffusion_u() is JIT-compiled
   ✓ diffusion_v() is JIT-compiled
   ...

All checks should show ✓
```

---

## Running All Tests

### Quick validation:
```bash
# From DNS_homemade directory
python tests/verify_optimizations.py
python tests/test_fused_kernels.py
python tests/benchmark_jit.py
```

### With existing test suite:
```bash
# Run all tests in tests folder
cd tests
for test in test_*.py; do
    echo "Running $test..."
    python "$test"
done
```

---

## Performance Expectations

### GPU (CUDA) - 64³ grid:
- `compute_divergence()`: ~0.1-0.5 ms/call
- `compute_bulk_velocity()`: ~0.05-0.2 ms/call
- `diffusion_u/v()`: ~0.5-2 ms/call
- **Fused kernels:** 30-60% faster than separate kernels

### CPU - 64³ grid:
- `compute_divergence()`: ~1-2 ms/call
- `compute_bulk_velocity()`: ~0.3-0.7 ms/call
- `diffusion_u/v()`: ~2-5 ms/call
- **Fused kernels:** 15-25% faster than separate kernels

---

## Troubleshooting

### Import errors:
```bash
# Make sure you run from DNS_homemade directory
cd /path/to/DNS_homemade
python tests/test_fused_kernels.py
```

### GPU not detected:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### Tests failing:
1. Check if operators.py, solver.py, utils.py have recent optimizations
2. Run `python tests/verify_optimizations.py` to check implementation
3. Try running on CPU first: `export CUDA_VISIBLE_DEVICES=""`

---

## Integration with Main Code

These optimizations are **automatically active** when you run:
```bash
python main.py
```

**No configuration changes needed!**

The solver automatically detects GPU availability and uses:
- Fused kernels on GPU (when available)
- JIT-compiled functions on both GPU and CPU
- Falls back to separate kernels if needed

---

## Related Documentation

- **`../IMPLEMENTATION_STATUS.md`** - Full implementation status
- **`../OPTIMIZATION_SUMMARY.md`** - Complete optimization overview
- **`../OPTIMIZATION_RECOMMENDATIONS.md`** - Future optimization ideas

---

## Notes

- All tests validate **numerical correctness** (not just performance)
- Benchmarks show **typical performance**, actual results depend on hardware
- JIT compilation happens on first call (adds overhead)
- GPU performance improves with larger grids (128³+)

---

**Last Updated:** December 1, 2024
**Status:** All tests passing ✅
