"""
Verify which optimizations are already implemented in the code.
"""

import sys
import re

def check_file_for_pattern(filename, pattern, description):
    """Check if a file contains a specific pattern."""
    try:
        with open(filename, 'r') as f:
            content = f.read()
            if re.search(pattern, content, re.MULTILINE):
                return f"✓ {description}"
            else:
                return f"✗ {description}"
    except FileNotFoundError:
        return f"✗ File not found: {filename}"

print("=" * 80)
print("OPTIMIZATION VERIFICATION REPORT")
print("=" * 80)

print("\n1. FUSED KERNELS (operators.py):")
print("   " + check_file_for_pattern("operators.py", r"def compute_momentum_rhs_fused_v2\(", 
                                       "compute_momentum_rhs_fused_v2() exists"))
print("   " + check_file_for_pattern("operators.py", r"def compute_momentum_rhs_fused_imex\(", 
                                       "compute_momentum_rhs_fused_imex() exists"))
print("   " + check_file_for_pattern("operators.py", r"@torch\.jit\.script\s+def compute_momentum_rhs_fused_v2", 
                                       "fused_v2 is JIT-compiled"))
print("   " + check_file_for_pattern("operators.py", r"@torch\.jit\.script\s+def compute_momentum_rhs_fused_imex", 
                                       "fused_imex is JIT-compiled"))

print("\n2. JIT COMPILATION (operators.py):")
print("   " + check_file_for_pattern("operators.py", r"@torch\.jit\.script\s+def diffusion_u", 
                                       "diffusion_u() is JIT-compiled"))
print("   " + check_file_for_pattern("operators.py", r"@torch\.jit\.script\s+def diffusion_v", 
                                       "diffusion_v() is JIT-compiled"))
print("   " + check_file_for_pattern("operators.py", r"@torch\.jit\.script\s+def diffusion_xy_u", 
                                       "diffusion_xy_u() is JIT-compiled"))
print("   " + check_file_for_pattern("operators.py", r"@torch\.jit\.script\s+def diffusion_xy_v", 
                                       "diffusion_xy_v() is JIT-compiled"))

print("\n3. JIT COMPILATION (utils.py):")
print("   " + check_file_for_pattern("utils.py", r"@torch\.jit\.script\s+def compute_divergence", 
                                       "compute_divergence() is JIT-compiled"))
print("   " + check_file_for_pattern("utils.py", r"@torch\.jit\.script\s+def compute_bulk_velocity", 
                                       "compute_bulk_velocity() is JIT-compiled"))

print("\n4. SOLVER INTEGRATION (solver.py):")
print("   " + check_file_for_pattern("solver.py", r"compute_momentum_rhs_fused_v2", 
                                       "AB2 uses fused_v2 kernel"))
print("   " + check_file_for_pattern("solver.py", r"compute_momentum_rhs_fused_imex", 
                                       "IMEX uses fused_imex kernel"))
print("   " + check_file_for_pattern("solver.py", r"if self\.device\.type == 'cuda'", 
                                       "GPU/CPU fallback logic"))

print("\n5. TEST FILES:")
print("   " + check_file_for_pattern("test_fused_kernels.py", r"def test_fused_kernel_v2", 
                                       "test_fused_kernels.py exists"))
print("   " + check_file_for_pattern("benchmark_jit.py", r"def benchmark_function", 
                                       "benchmark_jit.py exists"))

print("\n6. DOCUMENTATION:")
import os
docs = [
    ("OPTIMIZATION_SUMMARY.md", "Optimization summary"),
    ("OPTIMIZATION_RECOMMENDATIONS.md", "Future recommendations"),
]
for doc_file, desc in docs:
    if os.path.exists(doc_file):
        print(f"   ✓ {desc}")
    else:
        print(f"   ✗ {desc}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
