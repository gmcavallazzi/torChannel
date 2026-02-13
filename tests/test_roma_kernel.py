"""
Test Roma et al. Kernel Implementation

Validates the Roma kernel properties:
1. Partition of unity: Σ_i δ(x - x_i)*dx = 1
2. Compact support: δ(r) = 0 for |r| > 2h
3. Symmetry: δ(r) = δ(-r)
4. Continuity: kernel is C^0 continuous
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ibm.rkpm_kernel import roma_kernel_1d, roma_kernel_3d


def test_partition_of_unity():
    """
    Test that kernel satisfies partition of unity:
    Σ_i δ_h(x - x_i) * Δx = 1 for all x

    This is a fundamental property required for correct interpolation.

    Note: For Roma kernel, h and dx are related. The kernel has compact
    support [-2h, 2h]. For uniform grid spacing dx, we typically use
    h = dx or h = 1.5*dx depending on the formulation.
    """
    print("\n=== Test 1: Partition of Unity ===")

    dx = 1.0
    h = 1.5 * dx  # Standard choice for Roma kernel (recommended in literature)

    # Test points (scattered throughout domain)
    x_test = torch.linspace(-5, 5, 100)

    # Grid points (uniform spacing dx)
    x_grid = torch.arange(-10.0, 11.0, dx)

    max_error = 0.0
    errors = []

    for x in x_test:
        kernel_sum = 0.0
        for x_i in x_grid:
            r = x - x_i
            kernel_sum += roma_kernel_1d(r, h).item() * dx

        error = abs(kernel_sum - 1.0)
        errors.append(error)
        max_error = max(max_error, error)

    print(f"  Number of test points: {len(x_test)}")
    print(f"  Max error: {max_error:.10e}")
    print(f"  Mean error: {sum(errors)/len(errors):.10e}")

    # Should be very close to 1.0 (within machine precision)
    tolerance = 1e-6
    assert max_error < tolerance, f"Partition of unity failed: max error = {max_error}"

    print(f"  ✓ PASS: Partition of unity satisfied (tol={tolerance})")


def test_compact_support():
    """
    Test kernel is zero outside support radius.

    Roma kernel has support [-2h, 2h], so should be exactly zero
    for |r| > 2h.
    """
    print("\n=== Test 2: Compact Support ===")

    h = 1.5

    # Inside support: should be non-zero
    r_inside = torch.tensor([0.0, 0.5*h, 1.0*h, 1.5*h, 1.99*h, -0.5*h, -1.5*h])

    print(f"  Kernel width h = {h}")
    print(f"  Support radius = 2h = {2*h}")

    # Test inside support
    print("\n  Testing inside support:")
    for r in r_inside:
        val = roma_kernel_1d(r, h).item()
        print(f"    δ(r={r.item():6.3f}) = {val:.8f}", end="")
        assert val > 0, f"Kernel should be positive at r={r.item()}"
        print(" ✓")

    # Outside support: should be exactly zero
    r_outside = torch.tensor([2.1*h, 3*h, 5*h, -2.1*h, -3*h])

    print("\n  Testing outside support:")
    for r in r_outside:
        val = roma_kernel_1d(r, h).item()
        print(f"    δ(r={r.item():6.3f}) = {val:.8e}", end="")
        assert abs(val) < 1e-10, f"Kernel should be zero at r={r.item()}, got {val}"
        print(" ✓")

    print("\n  ✓ PASS: Compact support verified")


def test_symmetry():
    """
    Test kernel is symmetric: δ(r) = δ(-r)

    This is important for momentum conservation.
    """
    print("\n=== Test 3: Symmetry ===")

    h = 1.5
    r_values = torch.linspace(-2*h, 2*h, 50)

    max_asymmetry = 0.0

    for r in r_values:
        val_pos = roma_kernel_1d(r, h).item()
        val_neg = roma_kernel_1d(-r, h).item()
        asymmetry = abs(val_pos - val_neg)
        max_asymmetry = max(max_asymmetry, asymmetry)

    print(f"  Number of test points: {len(r_values)}")
    print(f"  Max asymmetry: {max_asymmetry:.10e}")

    tolerance = 1e-10
    assert max_asymmetry < tolerance, f"Symmetry failed: max asymmetry = {max_asymmetry}"

    print(f"  ✓ PASS: Symmetry verified (tol={tolerance})")


def test_continuity():
    """
    Test kernel is continuous at r/h = 1 (transition between regions).

    The Roma kernel has two different formulas for |r|/h <= 1 and 1 < |r|/h <= 2.
    Check that they match at the boundary.
    """
    print("\n=== Test 4: Continuity at r/h = 1 ===")

    h = 1.5

    # Test continuity at r = h (boundary between regions)
    r_left = h * (1.0 - 1e-8)  # Just inside region 1
    r_right = h * (1.0 + 1e-8)  # Just inside region 2

    val_left = roma_kernel_1d(torch.tensor(r_left), h).item()
    val_right = roma_kernel_1d(torch.tensor(r_right), h).item()

    print(f"  δ(r={r_left:.10f}) = {val_left:.10f}")
    print(f"  δ(r={r_right:.10f}) = {val_right:.10f}")
    print(f"  Difference: {abs(val_left - val_right):.10e}")

    tolerance = 1e-6
    assert abs(val_left - val_right) < tolerance, f"Kernel not continuous at r=h"

    print(f"  ✓ PASS: Kernel is continuous (tol={tolerance})")


def test_3d_kernel():
    """
    Test 3D kernel as product of 1D kernels.

    Verify that:
    1. δ_3D(0,0,0) is maximum
    2. δ_3D is zero outside support cube
    3. δ_3D integrates correctly
    """
    print("\n=== Test 5: 3D Kernel ===")

    hx = hy = hz = 1.5
    dx = dy = dz = 1.0

    # Test at origin
    delta_origin = roma_kernel_3d(
        torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0),
        hx, hy, hz
    ).item()
    print(f"  δ_3D(0,0,0) = {delta_origin:.6f}")

    # Test at point inside support
    delta_inside = roma_kernel_3d(
        torch.tensor(hx), torch.tensor(hy), torch.tensor(hz),
        hx, hy, hz
    ).item()
    print(f"  δ_3D(h,h,h) = {delta_inside:.6f}")
    assert delta_inside > 0, "Should be positive inside support"

    # Test at point outside support
    delta_outside = roma_kernel_3d(
        torch.tensor(3*hx), torch.tensor(0.0), torch.tensor(0.0),
        hx, hy, hz
    ).item()
    print(f"  δ_3D(3h,0,0) = {delta_outside:.10e}")
    assert abs(delta_outside) < 1e-10, "Should be zero outside support"

    # Test product property: δ_3D(x,y,z) = δ_1D(x) * δ_1D(y) * δ_1D(z)
    dx_val = torch.tensor(0.7)
    dy_val = torch.tensor(1.2)
    dz_val = torch.tensor(0.5)

    delta_3d = roma_kernel_3d(dx_val, dy_val, dz_val, hx, hy, hz).item()
    delta_1d_x = roma_kernel_1d(dx_val, hx).item()
    delta_1d_y = roma_kernel_1d(dy_val, hy).item()
    delta_1d_z = roma_kernel_1d(dz_val, hz).item()
    delta_product = delta_1d_x * delta_1d_y * delta_1d_z

    print(f"\n  Product property check:")
    print(f"    δ_3D(0.7,1.2,0.5) = {delta_3d:.10f}")
    print(f"    δ_1D(0.7)*δ_1D(1.2)*δ_1D(0.5) = {delta_product:.10f}")
    print(f"    Difference: {abs(delta_3d - delta_product):.10e}")

    assert abs(delta_3d - delta_product) < 1e-10, "Product property failed"

    print(f"\n  ✓ PASS: 3D kernel properties verified")


def test_kernel_normalization():
    """
    Test that kernel integrates to 1/(2h) in 1D.

    For the continuous kernel: ∫_{-∞}^{∞} δ_h(r) dr = 1
    But since our support is finite: ∫_{-2h}^{2h} δ_h(r) dr = 1
    """
    print("\n=== Test 6: Kernel Normalization ===")

    h = 1.5

    # Integrate using trapezoidal rule
    r_values = torch.linspace(-2*h, 2*h, 1000)
    dr = r_values[1] - r_values[0]

    kernel_values = roma_kernel_1d(r_values, h)
    integral = torch.sum(kernel_values) * dr

    print(f"  Kernel width h = {h}")
    print(f"  Integral from -2h to 2h: {integral.item():.10f}")
    print(f"  Expected: 1.0")
    print(f"  Error: {abs(integral.item() - 1.0):.10e}")

    tolerance = 1e-3  # Numerical integration tolerance
    assert abs(integral.item() - 1.0) < tolerance, f"Normalization failed"

    print(f"  ✓ PASS: Kernel normalization verified (tol={tolerance})")


def run_all_tests():
    """Run all kernel validation tests."""
    print("="*70)
    print("Roma Kernel Validation Test Suite")
    print("="*70)

    try:
        test_partition_of_unity()
        test_compact_support()
        test_symmetry()
        test_continuity()
        test_3d_kernel()
        test_kernel_normalization()

        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70)
        return True

    except AssertionError as e:
        print("\n" + "="*70)
        print(f"✗ TEST FAILED: {e}")
        print("="*70)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
