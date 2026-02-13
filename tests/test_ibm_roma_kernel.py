"""
Unit tests for Roma kernel and RKPM delta functions.

Tests the core properties of the regularized delta function:
- Partition of unity
- Compact support
- Symmetry
- Non-negativity
- Polynomial reproduction (for RKPM-corrected version)
"""

import torch
import numpy as np
import pytest
import sys
import os

# Add parent directory to path to import ibm module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ibm.rkpm_kernel import roma_kernel_1d, rkpm_delta_3d


class TestRomaKernel1D:
    """Tests for the 1D Roma/Peskin kernel."""

    def test_partition_of_unity_uniform_grid(self):
        """
        Verify that ∑ φ(x_i) dx = 1 for a uniform grid.

        This is the fundamental property of delta function approximations.
        """
        h = 1.0
        x = torch.linspace(-3*h, 3*h, 100)
        dx = x[1] - x[0]
        phi = roma_kernel_1d(x, h)
        integral = torch.sum(phi) * dx

        assert torch.abs(integral - 1.0) < 0.01, \
            f"Partition of unity failed: ∑φ dx = {integral:.6f}, expected 1.0"

    def test_compact_support(self):
        """
        Verify that φ(r) = 0 for |r| > 2h.

        The Roma kernel should have compact support [-2h, 2h].
        """
        h = 1.0

        # Test at exactly 2h (should be ~0 but may not be exactly due to numerics)
        phi_2h = roma_kernel_1d(torch.tensor([2.0*h]), h)
        assert phi_2h[0] < 1e-6, \
            f"Kernel not zero at 2h: φ(2h) = {phi_2h[0]:.2e}"

        # Test beyond 2h (should be exactly 0)
        phi_beyond = roma_kernel_1d(torch.tensor([2.5*h, -2.5*h, 3.0*h]), h)
        assert torch.all(phi_beyond == 0.0), \
            f"Kernel not zero beyond 2h: values = {phi_beyond}"

    def test_symmetry(self):
        """
        Verify that φ(r) = φ(-r) (even function).
        """
        h = 1.0
        r_test = torch.tensor([0.5*h, h, 1.5*h])
        phi_pos = roma_kernel_1d(r_test, h)
        phi_neg = roma_kernel_1d(-r_test, h)

        error = torch.max(torch.abs(phi_pos - phi_neg))
        assert error < 1e-10, \
            f"Symmetry violation: max|φ(r) - φ(-r)| = {error:.2e}"

    def test_non_negativity(self):
        """
        Verify that φ(r) >= 0 for all r.

        The kernel should be non-negative everywhere.
        """
        h = 1.0
        r = torch.linspace(-3*h, 3*h, 1000)
        phi = roma_kernel_1d(r, h)

        min_val = torch.min(phi)
        assert min_val >= -1e-10, \
            f"Kernel is negative: min(φ) = {min_val:.2e}"

    def test_maximum_at_origin(self):
        """
        Verify that the kernel is maximized at r=0.
        """
        h = 1.0
        r = torch.linspace(-2*h, 2*h, 200)
        phi = roma_kernel_1d(r, h)

        max_idx = torch.argmax(phi)
        max_location = r[max_idx]

        # Maximum should be at or very close to r=0
        assert torch.abs(max_location) < 0.05*h, \
            f"Maximum not at origin: argmax(φ) = {max_location:.4f}"

    def test_continuity(self):
        """
        Verify that the kernel is continuous at the transition point r=h.

        The Roma kernel should be C0 continuous.
        """
        h = 1.0

        # Evaluate slightly before and after the transition
        r_before = torch.tensor([h - 1e-6])
        r_after = torch.tensor([h + 1e-6])

        phi_before = roma_kernel_1d(r_before, h)
        phi_after = roma_kernel_1d(r_after, h)

        # Should be approximately equal (C0 continuous)
        error = torch.abs(phi_before - phi_after)
        assert error < 1e-4, \
            f"Discontinuity at r=h: |φ(h-ε) - φ(h+ε)| = {error:.2e}"

    def test_vectorization(self):
        """
        Verify that the function works with batched inputs.
        """
        h = 1.0
        r = torch.tensor([0.0, 0.5*h, h, 1.5*h, 2.0*h, 2.5*h])

        # Should not raise an error
        phi = roma_kernel_1d(r, h)

        assert phi.shape == r.shape, \
            f"Shape mismatch: input {r.shape}, output {phi.shape}"


class TestRKPMDelta3D:
    """Tests for the 3D RKPM-corrected delta function."""

    def test_reduces_to_product_with_identity_coefficients(self):
        """
        With b_coef = [1, 0, 0, ...], RKPM delta should reduce to product of 1D kernels.
        """
        h = torch.tensor([0.1, 0.1, 0.1])
        x_lag = torch.tensor([0.0, 0.0, 0.0])

        # Identity correction: only b0=1, all others zero
        b_coef = torch.zeros(10)
        b_coef[0] = 1.0

        # Test point
        x_eul = torch.tensor([[0.05, 0.08, -0.12]])

        # RKPM delta
        delta_rkpm = rkpm_delta_3d(x_eul, x_lag, h, b_coef)

        # Product of 1D kernels
        rx, ry, rz = x_eul[0] - x_lag
        delta_product = (roma_kernel_1d(torch.tensor([rx]), h[0]) *
                        roma_kernel_1d(torch.tensor([ry]), h[1]) *
                        roma_kernel_1d(torch.tensor([rz]), h[2]))

        error = torch.abs(delta_rkpm[0] - delta_product)
        assert error < 1e-10, \
            f"RKPM delta != product of 1D kernels: error = {error:.2e}"

    def test_polynomial_modification(self):
        """
        Verify that polynomial correction modifies the kernel value.
        """
        h = torch.tensor([0.1, 0.1, 0.1])
        x_lag = torch.tensor([0.0, 0.0, 0.0])
        x_eul = torch.tensor([[0.05, 0.08, -0.12]])

        # Identity correction
        b_identity = torch.zeros(10)
        b_identity[0] = 1.0

        # Linear correction: b0=1, b1=2 (adds 2*x term)
        b_linear = torch.zeros(10)
        b_linear[0] = 1.0
        b_linear[1] = 2.0

        delta_identity = rkpm_delta_3d(x_eul, x_lag, h, b_identity)
        delta_linear = rkpm_delta_3d(x_eul, x_lag, h, b_linear)

        # Expected ratio: (1 + 2*0.05) / 1 = 1.1
        expected_ratio = 1.0 + 2.0 * x_eul[0, 0]
        actual_ratio = delta_linear[0] / delta_identity[0]

        error = torch.abs(actual_ratio - expected_ratio)
        assert error < 1e-6, \
            f"Polynomial correction incorrect: ratio = {actual_ratio:.6f}, expected {expected_ratio:.6f}"

    def test_compact_support_3d(self):
        """
        Verify that RKPM delta is zero outside compact support region.
        """
        h = torch.tensor([0.1, 0.1, 0.1])
        x_lag = torch.tensor([0.0, 0.0, 0.0])
        b_coef = torch.ones(10)  # Non-trivial polynomial

        # Point far outside support (> 2h in all directions)
        x_far = torch.tensor([[0.3, 0.3, 0.3]])

        delta = rkpm_delta_3d(x_far, x_lag, h, b_coef)

        assert delta[0] == 0.0, \
            f"Delta not zero outside support: δ = {delta[0]:.2e}"

    def test_multiple_points(self):
        """
        Verify that the function works with multiple Eulerian points.
        """
        h = torch.tensor([0.1, 0.1, 0.1])
        x_lag = torch.tensor([0.0, 0.0, 0.0])
        b_coef = torch.zeros(10)
        b_coef[0] = 1.0

        # Multiple test points
        x_eul = torch.tensor([
            [0.05, 0.05, 0.05],
            [0.10, 0.10, 0.10],
            [0.15, 0.15, 0.15],
            [0.25, 0.25, 0.25]  # Outside support
        ])

        delta = rkpm_delta_3d(x_eul, x_lag, h, b_coef)

        assert delta.shape == (4,), \
            f"Shape mismatch: expected (4,), got {delta.shape}"

        # First 3 should be non-zero, last should be zero
        assert delta[0] > 0 and delta[1] > 0 and delta[2] > 0, \
            "Delta should be non-zero inside support"
        assert delta[3] == 0.0, \
            "Delta should be zero outside support"


@pytest.mark.parametrize("h_val", [0.05, 0.1, 0.2])
def test_kernel_scales_with_h(h_val):
    """
    Verify that kernel magnitude scales inversely with h (to maintain integral=1).
    """
    r = torch.tensor([0.0])

    phi_h = roma_kernel_1d(r, h_val)

    # At r=0, kernel should scale as 1/h (approximately)
    # For small h, φ(0) ≈ 3/(8h)
    expected_scaling = 3.0 / (8.0 * h_val)

    error = torch.abs(phi_h[0] - expected_scaling) / expected_scaling
    assert error < 0.1, \
        f"Scaling incorrect for h={h_val}: φ(0)={phi_h[0]:.4f}, expected≈{expected_scaling:.4f}"


def test_gradient_exists_almost_everywhere():
    """
    Verify that the kernel is differentiable almost everywhere.

    The Roma kernel is C0 but not C1 continuous, so derivatives may have jumps.
    This test just verifies that numerical differentiation doesn't explode.
    """
    h = 1.0
    r = torch.linspace(-2*h, 2*h, 1000, requires_grad=True)
    phi = roma_kernel_1d(r, h)

    # Compute "gradient" via finite differences (not true derivative at discontinuities)
    phi_sum = torch.sum(phi)
    phi_sum.backward()

    # Gradient should be finite everywhere
    assert torch.all(torch.isfinite(r.grad)), \
        "Gradient contains NaN or Inf"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
