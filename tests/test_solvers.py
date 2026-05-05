"""Tests for the NumPy reference solver."""

from __future__ import annotations

import numpy as np

from cfn.solvers import euler_numpy, rhs_flux_numpy, tvd_rk3_numpy


def test_rhs_constant_field_is_zero():
    """A spatially constant field has zero flux divergence under any reasonable scheme."""
    u = 0.77 * np.ones((1, 64), dtype=np.float64)
    rhs = rhs_flux_numpy(u, dx=0.4)
    np.testing.assert_allclose(rhs, 0.0, atol=1e-12)


def test_tvd_rk3_constant_field_unchanged():
    u = 0.8 * np.ones((1, 64), dtype=np.float64)
    u_next = tvd_rk3_numpy(u, dt=0.03125, dx=0.4)
    np.testing.assert_allclose(u_next, u, atol=1e-12)


def test_euler_smooth_step_consistent_with_rhs():
    """Euler step should equal u + dt * rhs(u) up to roundoff."""
    rng = np.random.default_rng(0)
    u = 0.77 + 0.02 * rng.standard_normal((1, 32))
    dt, dx = 0.03125, 0.4
    expected = u + dt * rhs_flux_numpy(u, dx)
    got = euler_numpy(u, dt, dx)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-15)


def test_tvd_rk3_preserves_shape():
    u = 0.77 * np.ones((1, 32))
    out = tvd_rk3_numpy(u, dt=0.03125, dx=0.4)
    assert out.shape == u.shape
