"""Sanity checks for the analytical Whitham flux."""

from __future__ import annotations

import numpy as np

from cfn.theoretical import build_flux, flux_function


def test_flux_function_monotone_in_physical_range():
    u = np.linspace(0.66, 0.88, 200)
    f = flux_function(u)
    assert f.shape == u.shape
    diffs = np.diff(f)
    # Strictly increasing: no negative or zero step.
    assert np.all(diffs > 0), "flux_function should be strictly increasing on [0.66, 0.88]"


def test_flux_function_handles_scalar():
    val = flux_function(0.77)
    assert np.isscalar(val) or np.ndim(val) == 0
    assert np.isfinite(float(val))


def test_flux_function_continuous_small_step():
    u = np.linspace(0.66, 0.88, 200)
    f = flux_function(u)
    # Lipschitz-style smoothness check: per-step jump should be small.
    step = np.max(np.abs(np.diff(f)))
    assert step < 0.05, f"flux_function jumps by {step:.3e} between adjacent samples"


def test_build_flux_returns_sorted_table():
    phi, flux = build_flux(num_points=4096)
    assert phi.shape == flux.shape
    assert np.all(np.diff(phi) >= 0), "phi must be sorted ascending"
    assert np.all(np.isfinite(phi))
    assert np.all(np.isfinite(flux))
