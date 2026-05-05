"""Tests for padding routines."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cfn.padding import input_circular_padding, input_nonperiodic_padding


@pytest.fixture
def x():
    # shape [1, 5, 1]; values [10, 20, 30, 40, 50]
    return torch.tensor([[[10.0], [20.0], [30.0], [40.0], [50.0]]])


def test_circular_shape(x):
    out = input_circular_padding(x, 2, 3)
    assert out.shape == (1, 1, 5 + 2 + 3)


def test_outflow_left_constant(x):
    out = input_nonperiodic_padding(x, 2, 0, mode="outflow").squeeze().numpy()
    # Two ghost cells on the left should both equal the boundary value 10
    assert out[0] == 10.0
    assert out[1] == 10.0
    np.testing.assert_array_equal(out[2:], np.array([10.0, 20.0, 30.0, 40.0, 50.0]))


def test_outflow_right_constant(x):
    out = input_nonperiodic_padding(x, 0, 2, mode="outflow").squeeze().numpy()
    assert out[-1] == 50.0
    assert out[-2] == 50.0


def test_reflect(x):
    out = input_nonperiodic_padding(x, 2, 2, mode="reflect").squeeze().numpy()
    # Reflect excluding the boundary cell: ghosts mirror cells [1,2] on the left
    # and [-2, -3] on the right.
    np.testing.assert_array_equal(out[:2], np.array([30.0, 20.0]))
    np.testing.assert_array_equal(out[-2:], np.array([40.0, 30.0]))


def test_extrapolate_linear(x):
    # On a linear sequence, linear extrapolation should be exact.
    out = input_nonperiodic_padding(x, 2, 2, mode="extrapolate").squeeze().numpy()
    expected = np.array([-10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
    np.testing.assert_allclose(out, expected, rtol=1e-6)


def test_unknown_mode(x):
    with pytest.raises(ValueError):
        input_nonperiodic_padding(x, 1, 1, mode="bogus")


def test_zero_padding(x):
    out = input_nonperiodic_padding(x, 0, 0, mode="extrapolate").squeeze().numpy()
    np.testing.assert_array_equal(out, np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
