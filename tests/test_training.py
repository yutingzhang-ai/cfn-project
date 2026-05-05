"""Smoke tests for the training utilities."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from cfn import CFN, apply_model


def test_constant_input_produces_near_zero_rhs():
    """For a constant field, the conv-based numerical flux is also constant
    (extrapolation padding preserves the constant), so the flux divergence is zero
    to numerical precision."""
    torch.manual_seed(0)
    model = CFN(features=[8, 8, 1], dt=0.01, dx=0.1)
    model.eval()
    u = 0.77 * torch.ones(1, 32, 1)
    with torch.no_grad():
        rhs = model.rhs(u)
    assert torch.all(torch.isfinite(rhs))
    assert rhs.abs().max().item() < 1e-5


def test_apply_model_smoke():
    """One rollout should produce a finite scalar loss and the right per-step list."""
    torch.manual_seed(0)
    model = CFN(features=[8, 8, 1], dt=0.01, dx=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    Nx = 32
    rollout_steps = 3
    un = 0.77 + 0.01 * torch.randn(1, Nx, 1)
    u_np1 = 0.77 + 0.01 * torch.randn(1, rollout_steps, Nx, 1)

    loss, per_step = apply_model(
        model, un, u_np1, loss_fn, optimizer,
        is_training=True,
        k=2,
        margin=2,
        rollout_steps=rollout_steps,
    )
    assert np.isfinite(loss)
    assert len(per_step) == rollout_steps
    assert all(np.isfinite(v) for v in per_step)


def test_apply_model_supports_rk3_integrator():
    torch.manual_seed(0)
    model = CFN(features=[8, 8, 1], dt=0.01, dx=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    Nx = 32
    rollout_steps = 2
    un = 0.77 + 0.01 * torch.randn(1, Nx, 1)
    u_np1 = 0.77 + 0.01 * torch.randn(1, rollout_steps, Nx, 1)

    loss, per_step = apply_model(
        model, un, u_np1, loss_fn, optimizer,
        is_training=False,
        k=2,
        margin=2,
        rollout_steps=rollout_steps,
        integrator="rk3",
    )
    assert np.isfinite(loss)
    assert len(per_step) == rollout_steps


def test_apply_model_unknown_integrator_raises():
    import pytest
    torch.manual_seed(0)
    model = CFN(features=[8, 8, 1], dt=0.01, dx=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    un = 0.77 * torch.ones(1, 16, 1)
    u_np1 = 0.77 * torch.ones(1, 1, 16, 1)
    with pytest.raises(ValueError):
        apply_model(
            model, un, u_np1, loss_fn, optimizer,
            is_training=False,
            k=1,
            margin=1,
            rollout_steps=1,
            integrator="bogus",
        )


def test_apply_model_rollout_steps_inferred():
    """When ``rollout_steps`` is None, it should default to u_np1.shape[1]."""
    torch.manual_seed(0)
    model = CFN(features=[8, 8, 1], dt=0.01, dx=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    Nx = 32
    un = 0.77 + 0.01 * torch.randn(1, Nx, 1)
    u_np1 = 0.77 + 0.01 * torch.randn(1, 4, Nx, 1)

    _, per_step = apply_model(
        model, un, u_np1, loss_fn, optimizer,
        is_training=False,
        k=1,
        margin=2,
        rollout_steps=None,
    )
    assert len(per_step) == 4
