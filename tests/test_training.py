"""Smoke tests for the training utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from cfn import CFN, apply_model
from cfn.sampling import build_epoch_dataset
from cfn.training import _make_fixed_validation_set, train_epoch

TRAJ_PATH = Path(__file__).resolve().parents[1] / "data" / "one_traj_loss_check.npy"


@pytest.mark.skipif(not TRAJ_PATH.exists(), reason=f"missing trajectory: {TRAJ_PATH}")
def test_train_epoch_on_one_traj_loss_check():
    """Mini end-to-end smoke test on the bundled Whitham trajectory."""
    torch.manual_seed(0)
    traj = np.load(TRAJ_PATH)
    assert traj.shape == (1, 3201, 500, 1)

    rng = np.random.default_rng(0)
    data = build_epoch_dataset(
        traj,
        L=160,
        window_t=700,
        window_x=80,
        num_samples=8,
        rng=rng,
        per_corner_fraction=0.05,
    )
    assert data["un"].shape == (8, 80, 1)
    assert data["un_p1"].shape == (8, 4, 80, 1)

    device = torch.device("cpu")
    model = CFN(features=[8, 8, 1], dt=0.03125, dx=0.4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    val = _make_fixed_validation_set(
        traj,
        device=device,
        window_t=700,
        window_x=80,
        num_val_samples=8,
        L=160,
        val_per_corner_fraction=0.1,
    )

    train_loss, val_loss = train_epoch(
        model,
        traj,
        optimizer,
        loss_fn,
        device=device,
        fixed_val_data=val,
        window_t=700,
        window_x=80,
        num_samples=8,
        L=160,
        margin=16,
        rollout_steps=4,
        num_draws=1,
        train_per_corner_fraction=0.05,
    )
    assert np.isfinite(train_loss)
    assert np.isfinite(val_loss)


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
