"""Shape sanity checks for the CFN model."""

from __future__ import annotations

import numpy as np
import torch

from cfn import CFN, theoretical
from cfn.models import _get_torch_wave_speed
from cfn.sampling import build_epoch_dataset


def test_cfn_forward_shapes():
    model = CFN(features=[8, 8, 1], dt=0.01, dx=0.1)
    u = torch.randn(2, 32, 1)
    rhs = model.rhs(u)
    assert rhs.shape == u.shape

    u_next = model.euler(u)
    assert u_next.shape == u.shape

    u_rk = model.TVD_RK3(u)
    assert u_rk.shape == u.shape


def test_dataset_shapes():
    rng = np.random.default_rng(0)
    fake = rng.standard_normal((1, 200, 64, 1)).astype(np.float32)
    data = build_epoch_dataset(fake, L=10, window_t=40, window_x=32, num_samples=4, rng=rng)
    assert data["un"].shape == (4, 32, 1)
    # window_t // L - 1 = 3 future targets at indices 10, 20, 30
    assert data["un_p1"].shape == (4, 3, 32, 1)


def test_torch_wave_speed_matches_whitham_finite_diff():
    """Whitham is not in _TORCH_WAVE_SPEEDS; the numpy fallback must equal |F'|."""
    prev = theoretical.active_equation().name
    try:
        theoretical.set_equation("whitham")
        u = np.linspace(0.66, 0.88, 200)
        eps = 1e-4
        flux = theoretical.active_equation().flux
        fd = np.abs((flux(u + eps) - flux(u - eps)) / (2.0 * eps))

        wave = _get_torch_wave_speed()
        got = wave(torch.tensor(u, dtype=torch.float64)).numpy()
        np.testing.assert_allclose(got, fd, rtol=5e-3, atol=1e-4)
        # Must not have collapsed to the old constant-alpha=1 fallback.
        assert not np.allclose(got, 1.0, atol=1e-2)
    finally:
        theoretical.set_equation(prev)


def test_cfn_rhs_forward_backward_whitham():
    """CFN.rhs remains trainable after the Whitham wave-speed fallback."""
    prev = theoretical.active_equation().name
    try:
        theoretical.set_equation("whitham")
        torch.manual_seed(0)
        model = CFN(features=[8, 8, 1], dt=0.01, dx=0.1)
        u = (0.77 + 0.02 * torch.randn(2, 32, 1)).clamp(0.66, 0.88)

        rhs = model.rhs(u)
        assert rhs.shape == u.shape
        assert torch.all(torch.isfinite(rhs))

        loss = (rhs ** 2).mean()
        loss.backward()
        assert torch.isfinite(loss.detach())
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert grads and all(g is not None for g in grads)
        assert all(torch.all(torch.isfinite(g)) for g in grads)
    finally:
        theoretical.set_equation(prev)
