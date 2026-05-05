"""Shape sanity checks for the CFN model."""

from __future__ import annotations

import torch

from cfn import CFN
from cfn.data import build_epoch_dataset
import numpy as np


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
