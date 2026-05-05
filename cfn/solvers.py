"""NumPy reference solvers using the theoretical flux.

Provides a finite-volume solver against which the learned CFN can be
compared. The padding logic reuses :func:`cfn.padding.input_nonperiodic_padding`
so that boundary handling matches the network exactly.
"""

from __future__ import annotations

import numpy as np
import torch

from cfn.padding import input_nonperiodic_padding
from cfn.theoretical import flux_function


def rhs_flux_numpy(u_np: np.ndarray, dx: float) -> np.ndarray:
    """Right-hand side of the conservation law using the theoretical flux."""
    u_torch = torch.tensor(u_np, dtype=torch.float64, device="cpu").unsqueeze(-1)
    u_pad_torch = input_nonperiodic_padding(u_torch, 1, 1, mode="extrapolate")
    u_pad = u_pad_torch.squeeze(1).cpu().numpy()
    if u_pad.ndim == 1:
        u_pad = u_pad[np.newaxis, :]

    u_left = u_pad[:, :-1]
    u_right = u_pad[:, 1:]

    f_left = flux_function(u_left)
    f_right = flux_function(u_right)

    flux_interface = 0.5 * (f_left + f_right)
    return -(flux_interface[:, 1:] - flux_interface[:, :-1]) / dx


def tvd_rk3_numpy(u_np: np.ndarray, dt: float, dx: float) -> np.ndarray:
    """Third-order TVD Runge-Kutta time step."""
    u1 = u_np + dt * rhs_flux_numpy(u_np, dx)
    u2 = 0.75 * u_np + 0.25 * (u1 + dt * rhs_flux_numpy(u1, dx))
    u3 = (1 / 3) * u_np + (2 / 3) * (u2 + dt * rhs_flux_numpy(u2, dx))
    return u3


def euler_numpy(u_np: np.ndarray, dt: float, dx: float) -> np.ndarray:
    """Forward Euler time step."""
    return u_np + dt * rhs_flux_numpy(u_np, dx)
