"""NumPy reference solvers using the theoretical flux.

Provides a finite-volume solver against which the learned CFN can be
compared. The padding logic reuses :mod:`cfn.padding` so that boundary
handling matches the network exactly.

The boundary mode is passed through to the padding helpers. Supported:
- ``"periodic"`` (circular padding) -- for Burgers-style periodic domains.
- ``"outflow"`` / ``"reflect"`` / ``"extrapolate"`` -- non-periodic ghost
  cells (default ``"extrapolate"`` matches the original Whitham setup).
"""

from __future__ import annotations

import numpy as np
import torch

from cfn import theoretical
from cfn.padding import input_circular_padding, input_nonperiodic_padding


def _pad_state(
    u_torch: torch.Tensor, left: int, right: int, boundary_mode: str
) -> torch.Tensor:
    """Apply the requested padding to a ``[batch, Nx, 1]`` tensor."""
    if boundary_mode == "periodic":
        return input_circular_padding(u_torch, left, right)
    return input_nonperiodic_padding(u_torch, left, right, mode=boundary_mode)


def rhs_flux_numpy(
    u_np: np.ndarray,
    dx: float,
    boundary_mode: str = "extrapolate",
) -> np.ndarray:
    """Right-hand side of the conservation law using the analytical flux.

    Parameters
    ----------
    u_np : np.ndarray
        State, shape ``[batch, Nx]`` or ``[Nx]``.
    dx : float
        Cell width.
    boundary_mode : str, default ``"extrapolate"``
        Passed through to the padding helpers. Use ``"periodic"`` for
        circular domains (e.g. Burgers on ``[0, 2pi]``).
    """
    u_torch = torch.tensor(u_np, dtype=torch.float64, device="cpu").unsqueeze(-1)
    u_pad_torch = _pad_state(u_torch, 1, 1, boundary_mode)
    u_pad = u_pad_torch.squeeze(1).cpu().numpy()
    if u_pad.ndim == 1:
        u_pad = u_pad[np.newaxis, :]

    u_left = u_pad[:, :-1]
    u_right = u_pad[:, 1:]

    # Look up flux at call time so theoretical.set_equation() takes effect.
    f_left = theoretical.flux_function(u_left)
    f_right = theoretical.flux_function(u_right)

    flux_interface = 0.5 * (f_left + f_right)
    return -(flux_interface[:, 1:] - flux_interface[:, :-1]) / dx


def tvd_rk3_numpy(
    u_np: np.ndarray,
    dt: float,
    dx: float,
    boundary_mode: str = "extrapolate",
) -> np.ndarray:
    """Third-order TVD Runge-Kutta time step."""
    rhs = lambda u: rhs_flux_numpy(u, dx, boundary_mode=boundary_mode)
    u1 = u_np + dt * rhs(u_np)
    u2 = 0.75 * u_np + 0.25 * (u1 + dt * rhs(u1))
    u3 = (1 / 3) * u_np + (2 / 3) * (u2 + dt * rhs(u2))
    return u3


def euler_numpy(
    u_np: np.ndarray,
    dt: float,
    dx: float,
    boundary_mode: str = "extrapolate",
) -> np.ndarray:
    """Forward Euler time step."""
    return u_np + dt * rhs_flux_numpy(u_np, dx, boundary_mode=boundary_mode)
