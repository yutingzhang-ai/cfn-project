"""NumPy reference solvers using the theoretical flux.

State convention is ``[batch, Nx, C]`` with ``C = n_components``. For
back-compat, ``[batch, Nx]`` and ``[Nx]`` scalar inputs are also accepted
and returned in the same shape they arrived in.

The default interface flux is **Rusanov** (local Lax-Friedrichs):

    F_hat(uL, uR) = 1/2 (F(uL) + F(uR)) - alpha/2 (uR - uL),
    alpha        = max(wave_speed(uL), wave_speed(uR)),

matching the scheme analysed in the discrete rollout error bound
(Theorem 1). The old central-average flux is still available via
``scheme="central"`` but is unstable across shocks (Burgers, traffic) and
unstable for systems with bidirectional waves (Saint-Venant), so use it
only on smooth scalar problems.

CFL: the caller is responsible for choosing ``dt`` so that
``max(alpha) * dt / dx <= 1`` over the visited state range.
"""

from __future__ import annotations

import numpy as np
import torch

from cfn import theoretical
from cfn.padding import input_circular_padding, input_nonperiodic_padding


# ---------------------------------------------------------------------------
# Rank normalization helpers (allow legacy [Nx] / [batch, Nx] scalar inputs)
# ---------------------------------------------------------------------------

def _ensure_bxc(u_np: np.ndarray) -> tuple[np.ndarray, int]:
    """Normalize input to ``[batch, Nx, C]``; return (array, original_rank)."""
    u_np = np.asarray(u_np)
    if u_np.ndim == 1:
        return u_np[None, :, None], 1
    if u_np.ndim == 2:
        return u_np[..., None], 2
    if u_np.ndim == 3:
        return u_np, 3
    raise ValueError(
        f"Unexpected input shape {u_np.shape}; want 1D [Nx], 2D [batch, Nx], "
        f"or 3D [batch, Nx, C]."
    )


def _restore_rank(u_bxc: np.ndarray, original_rank: int) -> np.ndarray:
    """Inverse of :func:`_ensure_bxc`."""
    if original_rank == 1:
        return u_bxc[0, :, 0]
    if original_rank == 2:
        return u_bxc[..., 0]
    return u_bxc


# ---------------------------------------------------------------------------
# Padding & interface reconstruction
# ---------------------------------------------------------------------------

def _pad_state(u_torch, left, right, boundary_mode):
    if boundary_mode == "periodic":
        return input_circular_padding(u_torch, left, right)
    return input_nonperiodic_padding(u_torch, left, right, mode=boundary_mode)


def _interface_states(u_bxc: np.ndarray, boundary_mode: str
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Return (u_left, u_right), each shape ``[batch, Nx+1, C]``."""
    u_torch = torch.tensor(u_bxc, dtype=torch.float64, device="cpu")
    u_pad_bcn = _pad_state(u_torch, 1, 1, boundary_mode)   # [batch, C, Nx+2]
    u_pad = u_pad_bcn.transpose(1, 2).cpu().numpy()        # [batch, Nx+2, C]
    return u_pad[:, :-1, :], u_pad[:, 1:, :]


def _interface_flux(u_left: np.ndarray, u_right: np.ndarray,
                    scheme: str) -> np.ndarray:
    """Combine left/right interface states into a numerical flux.

    Both ``u_left`` and ``u_right`` are ``[batch, Nx+1, C]``; output has the
    same shape. Look up flux/wave_speed at call time so
    :func:`theoretical.set_equation` takes effect dynamically.
    """
    f_left = theoretical.flux_function(u_left)
    f_right = theoretical.flux_function(u_right)

    if scheme == "central":
        return 0.5 * (f_left + f_right)

    if scheme == "rusanov":
        wave = theoretical.active_equation().wave_speed
        # wave_speed returns a scalar per cell, shape [batch, Nx+1].
        # Broadcast across components for the dissipative jump term.
        alpha = np.maximum(wave(u_left), wave(u_right))
        # Floor the local dissipation coefficient. Keeping alpha >= 1.5
        # guarantees a uniform minimum amount of upwind diffusion across
        # every equation/state, which stabilises low-speed regions where
        # |F'(u)| is small (e.g. Whitham, sine, linearised SW perturbations).
        alpha = np.maximum(alpha, 1.5)
        if alpha.ndim == u_left.ndim - 1:
            alpha = alpha[..., None]
        return 0.5 * (f_left + f_right) - 0.5 * alpha * (u_right - u_left)

    raise ValueError(f"Unknown scheme {scheme!r}; expected 'rusanov' or 'central'.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rhs_flux_numpy(
    u_np: np.ndarray,
    dx: float,
    boundary_mode: str = "extrapolate",
    scheme: str = "rusanov",
) -> np.ndarray:
    """Right-hand side of the conservation law using the analytical flux.

    Parameters
    ----------
    u_np : np.ndarray
        State, shape ``[Nx]``, ``[batch, Nx]`` (scalar shorthand) or
        ``[batch, Nx, C]`` (general). The output shape matches the input.
    dx : float
        Cell width.
    boundary_mode : str, default ``"extrapolate"``
        Passed through to the padding helpers. Use ``"periodic"`` for
        circular domains (e.g. Burgers on ``[0, 2pi]``).
    scheme : {"rusanov", "central"}, default ``"rusanov"``
        Interface flux. Rusanov is monotone and conservative; matches the
        scheme in the discrete rollout error bound.
    """
    u_bxc, rank = _ensure_bxc(u_np)
    u_left, u_right = _interface_states(u_bxc, boundary_mode)
    flux_interface = _interface_flux(u_left, u_right, scheme)
    rhs_bxc = -(flux_interface[:, 1:, :] - flux_interface[:, :-1, :]) / dx
    return _restore_rank(rhs_bxc, rank)


def tvd_rk3_numpy(
    u_np: np.ndarray,
    dt: float,
    dx: float,
    boundary_mode: str = "extrapolate",
    scheme: str = "rusanov",
) -> np.ndarray:
    """Third-order TVD Runge-Kutta time step (Shu-Osher)."""
    rhs = lambda u: rhs_flux_numpy(u, dx, boundary_mode=boundary_mode, scheme=scheme)
    u1 = u_np + dt * rhs(u_np)
    u2 = 0.75 * u_np + 0.25 * (u1 + dt * rhs(u1))
    u3 = (1 / 3) * u_np + (2 / 3) * (u2 + dt * rhs(u2))
    return u3


def euler_numpy(
    u_np: np.ndarray,
    dt: float,
    dx: float,
    boundary_mode: str = "extrapolate",
    scheme: str = "rusanov",
) -> np.ndarray:
    """Forward Euler time step.

    With ``scheme='rusanov'`` this is exactly the update analysed in
    Theorem 1 of the rollout error bound.
    """
    return u_np + dt * rhs_flux_numpy(u_np, dx, boundary_mode=boundary_mode, scheme=scheme)
