"""Padding utilities for finite-volume style 1D networks.

These functions add ghost cells to a 1D field so that convolutions on the
interior produce flux values at the right number of cell interfaces.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def input_circular_padding(x: torch.Tensor, left: int, right: int) -> torch.Tensor:
    """Periodic (circular) padding for tensors of shape ``[batch, Nx, 1]``.

    Returns a tensor of shape ``[batch, 1, Nx + left + right]`` ready for Conv1d.
    """
    x_d2 = torch.squeeze(x, 2)
    x_d2 = torch.unsqueeze(x_d2, 1)
    return F.pad(x_d2, (left, right), mode="circular")


def input_nonperiodic_padding(
    x: torch.Tensor,
    left: int,
    right: int,
    mode: str = "outflow",
) -> torch.Tensor:
    """Non-periodic ghost-cell padding for tensors of shape ``[batch, Nx, 1]``.

    Parameters
    ----------
    x : torch.Tensor
        Input field, shape ``[batch, Nx, 1]``.
    left, right : int
        Number of ghost cells to add on each side.
    mode : {"outflow", "reflect", "extrapolate"}
        - ``outflow``    : constant extrapolation (Neumann / open boundary).
        - ``reflect``    : mirror reflection, excluding the boundary cell.
        - ``extrapolate``: linear extrapolation, ``u_ghost = 2*u_boundary - u_next``.

    Returns
    -------
    torch.Tensor
        Padded tensor of shape ``[batch, 1, Nx + left + right]``.
    """
    x_d2 = torch.squeeze(x, 2)       # [batch, Nx]
    x_d2 = torch.unsqueeze(x_d2, 1)  # [batch, 1, Nx]

    if mode == "outflow":
        left_vals = x_d2[:, :, :1].repeat(1, 1, left) if left > 0 else None
        right_vals = x_d2[:, :, -1:].repeat(1, 1, right) if right > 0 else None

    elif mode == "reflect":
        left_vals = x_d2[:, :, 1:left + 1].flip(-1) if left > 0 else None
        right_vals = x_d2[:, :, -(right + 1):-1].flip(-1) if right > 0 else None

    elif mode == "extrapolate":
        left_vals = _linear_extrapolate(
            base=x_d2[:, :, :1], neighbor=x_d2[:, :, 1:2], n=left, side="left"
        )
        right_vals = _linear_extrapolate(
            base=x_d2[:, :, -1:], neighbor=x_d2[:, :, -2:-1], n=right, side="right"
        )

    else:
        raise ValueError(f"Unknown non-periodic padding mode: {mode!r}")

    parts = []
    if left > 0:
        parts.append(left_vals)
    parts.append(x_d2)
    if right > 0:
        parts.append(right_vals)
    return torch.cat(parts, dim=-1)


def _linear_extrapolate(
    base: torch.Tensor, neighbor: torch.Tensor, n: int, side: str
) -> torch.Tensor | None:
    """Iteratively linear-extrapolate ``n`` ghost cells from ``base`` outward.

    Uses the relation ``new = 2*prev - curr`` to march outward.
    """
    if n <= 0:
        return None

    vals = []
    prev, curr = base, neighbor
    for _ in range(n):
        new = 2 * prev - curr
        vals.append(new)
        curr, prev = prev, new

    if side == "left":
        # ghost cells were generated from boundary outward; reverse so the
        # outermost ghost comes first
        return torch.cat(vals[::-1], dim=-1)
    return torch.cat(vals, dim=-1)
