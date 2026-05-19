"""Conservative Flux Network (CFN) and its residual flux backbone."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cfn.padding import input_circular_padding, input_nonperiodic_padding


class ResidualBlock(nn.Module):
    """Residual block with two Conv1d layers and a skip connection.

    Notes
    -----
    BatchNorm is intentionally omitted to preserve time-marching consistency
    when the block is rolled out many steps inside the CFN training loop.
    The post-residual activation is also omitted to avoid staircase
    derivative behaviour from piecewise-linear activations.
    """

    def __init__(self, channels: int, kernel_size: int = 3, negative_slope: float = 0.01):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        # h = F.leaky_relu(h, negative_slope=self.negative_slope)
        h = torch.tanh(h)
        # h = F.silu(h)
        h = self.conv2(h)
        return h + x


# Boundary modes
_NONPERIODIC_MODES = {"outflow", "reflect", "extrapolate"}
_BOUNDARY_MODES = _NONPERIODIC_MODES | {"periodic"}


class Flux(nn.Module):
    """Numerical flux network ``u -> F(u)``.

    Maps ``[batch, Nx, 1]`` inputs to ``[batch, Nx + 1, out_channels]`` flux
    values defined on cell interfaces.

    Parameters
    ----------
    boundary_mode : str
        One of ``"periodic"`` (circular padding) or
        ``"outflow"`` / ``"reflect"`` / ``"extrapolate"`` (non-periodic ghost
        cells). The non-periodic modes are forwarded to
        :func:`cfn.padding.input_nonperiodic_padding`.
    """

    def __init__(
        self,
        features: list[int],
        left_padding: int,
        right_padding: int,
        num_blocks: int = 1,
        kernel_size: int = 3,
        boundary_mode: str = "extrapolate",
    ):
        super().__init__()
        if boundary_mode not in _BOUNDARY_MODES:
            raise ValueError(
                f"Unknown boundary_mode {boundary_mode!r}; "
                f"expected one of {sorted(_BOUNDARY_MODES)}"
            )
        self.features = features
        self.left_padding = left_padding
        self.right_padding = right_padding
        self.boundary_mode = boundary_mode

        # Lift scalar input to hidden channels
        self.input_conv = nn.Conv1d(1, features[0], kernel_size=1)
        # Deep residual representation
        self.blocks = nn.Sequential(
            *[ResidualBlock(features[0], kernel_size) for _ in range(num_blocks)]
        )
        # Project back to flux channels
        self.output_conv = nn.Conv1d(features[0], features[-1], kernel_size=1)

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        """Dispatch to the right padding function based on ``boundary_mode``."""
        if self.boundary_mode == "periodic":
            return input_circular_padding(x, self.left_padding, self.right_padding)
        return input_nonperiodic_padding(
            x, self.left_padding, self.right_padding, mode=self.boundary_mode
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_padded = self._pad(x)

        h = self.input_conv(x_padded)
        # h = F.leaky_relu(h)
        h = torch.tanh(h)
        # h = F.silu(h)
        h = self.blocks(h)
        out = self.output_conv(h)

        # Crop padding so the output sits on Nx + 1 interfaces
        if self.left_padding or self.right_padding:
            start = self.left_padding
            end = -self.right_padding + 1 if self.right_padding > 0 else None
            out = out[:, :, start:end]

        return out.transpose(1, 2)


class CFN(nn.Module):
    """Conservative Flux Network wrapping a learnable numerical flux.

    Implements forward Euler and TVD-RK3 time stepping for a 1D conservation
    law ``u_t + F(u)_x = 0``.

    Parameters
    ----------
    boundary_mode : str
        Padding mode for the flux network. ``"extrapolate"`` (default) matches
        the original non-periodic behaviour; pass ``"periodic"`` for circular
        boundary conditions (e.g. Burgers on a periodic domain).
    """

    DEFAULT_DX = 2 * math.pi / 512

    def __init__(
        self,
        features: list[int] | None = None,
        dt: float = 0.005,
        dx: float = DEFAULT_DX,
        boundary: str = "same",
        limiter: str = "minmod",
        left_padding: int = 2,
        right_padding: int = 3,
        boundary_mode: str = "extrapolate",
    ):
        super().__init__()
        if features is None:
            features = [64, 64, 64, 64, 64, 1]
        self.num_flux = Flux(
            features,
            left_padding=left_padding,
            right_padding=right_padding,
            boundary_mode=boundary_mode,
        )
        self.dt = dt
        self.dx = dx
        self.boundary = boundary.lower()
        self.limiter = limiter.lower()  # reserved for future use
        self.boundary_mode = boundary_mode

    def flux(self, up: torch.Tensor) -> torch.Tensor:
        return self.num_flux(up)

    def rhs(self, u: torch.Tensor) -> torch.Tensor:
        flux = self.num_flux(u)
        return -(flux[:, 1:, :] - flux[:, :-1, :]) / self.dx

    def euler(self, u: torch.Tensor) -> torch.Tensor:
        """Forward Euler time step."""
        return u + self.dt * self.rhs(u)

    def TVD_RK3(self, u: torch.Tensor) -> torch.Tensor:
        """Third-order TVD Runge-Kutta time step (Shu-Osher)."""
        u1 = u + self.dt * self.rhs(u)
        u2 = 0.75 * u + 0.25 * u1 + 0.25 * self.dt * self.rhs(u1)
        u3 = (1 / 3) * u + (2 / 3) * u2 + (2 / 3) * self.dt * self.rhs(u2)
        return u3
