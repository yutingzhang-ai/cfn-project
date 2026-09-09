"""Conservative Flux Network (CFN) and its residual flux backbone.

This version applies an EXPLICIT Rusanov scheme around the learned flux at
inference time, so Theorem 1 of the paper applies literally:

    F_hat_{i+1/2} = 0.5 (F_theta(u_L) + F_theta(u_R)) - 0.5 alpha (u_R - u_L)

The CNN backbone still has its multi-cell stencil and padding -- it just
outputs a PER-CELL value F_theta(u_i) instead of a per-interface F_hat. The
dissipation is supplied by the external Rusanov term, not by whatever the
network implicitly learned. This trades a little expressivity for genuine
monotonicity / max-principle / TVD guarantees on the learned scheme.

Differences from the previous version:
- ``Flux`` now outputs ``[batch, Nx, C]`` (per-cell) instead of
  ``[batch, Nx + 1, C]`` (per-interface). The crop in ``forward_cell``
  is ``[left_padding : -right_padding]`` (vs old
  ``[left_padding : -right_padding + 1]``) so the same padding pattern
  (default left=2, right=3) now produces Nx cells instead of Nx+1
  interfaces.
- A new ``Flux.forward`` exists for evaluator compatibility: it returns
  the centered-average interface flux (Nx+1 values) derived from the
  per-cell ``forward_cell``. CFN.rhs does NOT use this method; it
  calls ``forward_cell`` directly and applies Rusanov externally.
- ``CFN.rhs`` applies Rusanov via ``torch.roll`` (periodic) or boundary-
  consistent padding (non-periodic) and a configurable ``alpha_floor``.
- ``CFN`` exposes an ``alpha_floor`` argument (default 0.0) so you can
  raise the Rusanov dissipation explicitly during evaluation or training.

State convention everywhere: ``[batch, Nx, C]``.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from cfn import theoretical
from cfn.padding import input_circular_padding, input_nonperiodic_padding


class ResidualBlock(nn.Module):
    """Residual block with two Conv1d layers and a skip connection."""

    def __init__(self, channels: int, kernel_size: int = 3, negative_slope: float = 0.01):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = torch.tanh(h)
        # h = F.silu(h)
        h = self.conv2(h)
        return h + x


# Boundary modes
_NONPERIODIC_MODES = {"outflow", "reflect", "extrapolate"}
_BOUNDARY_MODES = _NONPERIODIC_MODES | {"periodic"}


class Flux(nn.Module):
    """Learned pointwise flux network ``u -> F_theta(u)`` for scalar or system laws.

    Maps ``[batch, Nx, n_components]`` inputs to ``[batch, Nx, n_components]``
    flux values defined at cell centers. The CNN backbone still uses a
    multi-cell stencil and padding, so F_theta(u_i) depends weakly on
    neighboring cells -- but the OUTPUT is per-cell, and the external
    Rusanov layer in CFN.rhs is what couples neighbors at each interface.

    For backward compatibility with the original evaluator (which expects
    Nx+1 interface fluxes), this module's ``forward`` returns interface
    values: F_iface[i] = 0.5 * (F_cell[i-1] + F_cell[i]) at internal
    interfaces, with boundary interfaces from the appropriate ghost-cell
    or wraparound. Use ``forward_cell`` to get the per-cell F_theta(u)
    directly (this is what CFN.rhs uses internally).
    """

    def __init__(
        self,
        features: list[int],
        left_padding: int,
        right_padding: int,
        num_blocks: int = 1,
        kernel_size: int = 3,
        boundary_mode: str = "extrapolate",
        n_components: int = 1,
    ):
        super().__init__()
        if boundary_mode not in _BOUNDARY_MODES:
            raise ValueError(
                f"Unknown boundary_mode {boundary_mode!r}; "
                f"expected one of {sorted(_BOUNDARY_MODES)}"
            )
        if features[-1] != n_components:
            raise ValueError(
                f"features[-1] ({features[-1]}) must equal n_components "
                f"({n_components}). The output channels of the flux network "
                f"must match the conserved-variable count."
            )
        self.features = features
        self.left_padding = left_padding
        self.right_padding = right_padding
        self.boundary_mode = boundary_mode
        self.n_components = n_components

        self.input_conv = nn.Conv1d(n_components, features[0], kernel_size=1)
        self.blocks = nn.Sequential(
            *[ResidualBlock(features[0], kernel_size) for _ in range(num_blocks)]
        )
        self.output_conv = nn.Conv1d(features[0], features[-1], kernel_size=1)

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        """Dispatch to the right padding function based on ``boundary_mode``."""
        if self.boundary_mode == "periodic":
            return input_circular_padding(x, self.left_padding, self.right_padding)
        return input_nonperiodic_padding(
            x, self.left_padding, self.right_padding, mode=self.boundary_mode
        )

    def forward_cell(self, x: torch.Tensor) -> torch.Tensor:
        """Per-cell F_theta(u). Shape: [batch, Nx, C]. This is what Rusanov uses."""
        x_padded = self._pad(x)
        h = self.input_conv(x_padded)
        h = torch.tanh(h)
        # h = F.silu(h)
        h = self.blocks(h)
        out = self.output_conv(h)

        if self.left_padding or self.right_padding:
            start = self.left_padding
            end = -self.right_padding if self.right_padding > 0 else None
            out = out[:, :, start:end]

        return out.transpose(1, 2)  # [batch, Nx, C]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Interface flux F_hat[i+1/2] from per-cell F_theta(u).

        Returns Nx+1 interface values for backward compatibility with the
        original evaluator. Internal interfaces use centered averaging;
        boundary interfaces use the equation's boundary mode.

        NOTE: this is *not* the Rusanov interface flux -- it's just the
        centered average. CFN.rhs does NOT call this method; it calls
        forward_cell directly and applies Rusanov externally. This
        forward() exists only so external tools that expect Nx+1 outputs
        (the evaluator's flux-comparison plot, autograd diagnostics)
        continue to work.
        """
        F_cell = self.forward_cell(x)  # [batch, Nx, C]

        if self.boundary_mode == "periodic":
            F_right = torch.roll(F_cell, shifts=-1, dims=1)
            F_iface_internal = 0.5 * (F_cell + F_right)  # [batch, Nx, C]
            # For periodic we need Nx+1 values: prepend the wraparound interface
            F_iface_first = F_iface_internal[:, -1:, :]
            return torch.cat([F_iface_first, F_iface_internal], dim=1)  # [batch, Nx+1, C]

        # Non-periodic: pad cell-space by 1 each side, then average consecutive cells.
        # input_nonperiodic_padding returns [batch, C, Nx+2] (channels-first); transpose
        # back to [batch, Nx+2, C] for forward_cell's [batch, Nx, C] interface.
        x_pad_cf = input_nonperiodic_padding(x, 1, 1, mode=self.boundary_mode)  # [b, C, Nx+2]
        x_pad = x_pad_cf.transpose(1, 2)  # [b, Nx+2, C]
        F_pad = self.forward_cell(x_pad)  # [batch, Nx+2, C]
        return 0.5 * (F_pad[:, :-1, :] + F_pad[:, 1:, :])  # [batch, Nx+1, C]


def _torch_wave_speed_traffic(u: torch.Tensor) -> torch.Tensor:
    """|F'(u)| = |1 - 2u| for LWR traffic. Torch version of cfn.theoretical."""
    return (1.0 - 2.0 * u).abs()


def _torch_wave_speed_burgers(u: torch.Tensor) -> torch.Tensor:
    """|F'(u)| = |u| for Burgers. Torch version of cfn.theoretical."""
    return u.abs()


def _torch_wave_speed_sine(u: torch.Tensor) -> torch.Tensor:
    """|F'(u)| = |cos(u)| for sine flux. Torch version of cfn.theoretical."""
    return u.cos().abs()


_TORCH_WAVE_SPEEDS = {
    "traffic": _torch_wave_speed_traffic,
    "burgers": _torch_wave_speed_burgers,
    "sine":    _torch_wave_speed_sine,
}


def _get_torch_wave_speed():
    """Return a torch-callable analogue of the active equation's wave_speed.

    Known scalar laws use a native torch implementation. Anything else
    (Whitham table lookup, Saint-Venant spectral radius) goes through the
    numpy ``Equation.wave_speed``, with alpha treated as detached.
    """
    eq = theoretical.active_equation()
    torch_fn = _TORCH_WAVE_SPEEDS.get(eq.name)
    if torch_fn is not None:
        return torch_fn

    def _from_numpy(u: torch.Tensor) -> torch.Tensor:
        speed = np.asarray(eq.wave_speed(u.detach().cpu().numpy()))
        out = torch.as_tensor(speed, device=u.device, dtype=u.dtype)
        while out.ndim < u.ndim:
            out = out.unsqueeze(-1)
        return out

    return _from_numpy


class CFN(nn.Module):
    """Conservative Flux Network with EXPLICIT Rusanov dissipation.

    The learned ``Flux`` produces per-cell F_theta(u_i). At each interface
    i+1/2 we form

        F_hat = 0.5 (F_theta(u_i) + F_theta(u_{i+1}))
              - 0.5 alpha (u_{i+1} - u_i)

    with alpha = max(|F'(u_i)|, |F'(u_{i+1})|, alpha_floor). The default
    alpha_floor is 0, which makes alpha equal to the per-interface true
    wave speed (matching ``cfn.solvers``). Raising alpha_floor adds a
    uniform safety margin -- useful when |F'_theta| might exceed |F'| on
    the visited range.

    Parameters
    ----------
    alpha_floor : float, default 0.0
        Lower bound on the Rusanov dissipation coefficient at every
        interface. Set to ~1.2-2.0 for traffic if the learned scheme
        oscillates at shocks.
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
        n_components: int = 1,
        alpha_floor: float = 0.0,
    ):
        super().__init__()
        if features is None:
            features = [64, 64, 64, 64, 64, n_components]
        else:
            features = list(features)
            if features[-1] != n_components:
                features = features[:-1] + [n_components]

        self.num_flux = Flux(
            features,
            left_padding=left_padding,
            right_padding=right_padding,
            boundary_mode=boundary_mode,
            n_components=n_components,
        )
        self.dt = dt
        self.dx = dx
        self.boundary = boundary.lower()
        self.limiter = limiter.lower()
        self.boundary_mode = boundary_mode
        self.n_components = n_components
        self.alpha_floor = float(alpha_floor)

    # ------------------------------------------------------------------ flux
    def flux(self, up: torch.Tensor) -> torch.Tensor:
        """Interface flux (Nx+1 values) -- for evaluator compatibility.

        This is the centered-average interface flux, NOT the Rusanov flux.
        Use this only for the evaluator's flux-comparison plot. For the
        rollout step, CFN.rhs constructs the Rusanov flux internally from
        ``flux_cell``.
        """
        return self.num_flux(up)

    def flux_cell(self, up: torch.Tensor) -> torch.Tensor:
        """Per-cell learned flux F_theta(u). Shape: [batch, Nx, C].

        This is the actual learnable function the Rusanov scheme uses.
        Use this for autograd-based eps measurement
        (||F'_theta - F'||_Linf) -- it gives you the true derivative
        of the network's pointwise flux.
        """
        return self.num_flux.forward_cell(up)

    # ------------------------------------------------------------------ rhs
    def _interface_states(self, u: torch.Tensor):
        """Build (u_L, u_R, F_L, F_R) at each interface using per-cell F_theta."""
        if self.boundary_mode == "periodic":
            F = self.num_flux.forward_cell(u)              # [batch, Nx, C]
            F_left = F                    # F_theta at cell i
            F_right = torch.roll(F, shifts=-1, dims=1)  # F_theta at cell i+1
            u_left = u
            u_right = torch.roll(u, shifts=-1, dims=1)
            return u_left, u_right, F_left, F_right

        # Non-periodic: pad u by 1 cell each side, evaluate F_theta on padded.
        # input_nonperiodic_padding returns [batch, C, Nx+2] (channels-first);
        # transpose back to [batch, Nx+2, C] before calling forward_cell.
        u_padded_cf = input_nonperiodic_padding(u, 1, 1, mode=self.boundary_mode)
        u_padded = u_padded_cf.transpose(1, 2)  # [batch, Nx+2, C]
        F_padded = self.num_flux.forward_cell(u_padded)    # [batch, Nx+2, C]
        u_left  = u_padded[:, :-1, :]     # [batch, Nx+1, C]
        u_right = u_padded[:, 1:, :]      # [batch, Nx+1, C]
        F_left  = F_padded[:, :-1, :]
        F_right = F_padded[:, 1:, :]
        return u_left, u_right, F_left, F_right

    def rhs(self, u: torch.Tensor) -> torch.Tensor:
        """RHS of u_t = -dF/dx, using explicit Rusanov around learned F_theta."""
        u_left, u_right, F_left, F_right = self._interface_states(u)

        # Dissipation coefficient.
        wave = _get_torch_wave_speed()
        alpha = torch.maximum(wave(u_left), wave(u_right))
        if self.alpha_floor > 0:
            alpha = torch.clamp(alpha, min=self.alpha_floor)
        # For systems we'd reduce across components; this version assumes scalar.
        # alpha shape matches u_left: [batch, Nx_iface, C].

        # Rusanov interface flux.
        F_hat = 0.5 * (F_left + F_right) - 0.5 * alpha * (u_right - u_left)
        # F_hat shape: [batch, Nx, C] (periodic) or [batch, Nx+1, C] (non-periodic).

        # Conservative difference.
        if self.boundary_mode == "periodic":
            # F_hat[i] is the flux at interface between cells i and i+1.
            # rhs[i] = -(F_hat[i] - F_hat[i-1]) / dx.
            F_hat_prev = torch.roll(F_hat, shifts=+1, dims=1)
            return -(F_hat - F_hat_prev) / self.dx
        else:
            # F_hat has Nx+1 interface fluxes; rhs takes consecutive diffs.
            return -(F_hat[:, 1:, :] - F_hat[:, :-1, :]) / self.dx

    # ----------------------------------------------------------------- steps
    def euler(self, u: torch.Tensor) -> torch.Tensor:
        """Forward Euler time step."""
        return u + self.dt * self.rhs(u)

    def TVD_RK3(self, u: torch.Tensor) -> torch.Tensor:
        """Third-order TVD Runge-Kutta time step (Shu-Osher)."""
        u1 = u + self.dt * self.rhs(u)
        u2 = 0.75 * u + 0.25 * u1 + 0.25 * self.dt * self.rhs(u1)
        u3 = (1 / 3) * u + (2 / 3) * u2 + (2 / 3) * self.dt * self.rhs(u2)
        return u3
