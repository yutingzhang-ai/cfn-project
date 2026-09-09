"""Analytical reference fluxes F(u) for 1D scalar conservation laws.

Each equation exposes a callable ``flux_function(u) -> F(u)`` and a default
operating range ``(u_min, u_max)``. The currently-active flux is exposed at
module level as :data:`flux_function`, so existing callers
(``cfn.solvers``, ``scripts/evaluate.py``) keep working unchanged. To switch
equations, call :func:`set_equation` at startup, e.g.::

    from cfn import theoretical
    theoretical.set_equation("burgers")

Available equations
-------------------
- ``"whitham"``      : F built from elliptic integrals (Whitham modulation theory).
                       Default, matches the original behaviour.
- ``"burgers"``      : F(u) = 1/2 u^2.
- ``"saint_venant"`` : 1D shallow water system, F(U) = (q, q^2/h + 0.5 g h^2).
                       SYSTEM (n_components=2); requires a vector-state CFN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.special import ellipe, ellipk

# ---------------------------------------------------------------------------
# Whitham flux (elliptic-integral construction)
# ---------------------------------------------------------------------------

def build_flux(num_points: int = 2 ** 16) -> tuple[np.ndarray, np.ndarray]:
    """Build the (phi, F(phi)) Whitham reference table.

    Returns
    -------
    phi_sorted, flux_sorted : np.ndarray
        Both shape ``(num_points,)``, sorted by ``phi``.
    """
    m = np.linspace(1e-4, 0.999, num_points)
    K = ellipk(m)
    E = ellipe(m)

    phi = m - 1 + 2 * E / K
    phi_prime = 1 + (2 * E * K * (1 - m) - K * K * (1 - m) - E * E) / (m * (1 - m) * K * K)
    f = (1 / 3) * (m + 1) - (2 * m * (1 - m) * K) / (3 * (E - (1 - m) * K))
    zeta = -f
    flux = np.cumsum(-zeta * phi_prime) * (m[1] - m[0])

    idx = np.argsort(phi)
    return phi[idx], flux[idx]


# Build once at import time; cheap and constant.
_PHI_REF, _FLUX_REF = build_flux()
_DFLUX_DPHI = np.gradient(_FLUX_REF, _PHI_REF)


def flux_whitham(u: np.ndarray | float) -> np.ndarray | float:
    """Whitham theoretical flux via interpolation of the (phi, F) table."""
    return np.interp(u, _PHI_REF, _FLUX_REF, left=_FLUX_REF[0], right=_FLUX_REF[-1])


def wave_speed_whitham(u: np.ndarray | float) -> np.ndarray | float:
    """Local wave speed |F'(u)| from the Whitham interpolation table."""
    deriv = np.interp(u, _PHI_REF, _DFLUX_DPHI, left=_DFLUX_DPHI[0], right=_DFLUX_DPHI[-1])
    return np.abs(deriv)


# ---------------------------------------------------------------------------
# Burgers flux: F(u) = 1/2 u^2
# ---------------------------------------------------------------------------

def flux_burgers(u: np.ndarray | float) -> np.ndarray | float:
    """Burgers flux F(u) = 0.5 u^2.  Derivative F'(u) = u."""
    u_arr = np.asarray(u, dtype=np.float64)
    return 0.5 * u_arr ** 2


def wave_speed_burgers(u: np.ndarray | float) -> np.ndarray | float:
    """Local wave speed |F'(u)| = |u| for Burgers."""
    return np.abs(np.asarray(u, dtype=np.float64))


# ---------------------------------------------------------------------------
# Saint-Venant (1D shallow water) flux  --  SYSTEM of 2 equations
# ---------------------------------------------------------------------------
# State U = (h, q)^T where h is depth and q = h*u is momentum.
# Flux F(U) = (q, q^2/h + 0.5 g h^2)^T.
#
# This is a *system*, not a scalar law. The current scalar CFN cannot consume
# this directly: it expects shape [batch, Nx, 1], not [batch, Nx, 2]. Extending
# the model to systems requires:
#   1. Flux network output channels = 2.
#   2. Input channels = 2 (the model already takes a single channel; this
#      generalises straightforwardly via kernel_size=1 Conv1d width changes).
#   3. A numerical flux at interfaces that respects wave directionality
#      (Rusanov / HLL / Roe) -- central averaging is unstable for systems
#      with bidirectional waves.
#
# Until then, this flux is provided for unit testing and as a drop-in
# reference once the model is extended.

_G_GRAVITY = 9.81  # m / s^2, standard


def flux_saint_venant(
    U: np.ndarray, g: float = _G_GRAVITY
) -> np.ndarray:
    """Saint-Venant flux for state ``U = (h, q)``.

    Parameters
    ----------
    U : np.ndarray
        Shape ``(..., 2)`` with last axis = (h, q). ``h > 0`` required.
    g : float, default 9.81
        Gravitational acceleration.

    Returns
    -------
    F : np.ndarray
        Same shape as ``U``. ``F[..., 0] = q``, ``F[..., 1] = q^2/h + 0.5 g h^2``.
    """
    U_arr = np.asarray(U, dtype=np.float64)
    if U_arr.shape[-1] != 2:
        raise ValueError(
            f"Saint-Venant flux expects last axis size 2 (h, q); got shape {U_arr.shape}"
        )
    h = U_arr[..., 0]
    q = U_arr[..., 1]
    if np.any(h <= 0):
        raise ValueError("Saint-Venant flux requires h > 0 everywhere (no dry beds).")

    F = np.empty_like(U_arr)
    F[..., 0] = q
    F[..., 1] = q * q / h + 0.5 * g * h * h
    return F


def saint_venant_wave_speeds(
    U: np.ndarray, g: float = _G_GRAVITY
) -> tuple[np.ndarray, np.ndarray]:
    """Return the two characteristic wave speeds ``u - sqrt(gh)``, ``u + sqrt(gh)``.

    Useful for: CFL conditions, Rusanov / HLL numerical fluxes, and verifying
    that the learned flux Jacobian has the right spectrum.
    """
    U_arr = np.asarray(U, dtype=np.float64)
    h = U_arr[..., 0]
    q = U_arr[..., 1]
    u = q / h
    c = np.sqrt(g * h)
    return u - c, u + c


def wave_speed_saint_venant(U: np.ndarray, g: float = _G_GRAVITY) -> np.ndarray:
    """Rusanov spectral radius max(|u - sqrt(gh)|, |u + sqrt(gh)|)."""
    lam_m, lam_p = saint_venant_wave_speeds(U, g=g)
    return np.maximum(np.abs(lam_m), np.abs(lam_p))


# ---------------------------------------------------------------------------
# Equation registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Equation:
    """Bundle of a flux function with its natural operating range and anchors.

    For scalar conservation laws (``n_components == 1``), ``flux(u)`` accepts
    and returns arrays of the same shape, and ``u_min``/``u_max`` describe the
    1D operating range.

    For systems (``n_components > 1``), ``flux(U)`` accepts arrays with last
    axis = ``n_components`` and returns the same. ``u_min``/``u_max``/``anchors``
    refer to a *representative* scalar quantity (e.g. depth ``h`` for
    Saint-Venant) for plotting and diagnostic purposes only.

    ``wave_speed(u)`` is the local spectral radius used by the Rusanov
    flux: ``|F'(u)|`` for scalar laws, ``max |lambda|`` for systems.
    """
    name: str
    flux: Callable[[np.ndarray], np.ndarray]
    wave_speed: Callable[[np.ndarray], np.ndarray]
    u_min: float
    u_max: float
    anchors: tuple[float, ...]  # suggested anchor points for G-net
    n_components: int = 1       # 1 = scalar law, 2+ = system


EQUATIONS: dict[str, Equation] = {
    "whitham": Equation(
        name="whitham",
        flux=flux_whitham,
        wave_speed=wave_speed_whitham,
        u_min=0.66,
        u_max=0.88,
        anchors=(0.68, 0.72, 0.77, 0.82, 0.86),
    ),
    "burgers": Equation(
        name="burgers",
        flux=flux_burgers,
        wave_speed=wave_speed_burgers,
        u_min=-1.0,
        u_max=1.0,
        anchors=(-0.8, -0.4, 0.0, 0.4, 0.8),
    ),
    "saint_venant": Equation(
        name="saint_venant",
        flux=flux_saint_venant,
        wave_speed=wave_speed_saint_venant,
        # u_min/u_max here describe the depth h, not a scalar state.
        # Typical dam-break / Riemann problem ranges: h in [0.1, 2.0] m.
        u_min=0.1,
        u_max=2.0,
        anchors=(0.2, 0.5, 1.0, 1.5, 1.8),
        n_components=2,
    ),
}


def get_equation(name: str) -> Equation:
    """Look up an equation by name (case-insensitive)."""
    key = name.lower()
    if key not in EQUATIONS:
        raise KeyError(
            f"Unknown equation {name!r}. Available: {sorted(EQUATIONS)}"
        )
    return EQUATIONS[key]


# ---------------------------------------------------------------------------
# Module-level active flux
# ---------------------------------------------------------------------------
# ``flux_function`` is the symbol imported elsewhere in the codebase
# (``cfn.solvers``, ``scripts/evaluate.py``, tests). It defaults to Whitham
# so existing behaviour is preserved bit-for-bit; call ``set_equation(name)``
# to switch.

_ACTIVE: Equation = EQUATIONS["whitham"]
flux_function: Callable[[np.ndarray], np.ndarray] = _ACTIVE.flux


def set_equation(name: str) -> Equation:
    """Switch the module-level :data:`flux_function` to a different equation.

    Returns the newly-active :class:`Equation`. Callers that imported
    ``flux_function`` *before* this call still hold the old reference; use
    ``from cfn import theoretical`` and access ``theoretical.flux_function``
    if you need to follow the switch dynamically.

    Note
    ----
    Switching to a system equation (``n_components > 1``) makes
    ``flux_function`` expect arrays whose last axis is ``n_components``.
    Existing scalar callers (``cfn.solvers``, ``scripts/evaluate.py``) will
    not work until they're updated to handle vector states.
    """
    global _ACTIVE, flux_function
    _ACTIVE = get_equation(name)
    flux_function = _ACTIVE.flux
    return _ACTIVE


def active_equation() -> Equation:
    """Return the currently-active equation."""
    return _ACTIVE
