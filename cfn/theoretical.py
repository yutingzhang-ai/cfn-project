"""Theoretical reference flux F(u) from Whitham modulation theory.

Construction uses complete elliptic integrals K(m), E(m). The result is a
strictly increasing relationship between phi(m) and the integrated flux,
sorted by phi for use as an interpolation table.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ellipe, ellipk


def build_flux(num_points: int = 2 ** 16) -> tuple[np.ndarray, np.ndarray]:
    """Build the (phi, F(phi)) reference table.

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


def flux_function(u: np.ndarray | float) -> np.ndarray | float:
    """Evaluate the theoretical flux at ``u`` via interpolation."""
    return np.interp(u, _PHI_REF, _FLUX_REF, left=_FLUX_REF[0], right=_FLUX_REF[-1])
