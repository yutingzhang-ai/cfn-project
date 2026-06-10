"""Dataset construction: random space-time windows from a long trajectory."""

from __future__ import annotations

import itertools

import numpy as np


def _window_start_range(
    target: int, window: int, axis_len: int
) -> tuple[int, int]:
    """Inclusive range of starts ``i0`` such that ``[i0, i0+window)`` covers ``target``."""
    lo = max(0, target - window + 1)
    hi = min(axis_len - window, target)
    return lo, hi


def _find_corner_locations(full_data: np.ndarray) -> list[tuple[int, int]]:
    """Return one ``(t*, x*)`` per corner of the per-channel bounding box.

    For ``C`` channels there are ``2^C`` corners: each picks either the min
    or max along each channel. For each corner we find the trajectory cell
    closest to it in normalized channel space (so channels with different
    magnitudes contribute comparably to the distance).
    """
    field = full_data[0]                                   # [Nt, Nx, C]
    Nt, Nx, C = field.shape

    ch_min = field.reshape(-1, C).min(axis=0)              # (C,)
    ch_max = field.reshape(-1, C).max(axis=0)              # (C,)
    span = ch_max - ch_min
    span = np.where(span > 0, span, 1.0)                   # avoid divide-by-zero

    field_norm = (field - ch_min) / span                   # [Nt, Nx, C] in [0, 1]

    corners: list[tuple[int, int]] = []
    for corner in itertools.product([0.0, 1.0], repeat=C):
        target = np.asarray(corner, dtype=field_norm.dtype)  # (C,)
        dist = ((field_norm - target) ** 2).sum(axis=-1)     # [Nt, Nx]
        t_star, x_star = np.unravel_index(np.argmin(dist), dist.shape)
        corners.append((int(t_star), int(x_star)))
    return corners


def build_epoch_dataset(
    full_data: np.ndarray,
    noise_level: float = 0.0,
    L: int = 6,
    window_t: int = 16,
    window_x: int = 80,
    num_samples: int = 16,
    rng: np.random.Generator | None = None,
    per_corner_fraction: float = 0.05,
) -> dict[str, np.ndarray]:
    """Sample random space-time windows and build (u_n, future) pairs.

    Sampling strategy
    -----------------
    Each of the ``2^C`` corners of the per-channel bounding box of the
    trajectory receives a guaranteed share ``per_corner_fraction`` of the
    sample budget. The remaining
    ``(1 - per_corner_fraction * 2^C)`` fraction is sampled uniformly at
    random.

    Examples (``per_corner_fraction = 0.05``, ``num_samples = 126``):
      - ``C=1`` (2 corners): 6 per corner + 114 uniform   (10% total importance)
      - ``C=2`` (4 corners): 6 per corner + 102 uniform   (20% total)
      - ``C=3`` (8 corners): 6 per corner + 78 uniform    (40% total)

    Currently assumes ``C <= 3``; at ``C=4`` the corner budget would
    already consume 80% of samples.

    Set ``per_corner_fraction=0`` (default) to recover purely uniform sampling.

    Supports multi-channel trajectories (C >= 1). Noise, when enabled, is
    scaled per-channel using ``mean(|data[..., c]|)`` so each component
    receives noise proportional to its own magnitude -- important for
    systems like Saint-Venant where ``h`` and ``q = h u`` have very
    different scales.

    Parameters
    ----------
    full_data : np.ndarray
        Source trajectory, shape ``(1, Nt, Nx, C)``.
    noise_level : float
        Per-channel multiplicative noise added to ``un_p1``
        (relative to ``mean(|data[..., c]|)``).
    L : int
        Time gap (in source time-steps) between successive supervision points.
    window_t, window_x : int
        Size of each sampled space-time window.
    num_samples : int
        Number of windows to sample per call.
    rng : np.random.Generator, optional
        Random generator. Uses a fresh default RNG if ``None``.
    per_corner_fraction : float, default 0.0
        Fraction of total samples guaranteed to contain *each* bounding-box
        corner. Total importance fraction = ``per_corner_fraction * 2^C``.

    Returns
    -------
    dict
        ``{"un":    (num_samples, window_x, C),
           "un_p1": (num_samples, n_targets, window_x, C)}``.
    """
    if rng is None:
        rng = np.random.default_rng()

    if full_data.ndim != 4:
        raise ValueError(
            f"Expected full_data of shape (1, Nt, Nx, C); got {full_data.shape}"
        )
    _, Nt, Nx, C = full_data.shape
    if window_t > Nt or window_x > Nx:
        raise ValueError(
            f"window_t={window_t}, window_x={window_x} exceed trajectory shape "
            f"(Nt={Nt}, Nx={Nx})"
        )
    if per_corner_fraction < 0.0:
        raise ValueError(
            f"per_corner_fraction must be >= 0; got {per_corner_fraction}"
        )
    if C > 3 and per_corner_fraction > 0.0:
        # Guardrail: 2^C corners blows up the importance budget past C=3.
        # Remove this check if you genuinely want to handle larger systems.
        raise ValueError(
            f"per_corner_fraction assumes C <= 3 (current C={C}); "
            f"2^C = {2 ** C} corners would consume too much of num_samples."
        )

    # --- Budget split ---
    n_per_corner = int(per_corner_fraction * num_samples)
    n_corners = 2 ** C if n_per_corner > 0 else 0
    n_importance = n_per_corner * n_corners
    if n_importance > num_samples:
        raise ValueError(
            f"per_corner_fraction={per_corner_fraction} too large for C={C}: "
            f"{n_corners} corners * {n_per_corner} per corner = {n_importance} "
            f"> num_samples={num_samples}"
        )
    n_uniform = num_samples - n_importance

    # --- Build the (t0, x0) list ---
    starts: list[tuple[int, int]] = []

    if n_per_corner > 0:
        corners = _find_corner_locations(full_data)
        for (t_star, x_star) in corners:
            t_lo, t_hi = _window_start_range(t_star, window_t, Nt)
            x_lo, x_hi = _window_start_range(x_star, window_x, Nx)
            for _ in range(n_per_corner):
                t0 = rng.integers(t_lo, t_hi + 1)
                x0 = rng.integers(x_lo, x_hi + 1)
                starts.append((int(t0), int(x0)))

    for _ in range(n_uniform):
        t0 = rng.integers(0, Nt - window_t + 1)
        x0 = rng.integers(0, Nx - window_x + 1)
        starts.append((int(t0), int(x0)))

    # Shuffle so importance windows aren't grouped at the front of the batch.
    rng.shuffle(starts)

    # --- Extract windows ---
    windows = [
        full_data[0, t0:t0 + window_t, x0:x0 + window_x, :]
        for (t0, x0) in starts
    ]
    train_data = np.stack(windows, axis=0)                 # [N, T, X, C]

    un_list, un_p1_list = [], []
    for traj in train_data:                                # traj: [T, X, C]
        u0 = traj[0:1, :, :]
        T = traj.shape[0]
        indices = list(range(L, T, L))
        u_seq = traj[indices, :, :]
        un_list.append(u0)
        un_p1_list.append(u_seq)

    un = np.concatenate(un_list, axis=0)                   # [N, X, C]
    un_p1 = np.stack(un_p1_list, axis=0)                   # [N, n_targets, X, C]

    if noise_level > 0:
        # Per-channel magnitude, broadcastable against un_p1 [N, n_t, X, C].
        per_channel_scale = np.mean(
            np.abs(train_data), axis=(0, 1, 2)
        ).astype(un_p1.dtype)                              # (C,)
        scale = per_channel_scale * noise_level
        un_p1 = un_p1 + rng.normal(0.0, 1.0, size=un_p1.shape) * scale

    return {"un": un, "un_p1": un_p1}
