"""Dataset construction: random space-time windows from a long trajectory."""

from __future__ import annotations

import numpy as np


def build_epoch_dataset(
    full_data: np.ndarray,
    noise_level: float = 0.0,
    L: int = 160,
    window_t: int = 16,
    window_x: int = 80,
    num_samples: int = 16,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Sample random space-time windows and build (u_n, future) pairs.

    Parameters
    ----------
    full_data : np.ndarray
        Source trajectory, shape ``(1, Nt, Nx, C)``.
    noise_level : float
        Multiplicative noise added to ``un_p1`` (relative to mean(|data|)).
    L : int
        Time gap (in source time-steps) between successive supervision points.
    window_t, window_x : int
        Size of each sampled space-time window.
    num_samples : int
        Number of windows to sample per call.
    rng : np.random.Generator, optional
        Random generator. Uses the global numpy RNG if ``None``.

    Returns
    -------
    dict
        ``{"un": (num_samples, 1, window_x, C),
           "un_p1": (num_samples, n_targets, window_x, C)}``.
    """
    if rng is None:
        rng = np.random.default_rng()

    _, Nt, Nx, _C = full_data.shape
    if window_t > Nt or window_x > Nx:
        raise ValueError(
            f"window_t={window_t}, window_x={window_x} exceed trajectory shape "
            f"(Nt={Nt}, Nx={Nx})"
        )

    windows = []
    for _ in range(num_samples):
        t0 = rng.integers(0, Nt - window_t + 1)
        x0 = rng.integers(0, Nx - window_x + 1)
        windows.append(full_data[0, t0:t0 + window_t, x0:x0 + window_x, :])
    train_data = np.stack(windows, axis=0)

    un_list, un_p1_list = [], []
    for traj in train_data:                 # traj: [T, X, 1]
        u0 = traj[0:1, :, :]
        T = traj.shape[0]
        indices = list(range(L, T, L))
        u_seq = traj[indices, :, :]
        un_list.append(u0)
        un_p1_list.append(u_seq)

    un = np.concatenate(un_list, axis=0)
    un_p1 = np.stack(un_p1_list, axis=0)

    if noise_level > 0:
        scale = float(np.mean(np.abs(train_data))) * noise_level
        un_p1 = un_p1 + rng.normal(0.0, scale, size=un_p1.shape)

    return {"un": un, "un_p1": un_p1}
