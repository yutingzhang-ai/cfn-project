# Data

Place your trajectory files here. Expected format:

- `.npy` file of shape `(1, Nt, Nx, 1)` — single trajectory, single channel.
- `dtype` should be `float32` or `float64`.

The default config expects `data/one_traj_loss_check.npy`.

Files in this folder are gitignored (see `.gitignore`).
