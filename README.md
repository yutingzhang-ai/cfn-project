# CFN Project

Use Conservative Form Networks to predict or recover modulation equations.

## Project Structure

```text
cfn-project/
├── cfn/                        # main package
│   ├── models.py               # CFN, flux network, ResidualBlock
│   ├── padding.py              # circular + non-periodic ghost-cell padding
│   ├── data.py                 # random space-time window sampler
│   ├── training.py             # rollout loss, train_epoch, train_CFN
│   ├── theoretical.py          # analytical Whitham flux reference
│   ├── solvers.py              # NumPy reference TVD-RK3 / Euler solver
│   └── __init__.py
├── scripts/
│   ├── train.py                # CLI training entrypoint
│   └── evaluate.py             # post-training plots + RMSE metrics
├── configs/
│   ├── default.yaml            # short run, 66 epochs
│   └── long.yaml               # long fine run, 666 epochs
├── data/
│   └── one_traj_loss_check.npy # bundled trajectory, shape (1, Nt, Nx, 1)
├── tests/
│   ├── test_models.py
│   └── test_padding.py
├── pyproject.toml              # installable as `cfn` package
├── requirements.txt
└── README.md
