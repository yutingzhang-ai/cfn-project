# cfn-project
Use Conservertaive Form Network to predict or recover modulaton equations

cfn-project/
├── cfn/                        # the package
│   ├── models.py               # CFN, Flux network, ResidualBlock
│   ├── padding.py              # circular + non-periodic ghost-cell padding
│   ├── data.py                 # random space-time window sampler
│   ├── training.py             # rollout loss, train_epoch, train_CFN
│   ├── theoretical.py          # analytical Whitham flux reference
│   ├── solvers.py              # NumPy reference TVD-RK3 / Euler solver
│   └── __init__.py
├── scripts/
│   ├── train.py                # CLI training entrypoint (YAML + flags)
│   └── evaluate.py             # post-training plots + RMSE metrics
├── configs/
│   ├── default.yaml            # short run (66 epochs)
│   └── long.yaml               # long fine run (666 epochs, smaller windows)
├── data/
│   └── one_traj_loss_check.npy # bundled (1, Nt, Nx, 1) trajectory
├── tests/
│   ├── test_models.py
│   └── test_padding.py
├── pyproject.toml              # installable as `cfn` package
├── requirements.txt
└── README.md
