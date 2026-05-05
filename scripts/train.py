"""Train a CFN on a precomputed trajectory.

Usage
-----
    python scripts/train.py --config configs/default.yaml
or
    python scripts/train.py --data path/to/traj.npy --epochs 66 --lr 2e-3 ...
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from cfn import CFN, train_CFN


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Conservative Flux Network.")
    p.add_argument("--config", type=Path, default=None, help="YAML config (overridden by CLI flags).")
    p.add_argument("--data", type=Path, help="Path to .npy trajectory of shape (1, Nt, Nx, 1).")
    p.add_argument("--out-dir", type=Path, default=Path("runs/run_0"))

    # Model / physics
    p.add_argument("--features", nargs="+", type=int, default=[64, 64, 64, 64, 1])
    p.add_argument("--dt", type=float, default=100 / 3200)
    p.add_argument("--dx", type=float, default=0.4)

    # Training
    p.add_argument("--epochs", type=int, default=66)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--early-stop-patience", type=int, default=60)
    p.add_argument("--lr-patience", type=int, default=10)
    p.add_argument("--tol", type=float, default=1e-9)
    p.add_argument("--margin", type=int, default=16)

    # Window sampling
    p.add_argument("--window-t", type=int, default=700)
    p.add_argument("--window-x", type=int, default=80)
    p.add_argument("--num-samples", type=int, default=126)
    p.add_argument("--L", type=int, default=160)
    p.add_argument("--rollout-steps", type=int, default=4)
    p.add_argument("--num-draws", type=int, default=3)

    # Smoothness regularisation
    p.add_argument("--use-smooth", action="store_true")
    p.add_argument("--lambda-smooth", type=float, default=1e-4)

    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def merge_config(args: argparse.Namespace) -> dict:
    """Load YAML if provided, then overlay any explicit CLI flags."""
    cfg: dict = {}
    if args.config is not None:
        cfg = yaml.safe_load(args.config.read_text()) or {}

    cli = vars(args)
    cli.pop("config", None)
    for k, v in cli.items():
        if v is not None:
            cfg[k] = v
    return cfg


def main() -> None:
    args = parse_args()
    cfg = merge_config(args)

    if "data" not in cfg or cfg["data"] is None:
        raise SystemExit("Must specify --data or 'data:' in the config file.")

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- data ----
    phi_xt = np.load(cfg["data"])
    print(f"Loaded data shape: {phi_xt.shape}")
    assert phi_xt.ndim == 4, f"Expected 4D trajectory, got {phi_xt.shape}"

    # ---- model ----
    model = CFN(features=list(cfg["features"]), dt=cfg["dt"], dx=cfg["dx"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.MSELoss()

    # Save initial model for later before/after comparisons
    torch.save(model.state_dict(), out_dir / "model_init.pt")

    trained, history = train_CFN(
        model,
        phi_xt,
        loss_fn,
        optimizer,
        device=device,
        max_epochs=cfg["epochs"],
        early_stop_patience=cfg["early_stop_patience"],
        lr_patience=cfg["lr_patience"],
        tol=cfg["tol"],
        margin=cfg["margin"],
        window_t=cfg["window_t"],
        window_x=cfg["window_x"],
        num_samples=cfg["num_samples"],
        L=cfg["L"],
        rollout_steps=cfg["rollout_steps"],
        num_draws=cfg["num_draws"],
        use_smooth=cfg["use_smooth"],
        lambda_smooth=cfg["lambda_smooth"],
    )

    torch.save(trained.state_dict(), out_dir / "model_trained.pt")
    np.save(out_dir / "loss_history.npy", np.asarray(history))
    print(f"Saved model + loss history to {out_dir}/")


if __name__ == "__main__":
    main()
