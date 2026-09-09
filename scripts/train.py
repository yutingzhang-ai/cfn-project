"""Train a CFN on a precomputed trajectory.

Usage
-----
    python scripts/train.py --config configs/default.yaml
or
    python scripts/train.py --data path/to/traj.npy --epochs 66 --lr 2e-3 ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from cfn import CFN, theoretical, train_CFN


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Conservative Flux Network.")
    p.add_argument("--config", type=Path, default=None, help="YAML config (overridden by CLI flags).")
    p.add_argument("--data", type=Path, help="Path to .npy trajectory of shape (1, Nt, Nx, 1).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory. Default: runs/run_0 (or from config).")

    # Equation / boundary
    p.add_argument(
        "--equation",
        choices=["whitham", "burgers", "saint_venant"],
        default=None,
        help="Which analytical flux to learn against. Default: whitham.",
    )
    p.add_argument(
        "--boundary-mode",
        choices=["extrapolate", "outflow", "reflect", "periodic"],
        default=None,
        help="Padding mode for the flux network. Default: extrapolate (Whitham).",
    )

    # Model / physics
    p.add_argument("--features", nargs="+", type=int, default=None)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--dx", type=float, default=None)

    # Training
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--early-stop-patience", type=int, default=None)
    p.add_argument("--lr-patience", type=int, default=None)
    p.add_argument("--tol", type=float, default=None)
    p.add_argument("--margin", type=int, default=None)

    # Window sampling
    p.add_argument("--window-t", type=int, default=None)
    p.add_argument("--window-x", type=int, default=None)
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--L", type=int, default=None)
    p.add_argument("--rollout-steps", type=int, default=None)
    p.add_argument("--num-draws", type=int, default=None)

    # Smoothness regularisation
    p.add_argument("--use-smooth", action="store_true", default=None)
    p.add_argument("--lambda-smooth", type=float, default=None)

    p.add_argument("--integrator", choices=["euler", "rk3"], default=None,
                   help="Per-step integrator used during training rollouts.")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


# Hard defaults applied only when neither config nor CLI provide a value.
_DEFAULTS = {
    "equation": "whitham",
    "boundary_mode": "extrapolate",
    "features": [64, 64, 64, 64, 1],
    "dt": 100 / 3200,
    "dx": 0.4,
    "epochs": 66,
    "lr": 2e-3,
    "early_stop_patience": 60,
    "lr_patience": 10,
    "tol": 1e-9,
    "margin": 16,
    "window_t": 700,
    "window_x": 80,
    "num_samples": 126,
    "L": 160,
    "rollout_steps": 4,
    "num_draws": 3,
    "use_smooth": False,
    "lambda_smooth": 1e-4,
    "integrator": "euler",
    "seed": 0,
    "out_dir": Path("runs/run_0"),
}


def merge_config(args: argparse.Namespace) -> dict:
    """Defaults < YAML < CLI."""
    cfg: dict = dict(_DEFAULTS)
    if args.config is not None:
        cfg.update(yaml.safe_load(args.config.read_text()) or {})

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

    # ---- Activate the right analytical flux BEFORE building the model.
    # solvers.py reads theoretical.flux_function at call time, so this also
    # affects the numpy reference solver used downstream.
    eq = theoretical.set_equation(cfg["equation"])
    print(f"Equation: {eq.name}  (n_components={eq.n_components}, "
          f"u in [{eq.u_min}, {eq.u_max}])")
    if eq.n_components > 1:
        raise SystemExit(
            f"Equation {eq.name!r} is a system (n_components={eq.n_components}). "
            "The scalar CFN cannot train on this yet."
        )

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_to_save = {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.items()}
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg_to_save, sort_keys=True))

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- data ----
    phi_xt = np.load(cfg["data"])
    print(f"Loaded data shape: {phi_xt.shape}")
    assert phi_xt.ndim == 4, f"Expected 4D trajectory, got {phi_xt.shape}"

    # ---- model ----
    model = CFN(
        features=list(cfg["features"]),
        dt=cfg["dt"],
        dx=cfg["dx"],
        boundary_mode=cfg["boundary_mode"],
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.MSELoss()

    torch.save(model.state_dict(), out_dir / "model_init.pt")

    trained, history, info = train_CFN(
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
        integrator=cfg.get("integrator", "euler"),
    )

    torch.save(trained.state_dict(), out_dir / "model_trained.pt")
    # history is {"train": [...], "val": [...]} from the current train_CFN.
    # Save train-loss as the old single-array format for back-compat with
    # evaluate.py's loss_history.png plot, plus the full dict alongside.
    if isinstance(history, dict):
        np.save(out_dir / "loss_history.npy", np.asarray(history["train"]))
        np.save(out_dir / "loss_history_full.npy", history, allow_pickle=True)
    else:
        np.save(out_dir / "loss_history.npy", np.asarray(history))
    (out_dir / "train_summary.json").write_text(json.dumps(info, indent=2))
    print(f"Saved model + loss history + summary to {out_dir}/")


if __name__ == "__main__":
    main()
