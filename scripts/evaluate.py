"""Evaluate a trained CFN against the theoretical Whitham flux.

Produces:
- Loss-vs-epoch plot
- Flux comparison (before / after / analytical)
- Multi-resolution flux + derivative comparison

Usage
-----
    python scripts/evaluate.py --run-dir runs/run_0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from cfn import CFN
from cfn.theoretical import flux_function


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--features", nargs="+", type=int, default=[64, 64, 64, 64, 1])
    p.add_argument("--dt", type=float, default=100 / 3200)
    p.add_argument("--dx", type=float, default=0.4)
    p.add_argument("--u-min", type=float, default=0.66)
    p.add_argument("--u-max", type=float, default=0.88)
    p.add_argument("--pad", type=int, default=3, help="Boundary points to crop in flux plots.")
    return p.parse_args()


def load_model(state_path: Path, features, dt, dx, device) -> CFN:
    model = CFN(features=list(features), dt=dt, dx=dx).to(device)
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pad = args.pad
    eps = 1e-4

    model_init = load_model(args.run_dir / "model_init.pt", args.features, args.dt, args.dx, device)
    model_trained = load_model(args.run_dir / "model_trained.pt", args.features, args.dt, args.dx, device)
    history = np.load(args.run_dir / "loss_history.npy")

    # ---- 1. Loss curve ----
    plt.figure()
    plt.plot(history, label="training loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.yscale("log")
    plt.title("Loss vs Epoch"); plt.legend()
    plt.savefig(args.run_dir / "loss_history.png", dpi=150)
    plt.close()

    # ---- 2. Single-resolution flux comparison ----
    u_vals = torch.linspace(args.u_min, args.u_max, 100, dtype=torch.float32, device=device).view(1, -1, 1)
    with torch.no_grad():
        flux_before_full = model_init.num_flux(u_vals).cpu().numpy().flatten()
        flux_after_full = model_trained.num_flux(u_vals).cpu().numpy().flatten()
    u_np_full = u_vals.cpu().numpy().flatten()

    # u has length 100 (cell centers); flux has length 101 (cell interfaces).
    # Crop `pad` from each side: u -> 94, flux -> 95. Trim flux by 1 to match u.
    u_np = u_np_full[pad:-pad]
    flux_before = flux_before_full[pad:-pad][:-1]
    flux_after = flux_after_full[pad:-pad][:-1]

    # Analytical flux, aligned at midpoint of "after" curve
    flux_true = flux_function(u_np)
    mid = len(flux_true) // 2
    flux_true = flux_true - (flux_true[mid] - flux_after[mid])

    dfdu_true = (flux_function(u_np + eps) - flux_function(u_np - eps)) / (2 * eps)
    dfdu_before = np.gradient(flux_before, u_np)
    dfdu_after = np.gradient(flux_after, u_np)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(u_np, flux_before, label="before")
    axes[0].plot(u_np, flux_after, label="after")
    axes[0].plot(u_np, flux_true, "--", label="analytical (aligned)")
    axes[0].set_title("Flux comparison (cropped)"); axes[0].legend()

    axes[1].plot(u_np, dfdu_before, label="df/du before")
    axes[1].plot(u_np, dfdu_after, label="df/du after")
    axes[1].plot(u_np, dfdu_true, "--", label="df/du analytical")
    axes[1].set_title("Derivative comparison (cropped)"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(args.run_dir / "flux_comparison.png", dpi=150)
    plt.close(fig)

    # Quantitative metrics
    err_flux = float(np.sqrt(np.mean((flux_after - flux_true) ** 2)))
    err_dfdu = float(np.sqrt(np.mean((dfdu_after - dfdu_true) ** 2)))
    print(f"Flux RMSE:       {err_flux:.4e}")
    print(f"Derivative RMSE: {err_dfdu:.4e}")

    # ---- 3. Multi-resolution comparison (autograd derivative) ----
    res_list = [30, 100, 300]
    fig, axes = plt.subplots(len(res_list), 2, figsize=(10, 4 * len(res_list)))

    for i, N in enumerate(res_list):
        u_vals = torch.linspace(args.u_min, args.u_max, N, dtype=torch.float32, device=device).view(1, -1, 1)
        u_vals.requires_grad_(True)
        flux = model_trained.num_flux(u_vals)
        dfdu_auto = torch.autograd.grad(flux.sum(), u_vals, create_graph=False)[0]

        u_np_full = u_vals.detach().cpu().numpy().flatten()
        flux_full = flux.detach().cpu().numpy().flatten()
        dfdu_full = dfdu_auto.detach().cpu().numpy().flatten()

        # u: length N (centers).  flux: length N+1 (interfaces).  dfdu_auto: length N.
        u_np = u_np_full[pad:-pad]
        flux_np = flux_full[pad:-pad][:-1]
        dfdu_auto_np = dfdu_full[pad:-pad]

        flux_true = flux_function(u_np)
        mid = len(flux_true) // 2
        flux_true = flux_true - (flux_true[mid] - flux_np[mid])

        dfdu_fd = np.gradient(flux_np, u_np)
        dfdu_true = (flux_function(u_np + eps) - flux_function(u_np - eps)) / (2 * eps)

        axes[i, 0].plot(u_np, flux_np, label="learned")
        axes[i, 0].plot(u_np, flux_true, "--", label="analytical")
        axes[i, 0].set_title(f"Flux (N={N})"); axes[i, 0].legend()

        axes[i, 1].plot(u_np, dfdu_auto_np, label="autograd")
        axes[i, 1].plot(u_np, dfdu_fd, label="finite diff")
        axes[i, 1].plot(u_np, dfdu_true, "--", label="analytical")
        axes[i, 1].set_title(f"Derivative (N={N})"); axes[i, 1].legend()

    fig.tight_layout()
    fig.savefig(args.run_dir / "flux_multires.png", dpi=150)
    plt.close(fig)

    print(f"Wrote evaluation plots to {args.run_dir}/")


if __name__ == "__main__":
    main()
