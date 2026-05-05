"""Evaluate a trained CFN against the theoretical Whitham flux.

Produces:
- Loss-vs-epoch plot
- Flux comparison (before / after / analytical)
- Multi-resolution flux + derivative comparison
- Rollout comparison vs. NumPy reference solver (if data path is in config)
- ``metrics.csv`` with flux/derivative RMSE per resolution

Usage
-----
    python scripts/evaluate.py --run-dir runs/run_0
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from cfn import CFN
from cfn.theoretical import flux_function


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--features",
        nargs="+",
        type=int,
        default=None,
        help="Override model widths; defaults to run_dir/config.yaml.",
    )
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--dx", type=float, default=None)
    p.add_argument("--u-min", type=float, default=0.66)
    p.add_argument("--u-max", type=float, default=0.88)
    p.add_argument("--pad", type=int, default=3, help="Boundary points to crop in flux plots.")
    p.add_argument(
        "--rollout-snapshots",
        type=int,
        default=4,
        help="Number of L-step rollout comparisons (set 0 to skip).",
    )
    return p.parse_args()


def load_run_config(run_dir: Path) -> dict:
    """Read ``run_dir/config.yaml`` if it exists, else return an empty dict."""
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return {}
    return yaml.safe_load(cfg_path.read_text()) or {}


def resolve_model_params(args: argparse.Namespace, run_cfg: dict) -> tuple[list[int], float, float]:
    """Pick model params from CLI flags first, then run_cfg, then sensible defaults."""
    features = args.features if args.features is not None else run_cfg.get("features", [64, 64, 64, 64, 1])
    dt = args.dt if args.dt is not None else float(run_cfg.get("dt", 100 / 3200))
    dx = args.dx if args.dx is not None else float(run_cfg.get("dx", 0.4))
    return list(features), dt, dx


def load_model(state_path: Path, features, dt, dx, device) -> CFN:
    model = CFN(features=list(features), dt=dt, dx=dx).to(device)
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.eval()
    return model


def crop_to_centers(u_full: np.ndarray, flux_full: np.ndarray, pad: int) -> tuple[np.ndarray, np.ndarray]:
    """Crop ``pad`` cells from each boundary and trim flux by 1 to match centers.

    ``u_full`` has length ``Nx`` (cell centers); ``flux_full`` has length ``Nx + 1``
    (cell interfaces).  This returns equal-length arrays of length ``Nx - 2*pad``,
    where each flux value is associated with the cell center to its left.  This
    is approximate but adequate for visual / RMSE comparison.
    """
    u = u_full[pad:-pad]
    f = flux_full[pad:-pad][:-1]
    return u, f


def shift_to_anchor(curve: np.ndarray, anchor_value: float, anchor_idx: int) -> np.ndarray:
    """Subtract a constant from ``curve`` so that ``curve[anchor_idx] == anchor_value``."""
    return curve - (curve[anchor_idx] - anchor_value)


def central_diff(f: callable, u: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Central finite-difference derivative of a scalar function ``f(u)``."""
    return (f(u + eps) - f(u - eps)) / (2 * eps)


def do_rollout_comparison(
    model: CFN,
    run_cfg: dict,
    dt: float,
    dx: float,
    n_snapshots: int,
    device: torch.device,
    out_path: Path,
) -> list[tuple[int, float, float]] | None:
    """Compare model TVD-RK3 rollout against the NumPy reference solver and data.

    The data trajectory is assumed to be sampled every model dt; the L-th
    snapshot in the trajectory therefore corresponds to L model time steps
    of integration.

    Returns
    -------
    list of (k, numpy_rmse_vs_data, cfn_rmse_vs_data) tuples, or None if the
    comparison was skipped.
    """
    from cfn.solvers import tvd_rk3_numpy

    if n_snapshots <= 0:
        print("Skipping rollout comparison (--rollout-snapshots <= 0).")
        return None

    data_path = run_cfg.get("data") if run_cfg else None
    if not data_path or not Path(data_path).exists():
        print("Skipping rollout comparison: no resolvable data path in config.yaml.")
        return None

    L = int(run_cfg.get("L", 160))
    margin = int(run_cfg.get("margin", 16))

    traj = np.load(data_path)
    if traj.ndim != 4:
        raise ValueError(f"Expected (1, Nt, Nx, 1) trajectory, got shape {traj.shape}")

    Nt = traj.shape[1]
    max_snap = (Nt - 1) // L
    if max_snap <= 0:
        print(f"Skipping rollout comparison: trajectory too short (Nt={Nt}, L={L}).")
        return None
    if n_snapshots > max_snap:
        print(f"Reducing --rollout-snapshots to {max_snap} (trajectory cap).")
        n_snapshots = max_snap

    u_np = traj[0, 0, :, 0].astype(np.float64)[None, :]
    u_torch = torch.tensor(traj[0, 0], dtype=torch.float32, device=device).unsqueeze(0)

    fig, axes = plt.subplots(n_snapshots, 1, figsize=(10, 2.8 * n_snapshots), squeeze=False)
    rmse_rows: list[tuple[int, float, float]] = []
    Nx = u_np.shape[1]
    x = np.arange(Nx)
    interior = slice(margin, Nx - margin) if margin > 0 else slice(None)

    for k in range(1, n_snapshots + 1):
        for _ in range(L):
            u_np = tvd_rk3_numpy(u_np, dt, dx)
            with torch.no_grad():
                u_torch = model.TVD_RK3(u_torch)

        u_cfn_np = u_torch[0, :, 0].cpu().numpy()
        u_data = traj[0, k * L, :, 0]

        ax = axes[k - 1, 0]
        ax.plot(x, u_data, "k-", lw=1.2, label="data")
        ax.plot(x, u_np[0], "b--", lw=1.0, label="numpy ref")
        ax.plot(x, u_cfn_np, "r:", lw=1.2, label="CFN")
        if margin > 0:
            ax.axvspan(0, margin, color="grey", alpha=0.1)
            ax.axvspan(Nx - margin, Nx, color="grey", alpha=0.1)
        ax.set_title(f"Rollout snapshot k={k}  (after {k * L} model steps)")
        ax.set_xlabel("cell index")
        ax.set_ylabel("u")
        ax.legend(loc="upper right", fontsize="x-small")

        r_np = float(np.sqrt(np.mean((u_np[0, interior] - u_data[interior]) ** 2)))
        r_cfn = float(np.sqrt(np.mean((u_cfn_np[interior] - u_data[interior]) ** 2)))
        rmse_rows.append((k, r_np, r_cfn))
        print(f"  rollout k={k:>2}: numpy_RMSE={r_np:.4e}  cfn_RMSE={r_cfn:.4e}")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return rmse_rows


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pad = args.pad
    eps = 1e-4

    run_cfg = load_run_config(args.run_dir)
    features, dt, dx = resolve_model_params(args, run_cfg)
    if run_cfg:
        print(f"Using config from {args.run_dir / 'config.yaml'} (features={features}, dt={dt}, dx={dx}).")
    else:
        print("No config.yaml in run dir; using CLI/default values.")

    model_init = load_model(args.run_dir / "model_init.pt", features, dt, dx, device)
    model_trained = load_model(args.run_dir / "model_trained.pt", features, dt, dx, device)
    history = np.load(args.run_dir / "loss_history.npy")

    # ---- 1. Loss curve ----
    plt.figure()
    plt.plot(history, label="training loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.savefig(args.run_dir / "loss_history.png", dpi=150)
    plt.close()

    # ---- 2. Single-resolution flux comparison ----
    u_vals = torch.linspace(args.u_min, args.u_max, 100, dtype=torch.float32, device=device).view(1, -1, 1)
    with torch.no_grad():
        flux_before_full = model_init.num_flux(u_vals).cpu().numpy().flatten()
        flux_after_full = model_trained.num_flux(u_vals).cpu().numpy().flatten()
    u_np_full = u_vals.cpu().numpy().flatten()

    u_np, flux_before = crop_to_centers(u_np_full, flux_before_full, pad)
    _, flux_after = crop_to_centers(u_np_full, flux_after_full, pad)

    # Choose a single global alignment anchor (midpoint of the trained-model curve)
    # and reuse it across every panel below.
    mid = len(u_np) // 2
    anchor_u = float(u_np[mid])
    anchor_F = float(flux_after[mid])
    print(f"Alignment anchor: u={anchor_u:.4f}, F_model={anchor_F:.4e}")

    flux_true = flux_function(u_np)
    flux_true = flux_true - (flux_function(np.array(anchor_u)) - anchor_F)

    dfdu_true = central_diff(flux_function, u_np, eps)
    dfdu_before = np.gradient(flux_before, u_np)
    dfdu_after = np.gradient(flux_after, u_np)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(u_np, flux_before, label="before")
    axes[0].plot(u_np, flux_after, label="after")
    axes[0].plot(u_np, flux_true, "--", label="analytical (aligned)")
    axes[0].set_title("Flux comparison (cropped)")
    axes[0].legend()

    axes[1].plot(u_np, dfdu_before, label="df/du before")
    axes[1].plot(u_np, dfdu_after, label="df/du after")
    axes[1].plot(u_np, dfdu_true, "--", label="df/du analytical")
    axes[1].set_title("Derivative comparison (cropped)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(args.run_dir / "flux_comparison.png", dpi=150)
    plt.close(fig)

    err_flux = float(np.sqrt(np.mean((flux_after - flux_true) ** 2)))
    err_dfdu = float(np.sqrt(np.mean((dfdu_after - dfdu_true) ** 2)))
    print(f"Flux RMSE:       {err_flux:.4e}")
    print(f"Derivative RMSE: {err_dfdu:.4e}")

    metrics_rows: list[tuple[str, int, float, float]] = [
        ("single", 100, err_flux, err_dfdu),
    ]

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

        u_np_n, flux_np = crop_to_centers(u_np_full, flux_full, pad)
        dfdu_auto_np = dfdu_full[pad:-pad]

        # Use the SAME global anchor as Section 2 for analytical-flux alignment.
        flux_true = flux_function(u_np_n) - (flux_function(np.array(anchor_u)) - anchor_F)

        dfdu_fd = np.gradient(flux_np, u_np_n)
        dfdu_true = central_diff(flux_function, u_np_n, eps)

        axes[i, 0].plot(u_np_n, flux_np, label="learned")
        axes[i, 0].plot(u_np_n, flux_true, "--", label="analytical")
        axes[i, 0].set_title(f"Flux (N={N})")
        axes[i, 0].legend()

        axes[i, 1].plot(u_np_n, dfdu_auto_np, label="autograd")
        axes[i, 1].plot(u_np_n, dfdu_fd, label="finite diff")
        axes[i, 1].plot(u_np_n, dfdu_true, "--", label="analytical")
        axes[i, 1].set_title(f"Derivative (N={N})")
        axes[i, 1].legend()

        rmse_flux = float(np.sqrt(np.mean((flux_np - flux_true) ** 2)))
        rmse_dfdu = float(np.sqrt(np.mean((dfdu_auto_np - dfdu_true) ** 2)))
        metrics_rows.append(("multires", N, rmse_flux, rmse_dfdu))

    fig.tight_layout()
    fig.savefig(args.run_dir / "flux_multires.png", dpi=150)
    plt.close(fig)

    # ---- 4. Rollout comparison vs. NumPy reference solver ----
    rollout_rows = do_rollout_comparison(
        model_trained,
        run_cfg,
        dt=dt,
        dx=dx,
        n_snapshots=args.rollout_snapshots,
        device=device,
        out_path=args.run_dir / "rollout_comparison.png",
    )

    # ---- 5. Persist metrics ----
    metrics_path = args.run_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "N_or_k", "metric_a", "metric_b", "notes"])
        for section, N, a, b in metrics_rows:
            label_a, label_b = "flux_rmse", "dfdu_rmse"
            writer.writerow([section, N, f"{a:.6e}", f"{b:.6e}", f"{label_a}/{label_b}"])
        if rollout_rows:
            for k, r_np_, r_cfn in rollout_rows:
                writer.writerow(
                    ["rollout", k, f"{r_np_:.6e}", f"{r_cfn:.6e}", "numpy_rmse/cfn_rmse"]
                )
    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote evaluation plots to {args.run_dir}/")


if __name__ == "__main__":
    main()
