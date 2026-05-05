"""Evaluate a trained CFN against the theoretical Whitham flux.

Changes vs. the original
------------------------
1. Rollout comparison uses **forward Euler** for both the NumPy reference
   and the CFN by default, matching the default training-time integrator.
   Pass --integrator rk3 to restore the previous behaviour.

2. Flux alignment subtracts the midpoint difference between learned and
   analytical curves. Both are sampled at the same u-grid (interface
   midpoints), so alignment is exact — no half-grid offset.
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

# --- args / config -------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--features", nargs="+", type=int, default=None,
                   help="Override model widths; defaults to run_dir/config.yaml.")
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--dx", type=float, default=None)
    p.add_argument("--u-min", type=float, default=0.66)
    p.add_argument("--u-max", type=float, default=0.88)
    p.add_argument("--pad", type=int, default=3)
    p.add_argument("--rollout-snapshots", type=int, default=4)
    p.add_argument("--integrator", choices=["euler", "rk3"], default="euler",
                   help="Integrator for rollout comparison (default: euler).")
    return p.parse_args()


def load_run_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return {}
    return yaml.safe_load(cfg_path.read_text()) or {}


def resolve_model_params(args, run_cfg):
    features = args.features if args.features is not None else run_cfg.get("features", [64, 64, 64, 64, 1])
    dt = args.dt if args.dt is not None else float(run_cfg.get("dt", 100 / 3200))
    dx = args.dx if args.dx is not None else float(run_cfg.get("dx", 0.4))
    return list(features), dt, dx


def load_model(state_path, features, dt, dx, device) -> CFN:
    model = CFN(features=list(features), dt=dt, dx=dx).to(device)
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.eval()
    return model


# --- cropping & alignment ------------------------------------------------

def crop_to_interface_midpoints(u_full, flux_full, pad):
    """Return (u_mid, flux) of equal length, sampled at interface midpoints.

    u_full has Nx cell centers; flux_full has Nx+1 interface values.
    After cropping pad from each side and trimming the trailing flux interface,
    we evaluate u at the midpoints of each retained cell pair so the two
    arrays share an identical u-grid.
    """
    u_cropped = u_full[pad:-pad]                 # length Nx - 2*pad
    flux_cropped = flux_full[pad:-pad][:-1]      # length Nx - 2*pad - 1
    u_mid = 0.5 * (u_cropped[:-1] + u_cropped[1:])  # length Nx - 2*pad - 1
    return u_mid, flux_cropped


def align_by_midpoint(learned, reference):
    """Shift `reference` so it equals `learned` at the array midpoint."""
    if learned.shape != reference.shape:
        raise ValueError(f"shape mismatch: {learned.shape} vs {reference.shape}")
    mid = len(learned) // 2
    return reference - (reference[mid] - learned[mid]), mid


def central_diff(f, u, eps=1e-4):
    return (f(u + eps) - f(u - eps)) / (2 * eps)


# --- rollout -------------------------------------------------------------

def do_rollout_comparison(model, run_cfg, dt, dx, n_snapshots, device, out_path,
                          integrator="euler"):
    from cfn.solvers import euler_numpy, tvd_rk3_numpy

    if integrator == "euler":
        np_step, cfn_step = euler_numpy, model.euler
    elif integrator == "rk3":
        np_step, cfn_step = tvd_rk3_numpy, model.TVD_RK3
    else:
        raise ValueError(f"Unknown integrator {integrator!r}")

    if n_snapshots <= 0:
        print("Skipping rollout comparison (--rollout-snapshots <= 0).")
        return None

    data_path = run_cfg.get("data") if run_cfg else None
    if not data_path or not Path(data_path).exists():
        print("Skipping rollout comparison: no resolvable data path.")
        return None

    L = int(run_cfg.get("L", 160))
    margin = int(run_cfg.get("margin", 16))

    traj = np.load(data_path)
    if traj.ndim != 4:
        raise ValueError(f"Expected (1, Nt, Nx, 1), got {traj.shape}")

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
    rmse_rows = []
    Nx = u_np.shape[1]
    x = np.arange(Nx)
    interior = slice(margin, Nx - margin) if margin > 0 else slice(None)

    print(f"Rollout comparison using integrator={integrator!r}")
    for k in range(1, n_snapshots + 1):
        for _ in range(L):
            u_np = np_step(u_np, dt, dx)
            with torch.no_grad():
                u_torch = cfn_step(u_torch)

        u_cfn_np = u_torch[0, :, 0].cpu().numpy()
        u_data = traj[0, k * L, :, 0]

        ax = axes[k - 1, 0]
        ax.plot(x, u_data, "k-",  lw=1.2, label="data")
        ax.plot(x, u_np[0],"b--", lw=1.0, label=f"numpy ref ({integrator})")
        ax.plot(x, u_cfn_np,"r:", lw=1.2, label=f"CFN ({integrator})")
        if margin > 0:
            ax.axvspan(0, margin, color="grey", alpha=0.1)
            ax.axvspan(Nx - margin, Nx, color="grey", alpha=0.1)
        ax.set_title(f"Rollout snapshot k={k}  (after {k * L} model steps)")
        ax.set_xlabel("cell index")
        ax.set_ylabel("u")
        ax.legend(loc="upper right", fontsize="x-small")

        r_np  = float(np.sqrt(np.mean((u_np[0, interior] - u_data[interior]) ** 2)))
        r_cfn = float(np.sqrt(np.mean((u_cfn_np[interior]  - u_data[interior]) ** 2)))
        rmse_rows.append((k, r_np, r_cfn))
        print(f"  rollout k={k:>2}: numpy_RMSE={r_np:.4e}  cfn_RMSE={r_cfn:.4e}")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return rmse_rows


# --- main ----------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pad = args.pad
    eps = 1e-4

    run_cfg = load_run_config(args.run_dir)
    features, dt, dx = resolve_model_params(args, run_cfg)
    if run_cfg:
        print(f"Using config from {args.run_dir / 'config.yaml'} "
              f"(features={features}, dt={dt}, dx={dx}).")
    else:
        print("No config.yaml in run dir; using CLI/default values.")

    model_init    = load_model(args.run_dir / "model_init.pt",    features, dt, dx, device)
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
    u_centers = torch.linspace(args.u_min, args.u_max, 100,
                               dtype=torch.float32, device=device).view(1, -1, 1)
    with torch.no_grad():
        flux_before_full = model_init.num_flux(u_centers).cpu().numpy().flatten()
        flux_after_full  = model_trained.num_flux(u_centers).cpu().numpy().flatten()
    u_centers_np = u_centers.cpu().numpy().flatten()

    u_mid, flux_before = crop_to_interface_midpoints(u_centers_np, flux_before_full, pad)
    _,     flux_after  = crop_to_interface_midpoints(u_centers_np, flux_after_full,  pad)

    flux_true_raw = flux_function(u_mid)
    flux_true, mid_idx = align_by_midpoint(flux_after, flux_true_raw)
    anchor_u = float(u_mid[mid_idx])
    anchor_F = float(flux_after[mid_idx])
    print(f"Alignment anchor: u={anchor_u:.4f}, F_model={anchor_F:.4e}")

    dfdu_true   = central_diff(flux_function, u_mid, eps)
    dfdu_before = np.gradient(flux_before, u_mid)
    dfdu_after  = np.gradient(flux_after,  u_mid)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(u_mid, flux_before, label="before")
    axes[0].plot(u_mid, flux_after,  label="after")
    axes[0].plot(u_mid, flux_true, "--", label="analytical (aligned)")
    axes[0].axvline(anchor_u, color="grey", lw=0.5, alpha=0.5)
    axes[0].set_title("Flux comparison (cropped, midpoint-aligned)")
    axes[0].set_xlabel("u")
    axes[0].legend()

    axes[1].plot(u_mid, dfdu_before, label="df/du before")
    axes[1].plot(u_mid, dfdu_after,  label="df/du after")
    axes[1].plot(u_mid, dfdu_true, "--", label="df/du analytical")
    axes[1].set_title("Derivative comparison (cropped)")
    axes[1].set_xlabel("u")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(args.run_dir / "flux_comparison.png", dpi=150)
    plt.close(fig)

    err_flux = float(np.sqrt(np.mean((flux_after - flux_true) ** 2)))
    err_dfdu = float(np.sqrt(np.mean((dfdu_after - dfdu_true) ** 2)))
    print(f"Flux RMSE:       {err_flux:.4e}")
    print(f"Derivative RMSE: {err_dfdu:.4e}")

    metrics_rows = [("single", 100, err_flux, err_dfdu)]

    # ---- 3. Multi-resolution comparison ----
    res_list = [30, 100, 300]
    fig, axes = plt.subplots(len(res_list), 2, figsize=(10, 4 * len(res_list)))

    for i, N in enumerate(res_list):
        u_centers = torch.linspace(args.u_min, args.u_max, N,
                                   dtype=torch.float32, device=device).view(1, -1, 1)
        u_centers.requires_grad_(True)
        flux = model_trained.num_flux(u_centers)
        dfdu_auto = torch.autograd.grad(flux.sum(), u_centers, create_graph=False)[0]

        u_centers_np = u_centers.detach().cpu().numpy().flatten()
        flux_full    = flux.detach().cpu().numpy().flatten()
        dfdu_full    = dfdu_auto.detach().cpu().numpy().flatten()

        # Flux on interface midpoints (length N - 2*pad - 1).
        u_mid_n, flux_np = crop_to_interface_midpoints(u_centers_np, flux_full, pad)

        # dF/du from autograd lives at cell centers; average to interface midpoints.
        dfdu_centers = dfdu_full[pad:-pad]                              # length N - 2*pad
        dfdu_auto_np = 0.5 * (dfdu_centers[:-1] + dfdu_centers[1:])     # length N - 2*pad - 1

        flux_true_raw = flux_function(u_mid_n)
        flux_true, _ = align_by_midpoint(flux_np, flux_true_raw)

        dfdu_fd   = np.gradient(flux_np, u_mid_n)
        dfdu_true = central_diff(flux_function, u_mid_n, eps)

        axes[i, 0].plot(u_mid_n, flux_np,    label="learned")
        axes[i, 0].plot(u_mid_n, flux_true, "--", label="analytical")
        axes[i, 0].set_title(f"Flux (N={N})")
        axes[i, 0].set_xlabel("u")
        axes[i, 0].legend()

        axes[i, 1].plot(u_mid_n, dfdu_auto_np, label="autograd")
        axes[i, 1].plot(u_mid_n, dfdu_fd,      label="finite diff")
        axes[i, 1].plot(u_mid_n, dfdu_true, "--", label="analytical")
        axes[i, 1].set_title(f"Derivative (N={N})")
        axes[i, 1].set_xlabel("u")
        axes[i, 1].legend()

        rmse_flux = float(np.sqrt(np.mean((flux_np    - flux_true) ** 2)))
        rmse_dfdu = float(np.sqrt(np.mean((dfdu_auto_np - dfdu_true) ** 2)))
        metrics_rows.append(("multires", N, rmse_flux, rmse_dfdu))

    fig.tight_layout()
    fig.savefig(args.run_dir / "flux_multires.png", dpi=150)
    plt.close(fig)

    # ---- 4. Rollout comparison ----
    rollout_rows = do_rollout_comparison(
        model_trained, run_cfg, dt=dt, dx=dx,
        n_snapshots=args.rollout_snapshots, device=device,
        out_path=args.run_dir / "rollout_comparison.png",
        integrator=args.integrator,
    )

    # ---- 5. Persist metrics ----
    metrics_path = args.run_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "N_or_k", "metric_a", "metric_b", "notes"])
        for section, N, a, b in metrics_rows:
            writer.writerow([section, N, f"{a:.6e}", f"{b:.6e}", "flux_rmse/dfdu_rmse"])
        if rollout_rows:
            for k, r_np_, r_cfn in rollout_rows:
                writer.writerow(["rollout", k, f"{r_np_:.6e}", f"{r_cfn:.6e}",
                                 f"numpy_rmse/cfn_rmse ({args.integrator})"])
    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote evaluation plots to {args.run_dir}/")


if __name__ == "__main__":
    main()
