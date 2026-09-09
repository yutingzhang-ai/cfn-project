"""Evaluate a trained CFN against the analytical flux for its equation.

Supports:
- multiple equations (Whitham, Burgers, ...) via cfg["equation"]
- multiple boundary modes via cfg["boundary_mode"]
- single default rollout dataset via cfg["data"]
- multiple rollout datasets via cfg["rollout_datasets"]
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from cfn import CFN, theoretical

# --- args / config -------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--features", nargs="+", type=int, default=None,
                   help="Override model widths; defaults to run_dir/config.yaml.")
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--dx", type=float, default=None)
    p.add_argument("--u-min", type=float, default=None,
                   help="Lower u sampling bound. Default: from active equation.")
    p.add_argument("--u-max", type=float, default=None,
                   help="Upper u sampling bound. Default: from active equation.")
    p.add_argument("--pad", type=int, default=3)
    p.add_argument("--rollout-snapshots", type=int, default=4)
    p.add_argument("--integrator", choices=["euler", "rk3"], default="euler",
                   help="Integrator for rollout comparison (default: euler).")
    p.add_argument("--equation", choices=["whitham", "burgers", "saint_venant"],
                   default=None,
                   help="Override equation; defaults to run_dir/config.yaml.")
    p.add_argument("--boundary-mode",
                   choices=["extrapolate", "outflow", "reflect", "periodic"],
                   default=None,
                   help="Override boundary mode; defaults to run_dir/config.yaml.")
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


def resolve_equation_and_boundary(args, run_cfg):
    """Pick the equation and boundary mode, preferring CLI > config > defaults."""
    equation = args.equation or run_cfg.get("equation", "whitham")
    boundary_mode = args.boundary_mode or run_cfg.get("boundary_mode", "extrapolate")
    return equation, boundary_mode


def resolve_u_range(args, equation_obj):
    """u_min/u_max: CLI overrides, otherwise equation defaults."""
    u_min = args.u_min if args.u_min is not None else equation_obj.u_min
    u_max = args.u_max if args.u_max is not None else equation_obj.u_max
    return float(u_min), float(u_max)


def load_model(state_path, features, dt, dx, device, boundary_mode) -> CFN:
    model = CFN(features=list(features), dt=dt, dx=dx,
                boundary_mode=boundary_mode).to(device)
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.eval()
    return model


def sanitize_filename(name: str) -> str:
    """Make a string safe for use as a filename component."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


# --- cropping & alignment ------------------------------------------------

def crop_to_interface_midpoints(
    u_full: np.ndarray, flux_full: np.ndarray, pad: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (u, flux) of equal length, with flux averaged from surrounding interfaces."""
    u_at_centers = u_full[pad:-pad]
    flux_at_centers = 0.5 * (flux_full[pad:-pad - 1] + flux_full[pad + 1:-pad])
    return u_at_centers, flux_at_centers


def align_by_midpoint(learned, reference):
    """Shift `reference` so it equals `learned` at the array midpoint."""
    if learned.shape != reference.shape:
        raise ValueError(f"shape mismatch: {learned.shape} vs {reference.shape}")
    mid = len(learned) // 2
    return reference - (reference[mid] - learned[mid]), mid


def central_diff(f, u, eps=1e-4):
    return (f(u + eps) - f(u - eps)) / (2 * eps)


# --- rollout -------------------------------------------------------------

def do_rollout_comparison(model, dt, dx, n_snapshots, device, out_path,
                          data_path, L, margin, boundary_mode="extrapolate",
                          integrator="euler"):
    """Roll out the trained model vs the numpy-reference solver vs the data."""
    from cfn.solvers import euler_numpy, tvd_rk3_numpy

    if integrator == "euler":
        np_step_fn, cfn_step = euler_numpy, model.euler
    elif integrator == "rk3":
        np_step_fn, cfn_step = tvd_rk3_numpy, model.TVD_RK3
    else:
        raise ValueError(f"Unknown integrator {integrator!r}")

    def np_step(u, dt, dx):
        return np_step_fn(u, dt, dx, boundary_mode=boundary_mode)

    if n_snapshots <= 0:
        print("Skipping rollout comparison (--rollout-snapshots <= 0).")
        return None

    data_path = Path(data_path)
    if not data_path.exists():
        print(f"Skipping rollout comparison: data path does not exist: {data_path}")
        return None

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

    print(f"Rollout comparison: integrator={integrator!r}, boundary={boundary_mode!r}, data={data_path}")
    for k in range(1, n_snapshots + 1):
        for _ in range(L):
            u_np = np_step(u_np, dt, dx)
            with torch.no_grad():
                u_torch = cfn_step(u_torch)

        u_cfn_np = u_torch[0, :, 0].cpu().numpy()
        u_data = traj[0, k * L, :, 0]

        ax = axes[k - 1, 0]
        ax.plot(x, u_data, "k-", lw=1.2, label="data")
        ax.plot(x, u_np[0], "b--", lw=1.0, label=f"numpy ref ({integrator})")
        ax.plot(x, u_cfn_np, "r:", lw=1.2, label=f"CFN ({integrator})")
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


def plot_rollout_summary(all_rollout_rows: dict, out_path: Path, integrator: str):
    """Plot CFN and NumPy-ref rollout RMSE vs k, one line per dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    cmap = plt.get_cmap("tab10")

    for i, (name, rows) in enumerate(all_rollout_rows.items()):
        if not rows:
            continue
        ks = [r[0] for r in rows]
        r_np = [r[1] for r in rows]
        r_cfn = [r[2] for r in rows]
        color = cmap(i % 10)
        axes[0].plot(ks, r_cfn, marker="o", color=color, label=name)
        axes[1].plot(ks, r_np, marker="s", color=color, label=name)

    axes[0].set_yscale("log")
    axes[0].set_xlabel("snapshot k")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title(f"CFN rollout RMSE ({integrator})")
    axes[0].legend(fontsize="small")
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].set_yscale("log")
    axes[1].set_xlabel("snapshot k")
    axes[1].set_title(f"NumPy reference rollout RMSE ({integrator})")
    axes[1].legend(fontsize="small")
    axes[1].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def get_rollout_specs(run_cfg: dict):
    """Return a normalized list of rollout dataset specs."""
    rollout_specs = run_cfg.get("rollout_datasets", None)

    if rollout_specs:
        normalized = []
        for i, spec in enumerate(rollout_specs, start=1):
            normalized.append({
                "name": spec.get("name", f"dataset_{i}"),
                "path": spec["path"],
                "L": int(spec.get("L", run_cfg.get("L", 160))),
                "margin": int(spec.get("margin", run_cfg.get("margin", 16))),
            })
        return normalized

    data_path = run_cfg.get("data", None)
    if data_path:
        return [{
            "name": "default",
            "path": data_path,
            "L": int(run_cfg.get("L", 160)),
            "margin": int(run_cfg.get("margin", 16)),
        }]

    return []


# --- main ----------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pad = args.pad
    eps = 1e-4

    run_cfg = load_run_config(args.run_dir)
    features, dt, dx = resolve_model_params(args, run_cfg)
    equation, boundary_mode = resolve_equation_and_boundary(args, run_cfg)

    # Activate the analytical flux BEFORE anything else touches flux_function
    # or the solvers. theoretical.flux_function is what gets used everywhere.
    eq = theoretical.set_equation(equation)
    u_min, u_max = resolve_u_range(args, eq)

    if eq.n_components > 1:
        raise SystemExit(
            f"Equation {eq.name!r} is a system (n_components={eq.n_components}). "
            "Scalar evaluate.py cannot handle it yet."
        )

    if run_cfg:
        print(f"Using config from {args.run_dir / 'config.yaml'}")
    else:
        print("No config.yaml in run dir; using CLI/default values.")
    print(f"  equation       = {eq.name}")
    print(f"  boundary_mode  = {boundary_mode}")
    print(f"  features       = {features}")
    print(f"  dt             = {dt}")
    print(f"  dx             = {dx}")
    print(f"  u range        = [{u_min}, {u_max}]")

    flux_function = theoretical.flux_function  # local alias for brevity

    model_init = load_model(args.run_dir / "model_init.pt",
                            features, dt, dx, device, boundary_mode)
    model_trained = load_model(args.run_dir / "model_trained.pt",
                               features, dt, dx, device, boundary_mode)
    # loss_history.npy may be a plain 1-D array (list format) or a
    # 0-d object array wrapping a dict ({"train":[], "val":[]}).
    _hist_raw = np.load(args.run_dir / "loss_history.npy", allow_pickle=True)
    if _hist_raw.ndim == 0:
        _h = _hist_raw.item()
        history = np.asarray(_h["train"] if isinstance(_h, dict) else _h)
    else:
        history = np.asarray(_hist_raw)

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
    u_centers = torch.linspace(u_min, u_max, 100,
                               dtype=torch.float32, device=device).view(1, -1, 1)
    with torch.no_grad():
        flux_before_full = model_init.num_flux(u_centers).cpu().numpy().flatten()
        flux_after_full = model_trained.num_flux(u_centers).cpu().numpy().flatten()
    u_centers_np = u_centers.cpu().numpy().flatten()

    u_mid, flux_before = crop_to_interface_midpoints(u_centers_np, flux_before_full, pad)
    _, flux_after = crop_to_interface_midpoints(u_centers_np, flux_after_full, pad)

    flux_true_raw = flux_function(u_mid)
    flux_true, mid_idx = align_by_midpoint(flux_after, flux_true_raw)
    anchor_u = float(u_mid[mid_idx])
    anchor_F = float(flux_after[mid_idx])
    print(f"Alignment anchor: u={anchor_u:.4f}, F_model={anchor_F:.4e}")

    dfdu_true = central_diff(flux_function, u_mid, eps)
    dfdu_before = np.gradient(flux_before, u_mid)
    dfdu_after = np.gradient(flux_after, u_mid)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(u_mid, flux_before, label="before")
    axes[0].plot(u_mid, flux_after, label="after")
    axes[0].plot(u_mid, flux_true, "--", label="analytical (aligned)")
    axes[0].axvline(anchor_u, color="grey", lw=0.5, alpha=0.5)
    axes[0].set_title(f"Flux comparison ({eq.name}, cropped, midpoint-aligned)")
    axes[0].set_xlabel("u")
    axes[0].legend()

    axes[1].plot(u_mid, dfdu_before, label="df/du before")
    axes[1].plot(u_mid, dfdu_after, label="df/du after")
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

    metrics_rows = [("single", "global", 100, err_flux, err_dfdu, "flux_rmse/dfdu_rmse")]

    # ---- 3. Multi-resolution comparison ----
    res_list = [30, 100, 300]
    fig, axes = plt.subplots(len(res_list), 2, figsize=(10, 4 * len(res_list)))

    for i, N in enumerate(res_list):
        u_centers = torch.linspace(u_min, u_max, N,
                                   dtype=torch.float32, device=device).view(1, -1, 1)
        u_centers.requires_grad_(True)
        flux = model_trained.num_flux(u_centers)
        dfdu_auto = torch.autograd.grad(flux.sum(), u_centers, create_graph=False)[0]

        u_centers_np = u_centers.detach().cpu().numpy().flatten()
        flux_full = flux.detach().cpu().numpy().flatten()
        dfdu_full = dfdu_auto.detach().cpu().numpy().flatten()

        u_mid_n, flux_np = crop_to_interface_midpoints(u_centers_np, flux_full, pad)
        dfdu_auto_np = dfdu_full[pad:-pad]

        flux_true_raw = flux_function(u_mid_n)
        flux_true, _ = align_by_midpoint(flux_np, flux_true_raw)

        dfdu_fd = np.gradient(flux_np, u_mid_n)
        dfdu_true = central_diff(flux_function, u_mid_n, eps)

        axes[i, 0].plot(u_mid_n, flux_np, label="learned")
        axes[i, 0].plot(u_mid_n, flux_true, "--", label="analytical")
        axes[i, 0].set_title(f"Flux (N={N})")
        axes[i, 0].set_xlabel("u")
        axes[i, 0].legend()

        axes[i, 1].plot(u_mid_n, dfdu_auto_np, label="autograd")
        axes[i, 1].plot(u_mid_n, dfdu_fd, label="finite diff")
        axes[i, 1].plot(u_mid_n, dfdu_true, "--", label="analytical")
        axes[i, 1].set_title(f"Derivative (N={N})")
        axes[i, 1].set_xlabel("u")
        axes[i, 1].legend()

        rmse_flux = float(np.sqrt(np.mean((flux_np - flux_true) ** 2)))
        rmse_dfdu = float(np.sqrt(np.mean((dfdu_auto_np - dfdu_true) ** 2)))
        metrics_rows.append(("multires", "global", N, rmse_flux, rmse_dfdu, "flux_rmse/dfdu_rmse"))

    fig.tight_layout()
    fig.savefig(args.run_dir / "flux_multires.png", dpi=150)
    plt.close(fig)

    # ---- 4. Rollout comparison for one or more datasets ----
    rollout_specs = get_rollout_specs(run_cfg)
    all_rollout_rows: dict = {}

    if not rollout_specs:
        print("No rollout datasets found in config; skipping rollout comparison.")
    else:
        for j, spec in enumerate(rollout_specs):
            dataset_name = spec["name"]
            data_path = spec["path"]
            L = spec["L"]
            margin = spec["margin"]

            safe_name = sanitize_filename(dataset_name)

            print(f"\n=== Evaluating rollout dataset: {dataset_name} ===")
            rollout_plot_path = args.run_dir / f"rollout_comparison_{safe_name}.png"

            rollout_rows = do_rollout_comparison(
                model_trained,
                dt=dt,
                dx=dx,
                n_snapshots=args.rollout_snapshots,
                device=device,
                out_path=rollout_plot_path,
                data_path=data_path,
                L=L,
                margin=margin,
                boundary_mode=boundary_mode,
                integrator=args.integrator,
            )

            if j == 0 and rollout_plot_path.exists():
                shutil.copyfile(rollout_plot_path, args.run_dir / "rollout_comparison.png")

            if rollout_rows:
                all_rollout_rows[dataset_name] = rollout_rows
                for k, r_np_, r_cfn in rollout_rows:
                    metrics_rows.append((
                        "rollout",
                        dataset_name,
                        k,
                        r_np_,
                        r_cfn,
                        f"numpy_rmse/cfn_rmse ({args.integrator})"
                    ))

        if all_rollout_rows:
            summary_path = args.run_dir / "rollout_rmse_summary.png"
            plot_rollout_summary(all_rollout_rows, summary_path, args.integrator)
            print(f"\nWrote rollout RMSE summary to {summary_path}")

    # ---- 5. Persist metrics ----
    metrics_path = args.run_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "dataset", "N_or_k", "metric_a", "metric_b", "notes"])
        for section, dataset, N, a, b, notes in metrics_rows:
            writer.writerow([section, dataset, N, f"{a:.6e}", f"{b:.6e}", notes])

    print(f"\nWrote metrics to {metrics_path}")
    print(f"Wrote evaluation plots to {args.run_dir}/")


if __name__ == "__main__":
    main()
