"""Training loop for the Conservative Flux Network."""

from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from cfn.data import build_epoch_dataset


def _resolve_step_fn(model: nn.Module, integrator: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the model's per-step function for the named integrator."""
    name = integrator.lower()
    if name == "euler":
        return model.euler
    if name in ("rk3", "tvd_rk3", "tvdrk3"):
        return model.TVD_RK3
    raise ValueError(f"Unknown integrator {integrator!r}; expected 'euler' or 'rk3'.")


def apply_model(
    model: nn.Module,
    un: torch.Tensor,
    u_np1: torch.Tensor,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    lr_scheduler=None,
    *,
    is_training: bool = True,
    k: int = 160,
    margin: int = 20,
    lambda_smooth: float = 0.0,
    use_smooth: bool = False,
    rollout_steps: int | None = 9,
    integrator: str = "euler",
) -> tuple[float, list[float]]:
    """Roll the model out and accumulate loss against future snapshots.

    For each of ``rollout_steps`` targets, take ``k`` integrator steps from
    the current state, compare interior region (excluding ``margin`` points
    on each boundary) against the matching snapshot, and accumulate MSE.

    Parameters
    ----------
    use_smooth : bool
        If True, add ``lambda_smooth * mean((d/dx)(dF/du))^2`` per step.
    rollout_steps : int or None
        Number of supervised future points; ``None`` uses ``u_np1.shape[1]``.
    integrator : {"euler", "rk3"}
        Per-step time integrator. ``"rk3"`` uses TVD-RK3 (Shu-Osher).
    """
    optimizer.zero_grad()

    if rollout_steps is None:
        rollout_steps = int(u_np1.shape[1])

    step_fn = _resolve_step_fn(model, integrator)

    loss = torch.tensor(0.0, device=un.device)
    loss_list: list[float] = []
    um = un

    for target_id in range(rollout_steps):
        for _ in range(k):
            um = step_fn(um)

        pred = um[:, margin:-margin, :]
        target = u_np1[:, target_id, margin:-margin, :]
        loss_i = loss_fn(pred, target)

        if use_smooth and lambda_smooth > 0:
            f = model.num_flux(um)
            grad_f = torch.autograd.grad(
                f, um,
                grad_outputs=torch.ones_like(f),
                create_graph=True,
                retain_graph=True,
            )[0][:, margin:-margin, :]
            smooth_loss = ((grad_f[:, 1:, :] - grad_f[:, :-1, :]) ** 2).mean()
            total_i = loss_i + lambda_smooth * smooth_loss
        else:
            total_i = loss_i

        loss = loss + total_i
        loss_list.append(loss_i.item())

    if is_training:
        loss.backward()
        optimizer.step()

    return float(loss.detach().item()), loss_list


def train_epoch(
    model: nn.Module,
    full_data: np.ndarray,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    *,
    device: torch.device,
    window_t: int = 16,
    window_x: int = 80,
    num_samples: int = 16,
    L: int = 160,
    margin: int = 20,
    lambda_smooth: float = 0.0,
    use_smooth: bool = False,
    rollout_steps: int = 9,
    num_draws: int = 5,
    integrator: str = "euler",
) -> float:
    """Run ``num_draws`` independent random-window draws as one 'epoch'."""
    model.train()
    total_loss = 0.0

    for _ in range(num_draws):
        data = build_epoch_dataset(
            full_data,
            window_t=window_t,
            window_x=window_x,
            num_samples=num_samples,
            L=L,
        )
        un_t = torch.tensor(data["un"], dtype=torch.float32, device=device)
        u_np1_t = torch.tensor(data["un_p1"], dtype=torch.float32, device=device)

        loss, _ = apply_model(
            model, un_t, u_np1_t, loss_fn, optimizer,
            is_training=True,
            k=L,
            margin=margin,
            lambda_smooth=lambda_smooth,
            use_smooth=use_smooth,
            rollout_steps=rollout_steps,
            integrator=integrator,
        )
        total_loss += loss

    return total_loss / num_draws


def train_CFN(
    model: nn.Module,
    full_data: np.ndarray,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    max_epochs: int = 200,
    early_stop_patience: int = 20,
    lr_patience: int = 5,
    tol: float = 1e-6,
    lr_factor: float = 0.5,
    use_smooth: bool = False,
    lambda_smooth: float = 1e-4,
    margin: int = 20,
    window_t: int = 16,
    window_x: int = 80,
    num_samples: int = 16,
    L: int = 160,
    rollout_steps: int = 9,
    num_draws: int = 5,
    integrator: str = "euler",
    verbose: bool = True,
) -> tuple[nn.Module, list[float], dict]:
    """Train CFN with ReduceLROnPlateau, early stopping, and auto-smooth.

    The smoothness penalty is automatically enabled the first time the
    learning rate is reduced — under the heuristic that the network is
    near-converged on data fit and should now regularise its flux derivative.

    Returns
    -------
    model, loss_history, info
        ``info`` is a dict with ``epochs_run``, ``best_loss``, ``final_lr``,
        ``use_smooth_enabled``, ``early_stopped`` — useful for run summaries.
    """
    best_model = copy.deepcopy(model)
    best_loss = float("inf")
    no_improve = 0
    loss_history: list[float] = []
    smooth_enabled_input = bool(use_smooth)
    early_stopped = False
    epoch = 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience, threshold=tol
    )
    prev_lr = optimizer.param_groups[0]["lr"]

    for epoch in range(max_epochs):
        loss_val = train_epoch(
            model, full_data, optimizer, loss_fn,
            device=device,
            window_t=window_t, window_x=window_x, num_samples=num_samples,
            L=L, margin=margin,
            lambda_smooth=lambda_smooth, use_smooth=use_smooth,
            rollout_steps=rollout_steps, num_draws=num_draws,
            integrator=integrator,
        )
        loss_history.append(loss_val)

        current_lr = optimizer.param_groups[0]["lr"]
        if verbose:
            print(f"epoch={epoch} loss={loss_val:.4e} lr={current_lr:.2e} smooth={use_smooth}")

        scheduler.step(loss_val)
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr < prev_lr:
            use_smooth = True
            if verbose:
                print("--> LR dropped: enabling smooth loss")
        prev_lr = current_lr

        if loss_val < best_loss - tol:
            best_loss = loss_val
            best_model = copy.deepcopy(model)
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= early_stop_patience:
            early_stopped = True
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_model.state_dict())

    info = {
        "epochs_run": epoch + 1 if loss_history else 0,
        "best_loss": float(best_loss) if best_loss != float("inf") else None,
        "final_lr": float(optimizer.param_groups[0]["lr"]),
        "use_smooth_input": smooth_enabled_input,
        "use_smooth_enabled": bool(use_smooth),
        "early_stopped": early_stopped,
        "integrator": integrator,
    }
    return model, loss_history, info
