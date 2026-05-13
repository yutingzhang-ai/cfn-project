"""Training loop for the Conservative Flux Network (rollout-only, with validation).

Changes from the previous version:
  - Removed the smoothness penalty entirely (rollout MSE is the only loss).
  - Added held-out window-based validation each epoch.
  - Scheduler (ReduceLROnPlateau) and early stopping both watch VAL loss
    instead of training loss; training-loss signal is too noisy under
    random-window resampling.
  - Added gradient clipping (norm 1.0 by default) for parity with neural-operator
    baselines (FNO/CNN) trained with rollout loss.
  - Best-by-val checkpoint selection (kept from previous version, now driven by val).
  - Rollout loss is now the MEAN over rollout_steps (was the sum). Magnitudes
    are now comparable across different rollout horizons. If you are porting
    hyperparameters from the previous (sum-based) version, divide ``tol`` by
    roughly ``rollout_steps`` to retain equivalent stopping behaviour.

Validation rationale
--------------------
Time-based splits on a non-stationary trajectory are unreliable: the held-out
future timesteps live in a different region of state space than training,
conflating overfitting with extrapolation. Window-based splits sample held-out
windows from the *same* distribution as training, so val loss is a clean
generalization signal.
"""

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


def _rollout_loss(
    model: nn.Module,
    un: torch.Tensor,
    u_np1: torch.Tensor,
    loss_fn: Callable,
    *,
    k: int,
    margin: int,
    rollout_steps: int,
    integrator: str,
) -> torch.Tensor:
    """Compute mean rollout MSE over ``rollout_steps`` future targets.

    For each target, advance the state ``k`` integrator steps then compare
    interior region (excluding ``margin`` boundary points) against the
    corresponding snapshot. Returns the per-step MEAN as a tensor (sum
    divided by ``rollout_steps``) so loss magnitudes stay comparable
    across different choices of ``rollout_steps``. The caller can choose
    to backprop or not.
    """
    step_fn = _resolve_step_fn(model, integrator)
    loss = torch.tensor(0.0, device=un.device)
    um = un

    for target_id in range(rollout_steps):
        for _ in range(k):
            um = step_fn(um)
        pred = um[:, margin:-margin, :]
        target = u_np1[:, target_id, margin:-margin, :]
        loss = loss + loss_fn(pred, target)

    return loss / rollout_steps


def apply_model(
    model: nn.Module,
    un: torch.Tensor,
    u_np1: torch.Tensor,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    *,
    is_training: bool = True,
    k: int = 160,
    margin: int = 20,
    rollout_steps: int | None = 9,
    integrator: str = "euler",
    grad_clip: float | None = 1.0,
) -> float:
    """Roll the model out and accumulate rollout MSE against future snapshots.

    If ``is_training`` is True, runs backprop, gradient clip, and optimizer step.
    If False, no grads are computed.

    Parameters
    ----------
    grad_clip : float or None
        If positive, clip gradient norm to this value before stepping.
        Ignored when ``is_training`` is False.

    Returns
    -------
    loss_value : float
        The mean rollout loss (per-step) as a Python float. Independent of
        ``rollout_steps`` magnitude, so loss values are directly comparable
        across different rollout horizons.
    """
    if rollout_steps is None:
        rollout_steps = int(u_np1.shape[1])

    if is_training:
        optimizer.zero_grad()
        loss = _rollout_loss(
            model, un, u_np1, loss_fn,
            k=k, margin=margin, rollout_steps=rollout_steps,
            integrator=integrator,
        )
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        return float(loss.detach().item())
    else:
        with torch.no_grad():
            loss = _rollout_loss(
                model, un, u_np1, loss_fn,
                k=k, margin=margin, rollout_steps=rollout_steps,
                integrator=integrator,
            )
        return float(loss.item())


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
    num_val_samples: int = 4,
    L: int = 160,
    margin: int = 20,
    rollout_steps: int = 9,
    num_draws: int = 5,
    integrator: str = "euler",
    grad_clip: float | None = 1.0,
) -> tuple[float, float]:
    """Run ``num_draws`` random-window draws and return (train_loss, val_loss).

    Each draw samples ``num_samples + num_val_samples`` windows from
    ``build_epoch_dataset``, then splits them: first ``num_samples`` for
    training (with backprop + grad clip), last ``num_val_samples`` for
    held-out evaluation (no grad). Both losses are averaged over draws.
    """
    total_train = 0.0
    total_val = 0.0

    for _ in range(num_draws):
        data = build_epoch_dataset(
            full_data,
            window_t=window_t,
            window_x=window_x,
            num_samples=num_samples + num_val_samples,
            L=L,
        )
        un_all = torch.tensor(data["un"], dtype=torch.float32, device=device)
        u_np1_all = torch.tensor(data["un_p1"], dtype=torch.float32, device=device)

        un_train, un_val = un_all[:num_samples], un_all[num_samples:]
        u_np1_train, u_np1_val = u_np1_all[:num_samples], u_np1_all[num_samples:]

        # Training step
        model.train()
        loss_t = apply_model(
            model, un_train, u_np1_train, loss_fn, optimizer,
            is_training=True,
            k=L, margin=margin, rollout_steps=rollout_steps,
            integrator=integrator, grad_clip=grad_clip,
        )
        total_train += loss_t

        # Validation step
        model.eval()
        loss_v = apply_model(
            model, un_val, u_np1_val, loss_fn, optimizer,
            is_training=False,
            k=L, margin=margin, rollout_steps=rollout_steps,
            integrator=integrator,
        )
        total_val += loss_v

    return total_train / num_draws, total_val / num_draws


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
    margin: int = 20,
    window_t: int = 16,
    window_x: int = 80,
    num_samples: int = 16,
    num_val_samples: int = 4,
    L: int = 160,
    rollout_steps: int = 9,
    num_draws: int = 5,
    integrator: str = "euler",
    grad_clip: float | None = 1.0,
    verbose: bool = True,
) -> tuple[nn.Module, dict, dict]:
    """Train CFN with rollout MSE, val-based scheduler + early stopping.

    Each epoch holds out ``num_val_samples`` random windows for validation.
    ReduceLROnPlateau and early stopping both react to val loss. Best-by-val
    checkpoint is restored before returning.

    Returns
    -------
    model : nn.Module
        With best-by-val weights loaded.
    history : dict
        ``history['train']`` and ``history['val']`` are lists of per-epoch losses.
    info : dict
        Summary metadata (best_val, epochs_run, final_lr, early_stopped, integrator).
    """
    best_model = copy.deepcopy(model)
    best_val = float("inf")
    no_improve = 0
    history: dict[str, list[float]] = {"train": [], "val": []}
    early_stopped = False
    epoch = 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience, threshold=tol
    )

    for epoch in range(max_epochs):
        train_loss, val_loss = train_epoch(
            model, full_data, optimizer, loss_fn,
            device=device,
            window_t=window_t, window_x=window_x,
            num_samples=num_samples, num_val_samples=num_val_samples,
            L=L, margin=margin,
            rollout_steps=rollout_steps, num_draws=num_draws,
            integrator=integrator, grad_clip=grad_clip,
        )
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        # Both scheduler and early-stop watch val loss
        scheduler.step(val_loss)

        if verbose:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"epoch={epoch:3d} | train={train_loss:.4e} | "
                  f"val={val_loss:.4e} | lr={current_lr:.2e}")

        if val_loss < best_val - tol:
            best_val = val_loss
            best_model = copy.deepcopy(model)
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= early_stop_patience:
            early_stopped = True
            if verbose:
                print(f"Early stopping at epoch {epoch} (best val={best_val:.4e})")
            break

    model.load_state_dict(best_model.state_dict())

    info = {
        "epochs_run": epoch + 1 if history["train"] else 0,
        "best_val": float(best_val) if best_val != float("inf") else None,
        "final_lr": float(optimizer.param_groups[0]["lr"]),
        "early_stopped": early_stopped,
        "integrator": integrator,
    }
    return model, history, info
