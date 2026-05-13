"""Training loop for CFN (rollout-only, with validation)."""

from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from cfn.data import build_epoch_dataset


def _resolve_step_fn(model, integrator):
    name = integrator.lower()
    if name == "euler":
        return model.euler
    if name in ("rk3", "tvd_rk3", "tvdrk3"):
        return model.TVD_RK3
    raise ValueError(f"Unknown integrator {integrator!r}")


def _rollout_loss(model, un, u_np1, loss_fn, *, k, margin, rollout_steps, integrator):
    step_fn = _resolve_step_fn(model, integrator)
    loss = torch.tensor(0.0, device=un.device)
    um = un
    for target_id in range(rollout_steps):
        for _ in range(k):
            um = step_fn(um)
        if margin > 0:
            pred = um[:, margin:-margin, :]
            target = u_np1[:, target_id, margin:-margin, :]
        else:
            pred = um
            target = u_np1[:, target_id, :, :]
        loss = loss + loss_fn(pred, target)
    return loss / rollout_steps


def apply_model(model, un, u_np1, loss_fn, optimizer, *, is_training=True,
                k=160, margin=20, rollout_steps=9, integrator="euler", grad_clip=1.0):
    if rollout_steps is None:
        rollout_steps = int(u_np1.shape[1])

    if is_training:
        optimizer.zero_grad()
        loss = _rollout_loss(model, un, u_np1, loss_fn,
                             k=k, margin=margin, rollout_steps=rollout_steps,
                             integrator=integrator)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        return float(loss.detach().item())
    else:
        with torch.no_grad():
            loss = _rollout_loss(model, un, u_np1, loss_fn,
                                 k=k, margin=margin, rollout_steps=rollout_steps,
                                 integrator=integrator)
        return float(loss.item())


def train_epoch(model, full_data, optimizer, loss_fn, *, device,
                window_t=16, window_x=80, num_samples=16, num_val_samples=4,
                L=160, margin=20, rollout_steps=9, num_draws=5,
                integrator="euler", grad_clip=1.0):
    total_train = 0.0
    total_val = 0.0
    for _ in range(num_draws):
        data = build_epoch_dataset(
            full_data, window_t=window_t, window_x=window_x,
            num_samples=num_samples + num_val_samples, L=L,
        )
        un_all = torch.tensor(data["un"], dtype=torch.float32, device=device)
        u_np1_all = torch.tensor(data["un_p1"], dtype=torch.float32, device=device)

        un_train, un_val = un_all[:num_samples], un_all[num_samples:]
        u_np1_train, u_np1_val = u_np1_all[:num_samples], u_np1_all[num_samples:]

        model.train()
        loss_t = apply_model(model, un_train, u_np1_train, loss_fn, optimizer,
                             is_training=True, k=L, margin=margin,
                             rollout_steps=rollout_steps, integrator=integrator,
                             grad_clip=grad_clip)
        total_train += loss_t

        model.eval()
        loss_v = apply_model(model, un_val, u_np1_val, loss_fn, optimizer,
                             is_training=False, k=L, margin=margin,
                             rollout_steps=rollout_steps, integrator=integrator)
        total_val += loss_v
    return total_train / num_draws, total_val / num_draws


def train_CFN(model, full_data, loss_fn, optimizer, *, device,
              max_epochs=200, early_stop_patience=20, lr_patience=5,
              tol=1e-6, lr_factor=0.5, margin=20,
              window_t=16, window_x=80, num_samples=16, num_val_samples=4,
              L=160, rollout_steps=9, num_draws=5,
              integrator="euler", grad_clip=1.0, verbose=True):
    best_model = copy.deepcopy(model)
    best_val = float("inf")
    no_improve = 0
    history = {"train": [], "val": []}
    early_stopped = False
    epoch = 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience, threshold=tol
    )

    for epoch in range(max_epochs):
        train_loss, val_loss = train_epoch(
            model, full_data, optimizer, loss_fn, device=device,
            window_t=window_t, window_x=window_x,
            num_samples=num_samples, num_val_samples=num_val_samples,
            L=L, margin=margin, rollout_steps=rollout_steps,
            num_draws=num_draws, integrator=integrator, grad_clip=grad_clip,
        )
        history["train"].append(train_loss)
        history["val"].append(val_loss)

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
