"""Training loop for CFN (rollout-only, with validation).

Default training design:
    AdamW + linear warmup + cosine decay + best-validation checkpointing.

The validation set is built once at the start of training with a fixed seed
and a configurable ``val_per_corner_fraction`` so that low-density but
high-importance regions (extrema, shock fronts, rarefaction tails) are
guaranteed to be sampled. This keeps the validation signal both stable
(fixed seed) and representative of the visited state range R that controls
the Theorem 1 bound.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import torch

from cfn.sampling import build_epoch_dataset


def _resolve_step_fn(model, integrator):
    name = integrator.lower()
    if name == "euler":
        return model.euler
    if name in ("rk3", "tvd_rk3", "tvdrk3"):
        return model.TVD_RK3
    raise ValueError(f"Unknown integrator {integrator!r}")


def _rollout_loss(model, un, u_np1, loss_fn, *, k, margin, rollout_steps, integrator):
    step_fn = _resolve_step_fn(model, integrator)
    per_step_tensors = []
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
        per_step_tensors.append(loss_fn(pred, target))
    loss = torch.stack(per_step_tensors).mean()
    per_step = [float(t.detach().item()) for t in per_step_tensors]
    return loss, per_step


def apply_model(model, un, u_np1, loss_fn, optimizer, *, is_training=True,
                k=6, margin=20, rollout_steps=9, integrator="euler", grad_clip=1.0):
    if rollout_steps is None:
        rollout_steps = int(u_np1.shape[1])

    if is_training:
        optimizer.zero_grad()
        loss, per_step = _rollout_loss(
            model, un, u_np1, loss_fn,
            k=k, margin=margin, rollout_steps=rollout_steps,
            integrator=integrator,
        )
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        return float(loss.detach().item()), per_step
    with torch.no_grad():
        loss, per_step = _rollout_loss(
            model, un, u_np1, loss_fn,
            k=k, margin=margin, rollout_steps=rollout_steps,
            integrator=integrator,
        )
    return float(loss.item()), per_step


def _make_fixed_validation_set(full_data, *, device,
                               window_t=80, window_x=160, num_val_samples=16,
                               L=6, seed=1234,
                               val_per_corner_fraction=0.1):
    """Build a frozen validation set with guaranteed corner coverage.

    Uses a dedicated ``np.random.Generator`` (not global numpy state) seeded
    by ``seed`` so the validation set is reproducible and independent of
    other RNG consumers. ``val_per_corner_fraction`` is forwarded to
    :func:`build_epoch_dataset` so that each of the ``2^C`` per-channel
    bounding-box corners receives a guaranteed share of the validation
    budget -- this is what gives the val set coverage of the visited
    range R that controls the Theorem 1 bound, rather than just uniform
    coverage of the trajectory volume.

    With ``num_val_samples=16``, ``C=1``, ``val_per_corner_fraction=0.1``:
        int(0.1 * 16) = 1 per corner * 2 corners = 2 importance + 14 uniform.
    For more robust coverage on scalar laws, bump either argument.
    """
    rng = np.random.default_rng(seed)
    data = build_epoch_dataset(
        full_data,
        window_t=window_t,
        window_x=window_x,
        num_samples=num_val_samples,
        L=L,
        rng=rng,
        per_corner_fraction=val_per_corner_fraction,
    )

    un_val = torch.tensor(data["un"], dtype=torch.float32, device=device)
    u_np1_val = torch.tensor(data["un_p1"], dtype=torch.float32, device=device)
    return un_val, u_np1_val


def train_epoch(model, full_data, optimizer, loss_fn, *, device,
                fixed_val_data,
                window_t=80, window_x=160, num_samples=16,
                L=6, margin=20, rollout_steps=9, num_draws=5,
                integrator="euler", grad_clip=1.0,
                train_per_corner_fraction=0.0):
    total_train = 0.0
    un_val, u_np1_val = fixed_val_data

    for _ in range(num_draws):
        data = build_epoch_dataset(
            full_data, window_t=window_t, window_x=window_x,
            num_samples=num_samples, L=L,
            per_corner_fraction=train_per_corner_fraction,
        )
        un_train = torch.tensor(data["un"], dtype=torch.float32, device=device)
        u_np1_train = torch.tensor(data["un_p1"], dtype=torch.float32, device=device)

        model.train()
        loss_t, _ = apply_model(model, un_train, u_np1_train, loss_fn, optimizer,
                                is_training=True, k=L, margin=margin,
                                rollout_steps=rollout_steps, integrator=integrator,
                                grad_clip=grad_clip)
        total_train += loss_t

    model.eval()
    loss_v, _ = apply_model(model, un_val, u_np1_val, loss_fn, optimizer,
                            is_training=False, k=L, margin=margin,
                            rollout_steps=rollout_steps, integrator=integrator)

    return total_train / num_draws, loss_v


def _warmup_cosine_lr(epoch, *, max_epochs, warmup_epochs, min_lr_ratio):
    if max_epochs <= 1:
        return 1.0
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)

    decay_epochs = max(1, max_epochs - warmup_epochs)
    progress = float(epoch - warmup_epochs) / float(decay_epochs)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def train_CFN(model, full_data, loss_fn, optimizer, *, device,
              max_epochs=200, early_stop_patience=20,
              warmup_epochs=None, warmup_ratio=0.05, min_lr_ratio=0.01,
              tol=1e-6, margin=20,
              window_t=80, window_x=160, num_samples=16, num_val_samples=16,
              val_seed=1234,
              val_per_corner_fraction=0.1,
              train_per_corner_fraction=0.0,
              L=6, rollout_steps=9, num_draws=5,
              integrator="euler", grad_clip=1.0,
              checkpoint_dir=None, checkpoint_every=100,
              verbose=True):
    """Train a CFN with rollout loss + warmup-cosine LR + best-val checkpointing.

    Parameters
    ----------
    val_per_corner_fraction : float, default 0.1
        Fraction of the (small) fixed validation set guaranteed to contain
        each per-channel bounding-box corner. Default 0.1 gives ~20% corner
        coverage for scalar laws (2 corners), ~40% for C=2 systems (4
        corners). Set to 0 to recover purely uniform validation sampling.
    train_per_corner_fraction : float, default 0.0
        Same idea for training-window sampling. Default 0 keeps training
        purely uniform; bump if training is starving the extrema relative
        to validation.

    Raises
    ------
    ValueError
        If ``rollout_steps`` exceeds the number of supervision targets that
        ``build_epoch_dataset`` will pack into each window. With time gap
        ``L`` between targets, a window of length ``window_t`` yields
        ``(window_t - 1) // L`` targets (the ``range(L, window_t, L)``
        slice), so ``rollout_steps`` cannot exceed that count.
    """
    # Fail fast on the only off-by-one that bites silently: a window too
    # short to hold the requested number of rollout targets would otherwise
    # IndexError deep inside _rollout_loss with an unhelpful message.
    n_targets_available = (window_t - 1) // L
    if rollout_steps > n_targets_available:
        raise ValueError(
            f"rollout_steps={rollout_steps} exceeds available targets "
            f"({n_targets_available}) for window_t={window_t}, L={L}. "
            f"Increase window_t (>= 1 + rollout_steps * L) or decrease "
            f"rollout_steps / L."
        )

    best_model = copy.deepcopy(model)
    best_val = float("inf")
    no_improve = 0
    history = {"train": [], "val": []}
    early_stopped = False
    epoch = 0

    checkpoint_path = None
    if checkpoint_dir is not None:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    fixed_val_data = _make_fixed_validation_set(
        full_data,
        device=device,
        window_t=window_t,
        window_x=window_x,
        num_val_samples=num_val_samples,
        L=L,
        seed=val_seed,
        val_per_corner_fraction=val_per_corner_fraction,
    )

    if warmup_epochs is None:
        warmup_epochs = max(1, int(warmup_ratio * max_epochs))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda e: _warmup_cosine_lr(
            e,
            max_epochs=max_epochs,
            warmup_epochs=warmup_epochs,
            min_lr_ratio=min_lr_ratio,
        ),
    )

    for epoch in range(max_epochs):
        train_loss, val_loss = train_epoch(
            model, full_data, optimizer, loss_fn, device=device,
            window_t=window_t, window_x=window_x,
            fixed_val_data=fixed_val_data,
            num_samples=num_samples,
            L=L, margin=margin, rollout_steps=rollout_steps,
            num_draws=num_draws, integrator=integrator, grad_clip=grad_clip,
            train_per_corner_fraction=train_per_corner_fraction,
        )
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        scheduler.step()
        if verbose:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"epoch={epoch:3d} | train={train_loss:.4e} | "
                  f"val={val_loss:.4e} | lr={current_lr:.2e}")

        if val_loss < best_val - tol:
            best_val = val_loss
            best_model = copy.deepcopy(model)
            no_improve = 0
            if checkpoint_path is not None:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "best_val": best_val,
                        "history": history,
                        "integrator": integrator,
                    },
                    checkpoint_path / "best_model.pt",
                )
        else:
            no_improve += 1

        if checkpoint_path is not None and checkpoint_every is not None and checkpoint_every > 0:
            if (epoch + 1) % checkpoint_every == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "best_val": best_val,
                        "history": history,
                        "integrator": integrator,
                    },
                    checkpoint_path / f"checkpoint_epoch_{epoch + 1:04d}.pt",
                )

        if no_improve >= early_stop_patience:
            early_stopped = True
            if verbose:
                print(f"Early stopping at epoch {epoch} (best val={best_val:.4e})")
            break

    if checkpoint_path is not None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history,
                "best_val": best_val,
                "integrator": integrator,
            },
            checkpoint_path / "last_model.pt",
        )

    model.load_state_dict(best_model.state_dict())
    info = {
        "epochs_run": epoch + 1 if history["train"] else 0,
        "best_val": float(best_val) if best_val != float("inf") else None,
        "final_lr": float(optimizer.param_groups[0]["lr"]),
        "early_stopped": early_stopped,
        "integrator": integrator,
        "scheduler": "linear_warmup_cosine_decay",
        "warmup_epochs": int(warmup_epochs),
        "min_lr_ratio": float(min_lr_ratio),
        "checkpoint_dir": str(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_every": int(checkpoint_every) if checkpoint_every is not None else None,
        "num_val_samples": int(num_val_samples),
        "val_seed": int(val_seed),
        "val_per_corner_fraction": float(val_per_corner_fraction),
        "train_per_corner_fraction": float(train_per_corner_fraction),
    }
    return model, history, info
