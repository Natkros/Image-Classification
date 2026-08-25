"""Training and evaluation loops."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, ReduceLROnPlateau, SequentialLR
from tqdm.auto import tqdm

from .data import compute_class_weights
from .model import param_groups, set_backbone_frozen
from .utils import (
    AverageMeter,
    EarlyStopping,
    History,
    count_parameters,
    format_seconds,
    save_checkpoint,
)


def _make_grad_scaler(device: torch.device, enabled: bool):
    """torch.amp.GradScaler landed in 2.4; fall back to the cuda-namespaced one."""
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


# --------------------------------------------------------------------------- #
# Optimiser / scheduler / loss construction
# --------------------------------------------------------------------------- #
def build_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    groups = param_groups(
        model,
        head_lr=float(cfg.train.lr),
        backbone_lr=float(cfg.get_path("train.backbone_lr", cfg.train.lr)),
    )
    weight_decay = float(cfg.get_path("train.weight_decay", 0.0))
    kind = str(cfg.get_path("train.optimizer", "adamw")).lower()

    if kind == "sgd":
        return SGD(groups, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    return AdamW(groups, weight_decay=weight_decay)


def build_scheduler(optimizer, cfg, steps_per_epoch: int = 1):
    """Optional linear warmup followed by cosine decay (or plateau, or nothing)."""
    kind = str(cfg.get_path("train.scheduler", "cosine")).lower()
    epochs = int(cfg.train.epochs)
    warmup = int(cfg.get_path("train.warmup_epochs", 0) or 0)

    if kind == "none":
        return None
    if kind == "plateau":
        return ReduceLROnPlateau(optimizer, mode="max", factor=0.3, patience=2)

    cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup))
    if warmup <= 0:
        return cosine

    warm = LambdaLR(optimizer, lr_lambda=lambda e: min(1.0, (e + 1) / max(1, warmup)))
    return SequentialLR(optimizer, [warm, cosine], milestones=[warmup])


def build_criterion(cfg, train_targets: Sequence[int] | None, num_classes: int, device) -> nn.Module:
    weight = None
    if cfg.get_path("train.class_weights", False) and train_targets is not None:
        weight = compute_class_weights(train_targets, num_classes).to(device)
    return nn.CrossEntropyLoss(
        weight=weight,
        label_smoothing=float(cfg.get_path("train.label_smoothing", 0.0)),
    )


# --------------------------------------------------------------------------- #
# One epoch
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, grad_clip=0.0, desc="train"):
    model.train()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    use_amp = scaler is not None and scaler.is_enabled()

    progress = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch = targets.size(0)
        correct = (logits.argmax(1) == targets).sum().item()
        loss_meter.update(loss.item(), batch)
        acc_meter.update(correct / batch, batch)
        progress.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.4f}")

    return {"loss": loss_meter.avg, "acc": acc_meter.avg}


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="eval", return_outputs: bool = False):
    """Run the model over a loader; optionally return per-sample predictions."""
    model.eval()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for images, targets in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        batch = targets.size(0)
        loss_meter.update(loss.item(), batch)
        acc_meter.update((logits.argmax(1) == targets).sum().item() / batch, batch)

        if return_outputs:
            all_probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    result = {"loss": loss_meter.avg, "acc": acc_meter.avg}
    if return_outputs:
        probs = np.concatenate(all_probs)
        result["probs"] = probs
        result["preds"] = probs.argmax(1)
        result["targets"] = np.concatenate(all_targets)
    return result


# --------------------------------------------------------------------------- #
# Full training run
# --------------------------------------------------------------------------- #
def fit(model, loaders, cfg, device, classes, run_dir: Path, logger, train_targets=None) -> dict:
    """Train for cfg.train.epochs, checkpointing the best validation accuracy."""
    run_dir = Path(run_dir)
    if "val" in loaders:
        monitor_loader, monitor_name = loaders["val"], "val"
    else:
        # No validation split. Selecting the best epoch on the *test* set would leak it
        # into model selection and inflate the final number, so monitor the (un-augmented)
        # training data instead and say so loudly.
        monitor_loader = loaders.get("train_eval") or loaders["train"]
        monitor_name = "train_eval"
        logger.warning(
            "data.val_split is 0, so there is no validation set. Checkpoints will be "
            "selected on training accuracy, which overfits by construction. Set "
            "data.val_split=0.15 for an honest model-selection signal."
        )

    criterion = build_criterion(cfg, train_targets, len(classes), device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(loaders["train"]))

    use_amp = bool(cfg.get_path("train.amp", True)) and device.type == "cuda"
    scaler = _make_grad_scaler(device, use_amp)

    stopper = EarlyStopping(
        patience=int(cfg.get_path("train.early_stopping_patience", 0) or 0), mode="max"
    )
    history = History()
    unfreeze_at = cfg.get_path("model.unfreeze_at_epoch")
    grad_clip = float(cfg.get_path("train.grad_clip", 0.0) or 0.0)
    best_path = run_dir / "checkpoints" / "best.pt"
    best_acc = 0.0

    trainable, total = count_parameters(model)
    logger.info(
        f"Model {getattr(model, 'model_name', '?')} | {trainable:,} trainable / {total:,} total params"
    )
    logger.info(f"Device: {device} | AMP: {use_amp} | monitoring {monitor_name}_acc")

    start = time.time()
    for epoch in range(1, int(cfg.train.epochs) + 1):
        # Stage 2: unfreeze the backbone and rebuild the optimiser with two LR groups.
        if unfreeze_at and epoch == int(unfreeze_at) and cfg.get_path("model.freeze_backbone", False):
            set_backbone_frozen(model, False)
            optimizer = build_optimizer(model, cfg)
            scheduler = build_scheduler(optimizer, cfg)
            trainable, total = count_parameters(model)
            logger.info(f"Unfroze backbone at epoch {epoch} — {trainable:,} trainable params")

        epoch_start = time.time()
        train_metrics = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, scaler, grad_clip,
            desc=f"epoch {epoch}/{cfg.train.epochs}",
        )
        val_metrics = evaluate(model, monitor_loader, criterion, device, desc=monitor_name)

        # Capture the LR this epoch actually trained with, before the scheduler moves it on.
        lr_now = optimizer.param_groups[0]["lr"]

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics["acc"])
            else:
                scheduler.step()
        is_best = stopper.step(val_metrics["acc"])
        if is_best:
            best_acc = val_metrics["acc"]
            save_checkpoint(best_path, model, classes, cfg, epoch, val_metrics)

        if not cfg.get_path("output.save_best_only", True):
            save_checkpoint(run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt", model, classes, cfg, epoch, val_metrics)

        history.append(
            epoch=epoch,
            train_loss=train_metrics["loss"],
            train_acc=train_metrics["acc"],
            val_loss=val_metrics["loss"],
            val_acc=val_metrics["acc"],
            lr=lr_now,
            seconds=time.time() - epoch_start,
        )
        history.save(run_dir / "history.json")

        logger.info(
            f"epoch {epoch:>3}/{cfg.train.epochs} | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.4f} | "
            f"{monitor_name} loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.4f} | "
            f"lr {lr_now:.2e} | {format_seconds(time.time() - epoch_start)}"
            f"{'  <- best' if is_best else ''}"
        )

        if stopper.should_stop:
            logger.info(f"Early stopping at epoch {epoch} (no improvement for {stopper.patience} epochs)")
            break

    logger.info(f"Training finished in {format_seconds(time.time() - start)} | best {monitor_name}_acc {best_acc:.4f}")
    return {"history": history, "best_acc": best_acc, "best_checkpoint": best_path}
