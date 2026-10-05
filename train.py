"""Train both models (AdamW + cosine LR) and keep the checkpoint with the best validation accuracy."""
import argparse
import json
import os
import time

import torch
import torch.nn as nn

from dataset import get_dataloaders
from model import DEVICE, MODELS, get_model

EPOCHS = {"custom_cnn": 20, "mobilenet_v2": 5}
LR = {"custom_cnn": 1e-3, "mobilenet_v2": 3e-4}  # gentler for the pretrained backbone


def run_epoch(model, loader, optimizer=None):
    """One pass over `loader`; trains when an optimizer is given. Returns (loss, accuracy %)."""
    model.train(optimizer is not None)
    loss_sum = correct = n = 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = nn.functional.cross_entropy(out, y)
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return loss_sum / n, 100 * correct / n


def train(name: str, epochs: int, save_dir: str = "saved_models") -> dict:
    os.makedirs(save_dir, exist_ok=True)
    train_dl, val_dl, _, _ = get_dataloaders()
    model = get_model(name).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR[name], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    best, start = 0.0, time.time()

    for ep in range(1, epochs + 1):
        for split, dl, o in (("train", train_dl, opt), ("val", val_dl, None)):
            loss, acc = run_epoch(model, dl, o)
            hist[f"{split}_loss"].append(round(loss, 4))
            hist[f"{split}_acc"].append(round(acc, 2))
        sched.step()
        saved = hist["val_acc"][-1] > best
        if saved:
            best = hist["val_acc"][-1]
            torch.save(model.state_dict(), f"{save_dir}/best_{name}.pth")
        print(f"[{MODELS[name]}] epoch {ep:02d}/{epochs}  train {hist['train_acc'][-1]:5.1f}%  "
              f"val {hist['val_acc'][-1]:5.1f}% (loss {hist['val_loss'][-1]:.3f}){'  *' if saved else ''}", flush=True)

    hist |= {"best_val_acc": best, "train_seconds": round(time.time() - start)}
    with open(f"{save_dir}/history_{name}.json", "w") as f:
        json.dump(hist, f, indent=2)
    return hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=[*MODELS, "all"], default="all")
    ap.add_argument("--epochs", type=int, help="override the per-model default")
    args = ap.parse_args()
    for m in MODELS if args.model == "all" else [args.model]:
        train(m, args.epochs or EPOCHS[m])
