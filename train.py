"""
Final Project: Image Classification - Training Pipeline
Author: Kritarth Saxena
GitHub: @Natkros

Trains the CustomCNN from scratch and fine-tunes MobileNetV2 with AdamW and Cosine Annealing.
"""
import argparse
import json
import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import get_dataloaders, prepare_dataset
from model import count_parameters, get_model


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    """
    Helper function to run either a training step or an evaluation pass.
    If an optimizer is provided, runs backprop with gradient clipping.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(is_train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if is_train:
                optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def run_training(model_name: str = "custom_cnn", epochs: int = 8, batch_size: int = 32, lr: float = 1e-3,
                 data_dir: str = "data", save_dir: str = "saved_models") -> dict:
    """Trains the chosen model and saves checkpoints whenever validation accuracy improves."""
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[TRAIN] {model_name} on {device} (Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr})")

    prepare_dataset(data_dir=data_dir)
    train_loader, val_loader, _, classes = get_dataloaders(data_dir=data_dir, batch_size=batch_size)
    model = get_model(model_name, num_classes=len(classes), pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    # Use a lower learning rate for transfer learning so we don't destroy pre-trained features
    base_lr = min(lr, 3e-4) if ("mobilenet" in model_name or "resnet" in model_name) else lr
    optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = {
        "model_name": model_name, "classes": classes, "epochs": epochs,
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "best_val_acc": 0.0
    }
    best_acc = 0.0
    best_path = os.path.join(save_dir, f"best_{model_name}.pth")
    start_t = time.time()

    for ep in range(1, epochs + 1):
        t_loss, t_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = run_epoch(model, val_loader, criterion, None, device)
        scheduler.step()

        history["train_loss"].append(round(t_loss, 4))
        history["train_acc"].append(round(t_acc * 100, 2))
        history["val_loss"].append(round(v_loss, 4))
        history["val_acc"].append(round(v_acc * 100, 2))

        marker = ""
        if v_acc > best_acc:
            best_acc = v_acc
            history["best_val_acc"] = round(v_acc * 100, 2)
            torch.save({"epoch": ep, "model_state_dict": model.state_dict(), "classes": classes}, best_path)
            marker = " [BEST SAVED]"

        print(f"Epoch [{ep:02d}/{epochs:02d}] Train Loss: {t_loss:.4f} Acc: {t_acc*100:5.1f}% | "
              f"Val Loss: {v_loss:.4f} Acc: {v_acc*100:5.1f}%{marker}")

    history["training_time_sec"] = round(time.time() - start_t, 2)
    with open(os.path.join(save_dir, f"history_{model_name}.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"[OK] Training complete. Saved model to: {best_path} (Best Val Acc: {history['best_val_acc']}%)")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train image classifier models")
    parser.add_argument("--model", default="all", choices=["custom_cnn", "mobilenet_v2", "all"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    models = ["custom_cnn", "mobilenet_v2"] if args.model == "all" else [args.model]
    for m in models:
        run_training(m, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
