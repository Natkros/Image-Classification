"""Score both models on the held-out test set and draw every chart used in the report and the app."""
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve)

from dataset import CLASSES, IMG_SIZE, CatDog, denormalize, get_dataloaders
from gradcam import explain, overlay
from model import DEVICE, MODELS, count_params, load_trained

COLORS = {"custom_cnn": "#e07a5f", "mobilenet_v2": "#3d85c6"}
VIZ = "visualizations"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": .25, "font.size": 10, "savefig.dpi": 160, "savefig.bbox": "tight"})


def predict(model, loader):
    """Returns (true labels, class probabilities) for every image in `loader`."""
    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            probs.append(model(x.to(DEVICE)).softmax(1).cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(probs)


def latency_ms(model, runs: int = 50) -> float:
    """Median time to classify a single image."""
    x, times = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE), []
    with torch.no_grad():
        for _ in range(runs + 5):  # first 5 runs are warm-up
            t = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - t)
    return float(np.median(times[5:]) * 1000)


def score(name: str, test_dl) -> dict:
    model = load_trained(name)
    y, probs = predict(model, test_dl)
    pred = probs.argmax(1)
    hist = json.load(open(f"saved_models/history_{name}.json"))
    return {"accuracy": round(100 * (pred == y).mean(), 2),
            "precision": round(100 * precision_score(y, pred, average="macro"), 2),
            "recall": round(100 * recall_score(y, pred, average="macro"), 2),
            "f1_score": round(100 * f1_score(y, pred, average="macro"), 2),
            "latency_ms": round(latency_ms(model), 2), "params": count_params(model),
            "train_seconds": hist["train_seconds"],
            "confusion_matrix": confusion_matrix(y, pred).tolist(),
            "roc": [a.tolist() for a in roc_curve(y, probs[:, 1])[:2]]}


def save(fig, name):
    os.makedirs(VIZ, exist_ok=True)
    fig.savefig(f"{VIZ}/{name}.png")
    plt.close(fig)


def plot_curves():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, label in MODELS.items():
        h = json.load(open(f"saved_models/history_{name}.json"))
        for ax, key in zip(axes, ("loss", "acc")):
            ep = range(1, len(h[f"val_{key}"]) + 1)
            ax.plot(ep, h[f"train_{key}"], "--", color=COLORS[name], alpha=.6, label=f"{label} train")
            ax.plot(ep, h[f"val_{key}"], color=COLORS[name], lw=2, label=f"{label} val")
    for ax, title in zip(axes, ("Loss", "Accuracy (%)")):
        ax.set(title=title, xlabel="Epoch")
    axes[0].legend()
    save(fig, "training_curves")


def plot_confusion(results):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (name, r) in zip(axes, results.items()):
        cm = np.array(r["confusion_matrix"])
        ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        ax.set(title=f"{MODELS[name]} ({r['accuracy']:.1f}%)", xlabel="Predicted", ylabel="Actual",
               xticks=[0, 1], yticks=[0, 1], xticklabels=CLASSES, yticklabels=CLASSES)
        ax.grid(False)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, v, ha="center", va="center", fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] / cm[i].sum() > .5 else "black")
    save(fig, "confusion_matrix")


def plot_roc(results):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, r in results.items():
        fpr, tpr = r["roc"]
        ax.plot(fpr, tpr, color=COLORS[name], lw=2, label=f"{MODELS[name]} (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k:", alpha=.5)
    ax.set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(loc="lower right")
    save(fig, "roc_curves")


def plot_comparison(results):
    panels = [("accuracy", "Accuracy (%)"), ("f1_score", "F1 (%)"), ("latency_ms", "Latency (ms)"), ("params", "Parameters (M)")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    for ax, (key, title) in zip(axes, panels):
        vals = [r[key] / (1e6 if key == "params" else 1) for r in results.values()]
        bars = ax.bar(list(MODELS.values()), vals, color=list(COLORS.values()), width=.55)
        ax.bar_label(bars, fmt="%.2f" if key in ("latency_ms", "params") else "%.1f", fontweight="bold")
        ax.set(title=title, ylim=(0, max(vals) * 1.2))
        ax.grid(axis="x", visible=False)
    save(fig, "model_comparison")


def plot_gradcam(name, test_dl, n=6):
    """Spread-out test images with the model's verdict (top) and where it looked (bottom)."""
    model, ds = load_trained(name), test_dl.dataset
    fig, axes = plt.subplots(2, n, figsize=(2.3 * n, 5.2))
    for col, i in enumerate(np.linspace(0, len(ds) - 1, n, dtype=int)):
        x, y = ds[i]
        probs, cam = explain(model, x[None].to(DEVICE))
        img, ok = denormalize(x), probs.argmax() == y
        axes[0, col].imshow(img)
        axes[0, col].set_title(f"{CLASSES[probs.argmax()]} {probs.max():.0%}\n(actual: {CLASSES[y]})",
                               fontsize=9, color="#2a9d4f" if ok else "#d62828", fontweight="bold")
        axes[1, col].imshow(overlay(img, cam))
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"{MODELS[name]}: predictions and Grad-CAM attention", fontweight="bold")
    save(fig, "gradcam_examples")


def save_samples(per_class: int = 3, folder: str = "sample_images"):
    """Full-size copies of a few real test photos for the demo app."""
    os.makedirs(folder, exist_ok=True)
    ds, seen = CatDog(False), {0: 0, 1: 0}
    for img, y in ((Image.fromarray(a), l) for a, l in zip(ds.images, ds.labels)):
        if seen[y] < per_class:
            seen[y] += 1
            img.resize((256, 256), Image.LANCZOS).save(f"{folder}/{CLASSES[y]}_{seen[y]}.png")


if __name__ == "__main__":
    _, _, test_dl, _ = get_dataloaders()
    results = {name: score(name, test_dl) for name in MODELS}
    json.dump(results, open("saved_models/evaluation_summary.json", "w"), indent=2)
    plot_curves(), plot_confusion(results), plot_roc(results), plot_comparison(results)
    plot_gradcam("mobilenet_v2", test_dl)
    save_samples()
    for name, r in results.items():
        print(f"{MODELS[name]:12s} acc {r['accuracy']:.2f}%  f1 {r['f1_score']:.2f}%  {r['latency_ms']:.1f} ms  {r['params']:,} params")
