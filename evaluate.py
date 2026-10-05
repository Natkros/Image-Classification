"""
Final Project: Image Classification - Evaluation & Visualizations
Author: Kritarth Saxena
GitHub: @Natkros

Computes accuracy, precision, recall, F1-score, latency, and confusion matrices.
Generates publication-quality charts for the project report and notebook.
"""
import json
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_curve, auc, precision_recall_curve

from dataset import denormalize_image, get_dataloaders, prepare_dataset
from model import count_parameters, get_model


def load_model_for_eval(name: str, path: str, num_classes: int, device):
    """Loads a trained model checkpoint for evaluation."""
    model = get_model(name, num_classes=num_classes, pretrained=False)
    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


def evaluate_model_on_test(model: torch.nn.Module, test_loader, device):
    """Runs inference across the test loader and computes standard classification metrics."""
    y_true, y_pred, y_probs, latencies = [], [], [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            t0 = time.time()
            outputs = model(imgs.to(device))
            latencies.append((time.time() - t0) / imgs.size(0))
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(probs.argmax(1))
            y_probs.extend(probs)

    y_t, y_p, y_pr = np.array(y_true), np.array(y_pred), np.array(y_probs)
    return {
        "accuracy": float(accuracy_score(y_t, y_p)),
        "precision": float(precision_score(y_t, y_p, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_t, y_p, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_t, y_p, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_t, y_p).tolist(),
        "avg_latency_ms": round(float(np.mean(latencies) * 1000), 2),
        "y_true": y_t, "y_pred": y_p, "y_probs": y_pr
    }


def plot_training_curves(history_files: dict, save_path: str = "visualizations/training_curves.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=200)
    colors = {"custom_cnn": "#1e3c72", "mobilenet_v2": "#2e7d32"}
    for name, path in history_files.items():
        if os.path.exists(path):
            with open(path) as f:
                h = json.load(f)
            ep = range(1, len(h["train_loss"]) + 1)
            c = colors.get(name, "#333")
            lbl = "Custom CNN" if "custom" in name else "MobileNetV2"
            ax1.plot(ep, h["train_loss"], "--", color=c, alpha=0.6, label=f"{lbl} Train")
            ax1.plot(ep, h["val_loss"], "-", color=c, lw=2, label=f"{lbl} Val")
            ax2.plot(ep, h["train_acc"], "--", color=c, alpha=0.6, label=f"{lbl} Train")
            ax2.plot(ep, h["val_acc"], "-", color=c, lw=2, label=f"{lbl} Val")

    ax1.set_title("Loss Convergence", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.grid(True, alpha=0.4)
    ax1.legend()
    ax2.set_title("Accuracy Trajectory (%)", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.grid(True, alpha=0.4)
    ax2.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrices(results: dict, classes: list, save_path: str = "visualizations/confusion_matrix.png"):
    fig, axes = plt.subplots(1, len(results), figsize=(5.5 * len(results), 4.5), dpi=200)
    if len(results) == 1:
        axes = [axes]
    for idx, (name, res) in enumerate(results.items()):
        cm = np.array(res["confusion_matrix"])
        norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        ax = axes[idx]
        im = ax.imshow(norm, cmap=plt.cm.Blues)
        lbl = "Custom CNN" if "custom" in name else "MobileNetV2"
        ax.set_title(f"{lbl} (Acc: {res['accuracy']*100:.1f}%)", fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels([c.title() for c in classes])
        ax.set_yticklabels([c.title() for c in classes])
        ax.set_ylabel("Actual Label")
        ax.set_xlabel("Predicted Label")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]*100:.1f}%)", ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black", fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_pr_curves(results: dict, save_path: str = "visualizations/roc_pr_curves.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=200)
    colors = {"custom_cnn": "#1e3c72", "mobilenet_v2": "#2e7d32"}
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_probs"][:, 1])
        p, r, _ = precision_recall_curve(res["y_true"], res["y_probs"][:, 1])
        lbl = "Custom CNN" if "custom" in name else "MobileNetV2"
        ax1.plot(fpr, tpr, color=colors.get(name, "#333"), lw=2, label=f"{lbl} (AUC={auc(fpr, tpr):.3f})")
        ax2.plot(r, p, color=colors.get(name, "#333"), lw=2, label=f"{lbl} (PR-AUC={auc(r, p):.3f})")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax1.set_title("ROC Curve", fontweight="bold")
    ax1.grid(True, alpha=0.4)
    ax1.legend()
    ax2.set_title("Precision-Recall Curve", fontweight="bold")
    ax2.grid(True, alpha=0.4)
    ax2.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_sample_predictions(model, loader, classes, device, save_path: str = "visualizations/sample_predictions.png", n: int = 8):
    imgs, labels = next(iter(loader))
    imgs, labels = imgs[:n], labels[:n]
    with torch.no_grad():
        probs = F.softmax(model(imgs.to(device)), dim=1).cpu().numpy()
    preds = probs.argmax(1)

    fig, axes = plt.subplots(2, 4, figsize=(13, 6), dpi=200)
    for idx, ax in enumerate(axes.flatten()):
        ax.imshow(denormalize_image(imgs[idx]))
        ax.axis("off")
        corr = (labels[idx].item() == preds[idx])
        col = "#1b5e20" if corr else "#b71c1c"
        ax.set_title(f"{'PASS' if corr else 'FAIL'}: {classes[preds[idx]].title()} ({probs[idx][preds[idx]]*100:.1f}%)\nTrue: {classes[labels[idx]].title()}",
                     fontsize=9, color=col, fontweight="bold", pad=4)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_benchmark_comparison(results: dict, save_path: str = "visualizations/model_comparison_bar.png"):
    names = ["Custom CNN", "MobileNetV2"]
    keys = list(results.keys())
    acc = [results[k]["accuracy"] * 100 for k in keys]
    f1 = [results[k]["f1_score"] * 100 for k in keys]
    lat = [results[k]["avg_latency_ms"] for k in keys]
    params = [results[k]["param_count"] / 1e6 for k in keys]

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), dpi=200)
    metrics = [("Accuracy (%)", acc, 105), ("F1-Score (%)", f1, 105), ("Params (M)", params, max(params) * 1.3), ("Latency (ms)", lat, max(lat) * 1.4)]
    colors = ["#1e3c72", "#2e7d32"]
    for idx, (title, vals, y_max) in enumerate(metrics):
        bars = axes[idx].bar(names, vals, color=colors, width=0.5)
        axes[idx].set_title(title, fontweight="bold")
        axes[idx].set_ylim(0, y_max)
        axes[idx].grid(axis="y", alpha=0.4)
        for b in bars:
            h = b.get_height()
            axes[idx].text(b.get_x() + b.get_width() / 2, h + 1, f"{h:.2f}" if "Params" in title or "Latency" in title else f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def run_full_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepare_dataset()
    _, _, test_loader, classes = get_dataloaders()
    models = [("custom_cnn", "saved_models/best_custom_cnn.pth"), ("mobilenet_v2", "saved_models/best_mobilenet_v2.pth")]
    results = {}
    for name, path in models:
        m = load_model_for_eval(name, path, len(classes), device)
        res = evaluate_model_on_test(m, test_loader, device)
        res["param_count"] = count_parameters(m)["total_params"]
        results[name] = res

    os.makedirs("visualizations", exist_ok=True)
    plot_training_curves({"custom_cnn": "saved_models/history_custom_cnn.json", "mobilenet_v2": "saved_models/history_mobilenet_v2.json"})
    plot_confusion_matrices(results, classes)
    plot_roc_pr_curves(results)
    best_m = load_model_for_eval("mobilenet_v2", "saved_models/best_mobilenet_v2.pth", len(classes), device)
    plot_sample_predictions(best_m, test_loader, classes, device)
    plot_benchmark_comparison(results)

    clean = {"models": {k: {m: round(v[m] * (100 if m in ["accuracy", "f1_score", "precision", "recall"] else 1), 2)
                            for m in ["accuracy", "precision", "recall", "f1_score", "avg_latency_ms", "param_count"]}
                        for k, v in results.items()}}
    with open("saved_models/evaluation_summary.json", "w") as f:
        json.dump(clean, f, indent=2)
    print("[OK] Evaluation and visualizations complete.")


if __name__ == "__main__":
    run_full_evaluation()
