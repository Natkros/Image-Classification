"""Plotting helpers.

Colour choices follow one rule set:
  * training curves = two identities (train / val) -> categorical slots 1 and 2,
    assigned in fixed order and never cycled; loss and accuracy get their own
    axes rather than being crammed onto one dual-scale plot.
  * confusion matrix = continuous magnitude -> a single-hue blue ramp, light to dark.
  * per-class accuracy = one series -> one colour, no legend, direct value labels.
Grid and axes stay recessive; text never wears the series colour.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# --- palette -------------------------------------------------------------- #
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8983"
GRID = "#e6e5e1"

# Single-hue sequential ramp (blue 100 -> 700) for magnitude encodings.
SEQUENTIAL_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]
BLUE_CMAP = LinearSegmentedColormap.from_list("ic_blue", SEQUENTIAL_BLUE)


def use_style() -> None:
    """Apply the project's matplotlib defaults. Call once per notebook/script."""
    if "ic_blue" not in plt.colormaps():
        mpl.colormaps.register(BLUE_CMAP, name="ic_blue")
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.bbox": "tight",
            "savefig.dpi": 150,
            "figure.dpi": 110,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 10,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": GRID,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 1.0,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "text.color": INK_PRIMARY,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "image.cmap": "ic_blue",
        }
    )


def _save(fig, path: str | Path | None):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    return fig


# --- charts --------------------------------------------------------------- #
def plot_history(history, save_path: str | Path | None = None):
    """Loss and accuracy over epochs, one axis each (never a dual y-scale)."""
    records = history.to_dict() if hasattr(history, "to_dict") else list(history)
    if not records:
        raise ValueError("History is empty — nothing to plot.")

    epochs = [r["epoch"] for r in records]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    panels = [
        (axes[0], "train_loss", "val_loss", "Loss", "Cross-entropy loss"),
        (axes[1], "train_acc", "val_acc", "Accuracy", "Accuracy"),
    ]
    for ax, train_key, val_key, title, ylabel in panels:
        train_vals = [r[train_key] for r in records]
        val_vals = [r[val_key] for r in records]
        ax.plot(epochs, train_vals, color=SERIES[0], label="train", marker="o")
        ax.plot(epochs, val_vals, color=SERIES[1], label="validation", marker="o")

        # Direct labels on the final point only — never a number on every marker.
        # If the two end values are close, nudge the labels apart so they cannot collide.
        span = max(train_vals + val_vals) - min(train_vals + val_vals) or 1.0
        close = abs(train_vals[-1] - val_vals[-1]) < 0.08 * span
        offsets = (8, -8) if close and train_vals[-1] >= val_vals[-1] else (
            (-8, 8) if close else (0, 0)
        )
        for values, dy in ((train_vals, offsets[0]), (val_vals, offsets[1])):
            ax.annotate(
                f"{values[-1]:.3f}",
                (epochs[-1], values[-1]),
                textcoords="offset points",
                xytext=(7, dy),
                color=INK_SECONDARY,
                fontsize=9,
                va="center",
            )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")
        ax.margins(x=0.18)

    fig.suptitle("Training history", x=0.008, ha="left", fontsize=13, fontweight="semibold")
    fig.tight_layout()
    return _save(fig, save_path)


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: Sequence[str],
    normalize: bool = True,
    save_path: str | Path | None = None,
    title: str = "Confusion matrix",
):
    """Row-normalised confusion matrix on a single-hue ramp (magnitude, not identity)."""
    cm = np.asarray(cm, dtype=float)
    display = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None) if normalize else cm

    size = max(5.0, 0.85 * len(classes) + 2.5)
    fig, ax = plt.subplots(figsize=(size, size * 0.86))
    im = ax.imshow(display, cmap=BLUE_CMAP, vmin=0, vmax=display.max())

    ax.set_xticks(range(len(classes)), classes, rotation=35, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    ax.grid(False)
    ax.set_xticks(np.arange(-0.5, len(classes), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(classes), 1), minor=True)
    ax.tick_params(which="minor", length=0)
    # 2px surface gap between cells.
    ax.grid(which="minor", color=SURFACE, linewidth=2)

    threshold = display.max() * 0.55
    for i in range(len(classes)):
        for j in range(len(classes)):
            value = display[i, j]
            text = f"{value:.2f}" if normalize else f"{int(cm[i, j]):,}"
            ax.text(
                j, i, text,
                ha="center", va="center", fontsize=9,
                color="#ffffff" if value > threshold else INK_SECONDARY,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(color=GRID, labelcolor=INK_SECONDARY)
    cbar.set_label("Share of true class" if normalize else "Images", color=INK_SECONDARY)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_per_class_metric(
    values: Sequence[float],
    classes: Sequence[str],
    metric: str = "F1",
    save_path: str | Path | None = None,
):
    """Horizontal bars, one series, sorted worst-first so the weak classes lead."""
    order = np.argsort(values)
    sorted_vals = np.asarray(values)[order]
    sorted_names = [classes[i] for i in order]

    fig, ax = plt.subplots(figsize=(7.5, max(2.6, 0.42 * len(classes) + 1.4)))
    bars = ax.barh(sorted_names, sorted_vals, color=SERIES[0], height=0.62)
    for bar in bars:  # 4px rounded data-end
        bar.set_joinstyle("round")

    for bar, value in zip(bars, sorted_vals):
        ax.text(
            bar.get_width() + 0.012, bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}", va="center", fontsize=9, color=INK_SECONDARY,
        )

    ax.set_xlim(0, max(1.0, float(sorted_vals.max()) * 1.12))
    ax.set_xlabel(metric)
    ax.set_title(f"Per-class {metric} — weakest first")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_class_distribution(counts: dict, save_path: str | Path | None = None, title: str = "Images per class"):
    """One series of counts — no legend, direct labels."""
    names = list(counts)
    values = [counts[n] for n in names]

    fig, ax = plt.subplots(figsize=(7.5, 4))
    bars = ax.bar(names, values, color=SERIES[0], width=0.62)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
            f"{value:,}", ha="center", va="bottom", fontsize=9, color=INK_SECONDARY,
        )
    ax.set_ylabel("Images")
    ax.set_title(title)
    ax.set_ylim(0, max(values) * 1.14)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    return _save(fig, save_path)


def show_batch(
    images, labels, classes: Sequence[str], mean, std,
    preds=None, n: int = 12, cols: int = 6, save_path: str | Path | None = None,
):
    """Grid of sample images; when `preds` is given, wrong ones are flagged in red."""
    from .data import denormalize

    n = min(n, len(images))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.25))
    axes = np.atleast_1d(axes).ravel()

    for i in range(len(axes)):
        ax = axes[i]
        ax.axis("off")
        if i >= n:
            continue
        ax.imshow(denormalize(images[i], mean, std).permute(1, 2, 0).numpy())
        true_name = classes[int(labels[i])]
        if preds is None:
            ax.set_title(true_name, fontsize=9, color=INK_PRIMARY, loc="center")
        else:
            pred_name = classes[int(preds[i])]
            correct = pred_name == true_name
            ax.set_title(
                pred_name if correct else f"{pred_name}\n(true: {true_name})",
                fontsize=9, loc="center",
                color=INK_SECONDARY if correct else "#e34948",
            )
    fig.tight_layout()
    return _save(fig, save_path)


def plot_prediction_bars(probs: Sequence[float], classes: Sequence[str], save_path: str | Path | None = None):
    """Top-k confidence bars for a single prediction."""
    order = np.argsort(probs)[::-1]
    names = [classes[i] for i in order]
    values = [float(probs[i]) for i in order]

    fig, ax = plt.subplots(figsize=(6, max(2.2, 0.42 * len(names) + 1)))
    ax.barh(names[::-1], values[::-1], color=SERIES[0], height=0.6)
    for i, value in enumerate(values[::-1]):
        ax.text(value + 0.012, i, f"{value:.1%}", va="center", fontsize=9, color=INK_SECONDARY)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Probability")
    ax.set_title("Prediction confidence")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, save_path)
