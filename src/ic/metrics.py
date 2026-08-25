"""Classification metrics built on scikit-learn, packaged for reporting."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], probs: np.ndarray | None, classes: Sequence[str]) -> dict:
    """Headline numbers plus a per-class breakdown."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(len(classes)))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    summary = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if probs is not None and len(classes) > 2:
        summary["top3_accuracy"] = float(
            top_k_accuracy_score(y_true, probs, k=min(3, len(classes)), labels=labels)
        )

    per_class = {
        classes[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in labels
    }

    return {
        "summary": summary,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "report_text": classification_report(
            y_true, y_pred, labels=labels, target_names=list(classes), digits=4, zero_division=0
        ),
        "classes": list(classes),
    }


def most_confused_pairs(cm, classes: Sequence[str], top: int = 5) -> list[tuple[str, str, int]]:
    """The off-diagonal cells with the most images — where the model actually struggles."""
    cm = np.asarray(cm)
    pairs = [
        (classes[i], classes[j], int(cm[i, j]))
        for i in range(len(classes))
        for j in range(len(classes))
        if i != j and cm[i, j] > 0
    ]
    return sorted(pairs, key=lambda p: -p[2])[:top]


def format_summary(metrics: dict) -> str:
    lines = ["", "Overall", "-" * 40]
    for key, value in metrics["summary"].items():
        lines.append(f"  {key.replace('_', ' '):<20} {value:.4f}")
    lines += ["", metrics["report_text"]]
    confused = most_confused_pairs(metrics["confusion_matrix"], metrics["classes"])
    if confused:
        lines += ["Most confused pairs", "-" * 40]
        lines += [f"  {t:<14} mistaken for {p:<14} {n:>6,} times" for t, p, n in confused]
    return "\n".join(lines)
