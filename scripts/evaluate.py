#!/usr/bin/env python3
"""Evaluate a trained checkpoint on the test set.

    python scripts/evaluate.py --checkpoint outputs/<run>/checkpoints/best.pt
    python scripts/evaluate.py --checkpoint ... --split val --save-errors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ic.config import Config, _coerce, load_config                 # noqa: E402
from ic.data import build_dataloaders, build_datasets              # noqa: E402
from ic.engine import build_criterion, evaluate                    # noqa: E402
from ic.metrics import compute_metrics, format_summary             # noqa: E402
from ic.model import model_from_checkpoint                         # noqa: E402
from ic.utils import get_logger, load_checkpoint, resolve_device, save_json, set_seed  # noqa: E402
from ic.viz import plot_confusion_matrix, plot_per_class_metric, use_style  # noqa: E402


def latest_checkpoint(outputs_dir: Path) -> Path:
    candidates = sorted(outputs_dir.glob("*/checkpoints/best.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise SystemExit(
            f"No checkpoints found under {outputs_dir}. Train one first: python scripts/train.py"
        )
    return candidates[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=None, help="defaults to the most recent run")
    parser.add_argument("--config", default=None, help="defaults to the config stored in the checkpoint")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    parser.add_argument("--split", default="test", choices=["test", "val", "train"])
    parser.add_argument("--save-errors", action="store_true", help="write a CSV of misclassified images")
    parser.add_argument("--out", default=None, help="where to write metrics (defaults beside the checkpoint)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(REPO_ROOT / "outputs")
    print(f"Checkpoint: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path)
    if args.config:
        cfg = load_config(args.config, args.overrides)
    else:
        cfg = Config(checkpoint.get("config", {}))
        for item in args.overrides:
            key, _, raw = item.partition("=")
            cfg.set_path(key.strip(), _coerce(raw.strip()))

    set_seed(int(cfg.get_path("seed", 42)))
    device = resolve_device(cfg.get_path("device", "auto"))
    use_style()

    out_dir = Path(args.out) if args.out else checkpoint_path.parent.parent
    logger = get_logger("ic.eval", out_dir / "evaluate.log")

    datasets = build_datasets(cfg)
    loaders, classes = build_dataloaders(cfg, datasets)

    # "train" means the un-augmented, unshuffled view of the training images — scoring the
    # augmented loader would give random numbers and misaligned file paths.
    loader_key = "train_eval" if args.split == "train" else args.split
    if loader_key not in loaders:
        available = [k for k in loaders if k != "train"] + ["train"]
        raise SystemExit(f"Split {args.split!r} is not available (have: {', '.join(sorted(set(available)))})")

    model, classes = model_from_checkpoint(checkpoint, device)
    criterion = build_criterion(cfg, None, len(classes), device)

    outputs = evaluate(model, loaders[loader_key], criterion, device, desc=args.split, return_outputs=True)
    metrics = compute_metrics(outputs["targets"], outputs["preds"], outputs["probs"], classes)
    metrics["summary"]["loss"] = outputs["loss"]
    metrics["split"] = args.split
    metrics["checkpoint"] = str(checkpoint_path)

    print(format_summary(metrics))
    save_json(out_dir / f"{args.split}_metrics.json", metrics)
    plot_confusion_matrix(metrics["confusion_matrix"], classes,
                          save_path=out_dir / "figures" / f"confusion_matrix_{args.split}.png",
                          title=f"Confusion matrix — {args.split} set")
    plot_per_class_metric([metrics["per_class"][c]["f1"] for c in classes], classes, "F1",
                          out_dir / "figures" / f"per_class_f1_{args.split}.png")

    if args.save_errors:
        dataset = datasets[loader_key]
        samples = dataset.dataset.samples if hasattr(dataset, "dataset") else dataset.samples
        indices = dataset.indices if hasattr(dataset, "indices") else range(len(samples))
        paths = [samples[i][0] for i in indices]

        wrong = np.where(outputs["preds"] != outputs["targets"])[0]
        rows = ["path,true,predicted,confidence"]
        for i in wrong:
            confidence = float(outputs["probs"][i][outputs["preds"][i]])
            rows.append(
                f'"{paths[i]}",{classes[outputs["targets"][i]]},{classes[outputs["preds"][i]]},{confidence:.4f}'
            )
        errors_path = out_dir / f"errors_{args.split}.csv"
        errors_path.write_text("\n".join(rows), encoding="utf-8")
        print(f"\nWrote {len(wrong):,} misclassified images to {errors_path}")

    logger.info(f"{args.split} accuracy {metrics['summary']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
