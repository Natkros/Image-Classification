#!/usr/bin/env python3
"""Train the classifier.

    python scripts/train.py
    python scripts/train.py --set model.name=efficientnet_b0 train.epochs=15
    python scripts/train.py --config configs/default.yaml --run-name my-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ic.config import load_config                                  # noqa: E402
from ic.data import build_dataloaders, build_datasets, class_distribution  # noqa: E402
from ic.engine import build_criterion, evaluate, fit               # noqa: E402
from ic.metrics import compute_metrics, format_summary             # noqa: E402
from ic.model import build_model                                   # noqa: E402
from ic.utils import (                                             # noqa: E402
    get_logger, load_checkpoint, make_run_dir, resolve_device, save_json, set_seed,
)
from ic.viz import plot_class_distribution, plot_confusion_matrix, plot_history, plot_per_class_metric, use_style  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--set", dest="overrides", nargs="*", default=[],
                        help="dotted overrides, e.g. train.epochs=20 data.batch_size=64")
    parser.add_argument("--run-name", default=None, help="name of the folder under outputs/")
    parser.add_argument("--no-test", action="store_true", help="skip the final test-set evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    if args.run_name:
        cfg.set_path("output.run_name", args.run_name)

    set_seed(int(cfg.get_path("seed", 42)))
    device = resolve_device(cfg.get_path("device", "auto"))

    run_dir = make_run_dir(cfg.output.dir, cfg.get_path("output.run_name"), cfg.model.name)
    logger = get_logger("ic.train", run_dir / "train.log")
    logger.info(f"Run directory: {run_dir}")
    cfg.save(run_dir / "config.yaml")
    use_style()

    # ---- data --------------------------------------------------------- #
    datasets = build_datasets(cfg)
    loaders, classes = build_dataloaders(cfg, datasets)
    logger.info(f"Classes ({len(classes)}): {', '.join(classes)}")
    for split in ("train", "val", "test"):
        if datasets.get(split) is not None:
            logger.info(f"  {split:<5} {len(datasets[split]):>7,} images")

    distribution = class_distribution(datasets["train_targets"], classes)
    plot_class_distribution(distribution, run_dir / "figures" / "class_distribution.png",
                            title="Training images per class")

    # ---- model -------------------------------------------------------- #
    model = build_model(
        name=cfg.model.name,
        num_classes=len(classes),
        pretrained=bool(cfg.get_path("model.pretrained", True)),
        dropout=float(cfg.get_path("model.dropout", 0.2)),
        freeze_backbone=bool(cfg.get_path("model.freeze_backbone", False)),
    ).to(device)

    # ---- train -------------------------------------------------------- #
    result = fit(model, loaders, cfg, device, classes, run_dir, logger,
                 train_targets=datasets["train_targets"])
    plot_history(result["history"], run_dir / "figures" / "training_history.png")

    # ---- final evaluation on the held-out test folder ------------------ #
    if not args.no_test:
        checkpoint = load_checkpoint(result["best_checkpoint"], map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        criterion = build_criterion(cfg, None, len(classes), device)
        outputs = evaluate(model, loaders["test"], criterion, device, desc="test", return_outputs=True)

        metrics = compute_metrics(outputs["targets"], outputs["preds"], outputs["probs"], classes)
        metrics["summary"]["test_loss"] = outputs["loss"]
        logger.info(f"Test accuracy: {metrics['summary']['accuracy']:.4f}")
        print(format_summary(metrics))

        save_json(run_dir / "test_metrics.json", metrics)
        plot_confusion_matrix(metrics["confusion_matrix"], classes,
                              save_path=run_dir / "figures" / "confusion_matrix.png",
                              title="Confusion matrix — test set")
        plot_per_class_metric([metrics["per_class"][c]["f1"] for c in classes], classes, "F1",
                              run_dir / "figures" / "per_class_f1.png")

    logger.info(f"Best checkpoint: {result['best_checkpoint']}")
    logger.info(f"Figures and logs: {run_dir}")


if __name__ == "__main__":
    main()
