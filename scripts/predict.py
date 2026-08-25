#!/usr/bin/env python3
"""Classify new images with a trained checkpoint.

    python scripts/predict.py path/to/image.jpg
    python scripts/predict.py data/intel/seg_pred --top-k 3 --csv predictions.csv
    python scripts/predict.py image.jpg --gradcam gradcam.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ic.inference import Predictor, collect_images                 # noqa: E402


def latest_checkpoint(outputs_dir: Path) -> Path:
    candidates = sorted(outputs_dir.glob("*/checkpoints/best.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise SystemExit(
            f"No checkpoints found under {outputs_dir}. Train one first: python scripts/train.py"
        )
    return candidates[-1]


def write_csv(path: str | None, results: list[dict]) -> None:
    if not path:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["path", "prediction", "confidence"])
        for row in results:
            writer.writerow([row["path"], row.get("prediction", ""), f"{row.get('confidence', 0):.4f}"])
    print(f"\nWrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", help="an image file or a folder of images")
    parser.add_argument("--checkpoint", default=None, help="defaults to the most recent run")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--csv", default=None, help="write results to this CSV")
    parser.add_argument("--gradcam", default=None, help="save a Grad-CAM overlay (single image only)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(REPO_ROOT / "outputs")

    predictor = Predictor(checkpoint, device=args.device)
    print(f"Checkpoint: {checkpoint}")
    print(f"Classes:    {', '.join(predictor.classes)}\n")

    paths = collect_images(args.target)
    if not paths:
        raise SystemExit(f"No images found at {args.target}")

    if len(paths) == 1:
        ranked = predictor.predict(paths[0], top_k=args.top_k)
        print(f"{paths[0].name}")
        width = max(len(name) for name, _ in ranked)
        for rank, (name, prob) in enumerate(ranked, 1):
            bar = "█" * int(round(prob * 28))
            print(f"  {rank}. {name:<{width}}  {prob:6.2%}  {bar}")

        if args.gradcam:
            import matplotlib.pyplot as plt

            from ic.gradcam import overlay_heatmap
            from ic.viz import use_style

            use_style()
            rgb, cam = predictor.gradcam(paths[0])
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(rgb); axes[0].set_title("Input"); axes[0].axis("off")
            axes[1].imshow(overlay_heatmap(rgb, cam))
            axes[1].set_title(f"Grad-CAM — {ranked[0][0]}"); axes[1].axis("off")
            fig.tight_layout()
            Path(args.gradcam).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.gradcam)
            print(f"\nSaved Grad-CAM overlay to {args.gradcam}")

        results = [{
            "path": str(paths[0]),
            "prediction": ranked[0][0],
            "confidence": ranked[0][1],
            "top_k": ranked,
        }]
        write_csv(args.csv, results)
        return

    print(f"Classifying {len(paths):,} images…")
    results = predictor.predict_batch(paths, batch_size=args.batch_size, top_k=args.top_k)

    counts: dict[str, int] = {}
    for row in results:
        if "prediction" in row:
            counts[row["prediction"]] = counts.get(row["prediction"], 0) + 1

    print("\nPredicted class counts")
    for name in sorted(counts, key=lambda n: -counts[n]):
        print(f"  {name:<14} {counts[name]:>7,}")

    errors = [r for r in results if "error" in r]
    if errors:
        print(f"\n{len(errors)} file(s) could not be read; first: {errors[0]['path']}")

    write_csv(args.csv, results)


if __name__ == "__main__":
    main()
