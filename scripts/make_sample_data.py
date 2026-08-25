#!/usr/bin/env python3
"""Generate a tiny synthetic dataset so the pipeline can be smoke-tested offline.

    python scripts/make_sample_data.py --dest data/sample --per-class 24

The images are procedural colour/texture patterns, not real photos — they exist
only to prove that training, evaluation and inference run end to end.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
BASE_COLORS = {
    "buildings": (150, 140, 130),
    "forest": (40, 110, 55),
    "glacier": (200, 220, 235),
    "mountain": (120, 115, 130),
    "sea": (35, 95, 165),
    "street": (95, 95, 100),
}


def make_image(class_name: str, rng: np.random.Generator, size: int = 150) -> Image.Image:
    base = np.array(BASE_COLORS[class_name], dtype=float)
    canvas = np.tile(base, (size, size, 1))

    gradient = np.linspace(-40, 40, size).reshape(-1, 1, 1)
    canvas += gradient  # vertical shading, like a horizon

    index = CLASSES.index(class_name)
    if index % 3 == 0:      # vertical stripes
        canvas[:, ::7, :] += 35
    elif index % 3 == 1:    # horizontal bands
        canvas[::5, :, :] -= 30
    else:                   # blocks
        canvas[::11, ::11, :] += 45

    canvas += rng.normal(0, 12, canvas.shape)
    return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", default="data/sample")
    parser.add_argument("--per-class", type=int, default=24, help="training images per class")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    dest = Path(args.dest)

    for split, count in (("seg_train", args.per_class), ("seg_test", max(4, args.per_class // 4))):
        for class_name in CLASSES:
            folder = dest / split / split / class_name
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                make_image(class_name, rng).save(folder / f"{class_name}_{i:04d}.jpg", quality=90)

    total = len(CLASSES) * (args.per_class + max(4, args.per_class // 4))
    print(f"Wrote {total:,} synthetic images to {dest}")
    print(f"Try:  python scripts/train.py --set data.root={dest} train.epochs=2 data.num_workers=0")


if __name__ == "__main__":
    main()
