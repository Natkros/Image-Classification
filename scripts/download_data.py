#!/usr/bin/env python3
"""Download the Intel Image Classification dataset into data/intel/.

Two ways to authenticate with Kaggle:

  1. Put your API token at ~/.kaggle/kaggle.json  (Kaggle -> Settings -> Create New Token)
  2. Export KAGGLE_USERNAME and KAGGLE_KEY in your shell

Then:  python scripts/download_data.py

If you would rather download by hand, grab the archive from
https://www.kaggle.com/datasets/puneet6060/intel-image-classification
and unzip it into data/intel/ so you end up with data/intel/seg_train/seg_train/<class>/.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DATASET_SLUG = "puneet6060/intel-image-classification"
EXPECTED_CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def have_credentials() -> bool:
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def download(dest: Path, force: bool = False) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "intel-image-classification.zip"

    if archive.exists() and not force:
        print(f"Archive already present: {archive}")
        return archive

    if shutil.which("kaggle") is None:
        raise SystemExit(
            "The `kaggle` CLI is not installed.\n"
            "  pip install kaggle\n"
            "…or download the dataset manually (see the docstring at the top of this file)."
        )
    if not have_credentials():
        raise SystemExit(
            "No Kaggle credentials found.\n"
            "  Put your token at ~/.kaggle/kaggle.json, or export KAGGLE_USERNAME / KAGGLE_KEY.\n"
            "  Token: https://www.kaggle.com/settings -> API -> Create New Token"
        )

    print(f"Downloading {DATASET_SLUG} -> {dest} (about 350 MB)…")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET_SLUG, "-p", str(dest)],
        check=True,
    )
    return archive


def extract(archive: Path, dest: Path) -> None:
    print(f"Extracting {archive.name}…")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)
    print("Extracted.")


def verify(dest: Path) -> bool:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from ic.data import find_split_dir  # imported late so --help works without torch

    ok = True
    for split in ("train", "test"):
        try:
            split_dir = find_split_dir(dest, split)
        except FileNotFoundError as exc:
            print(f"  [!] {exc}")
            ok = False
            continue
        classes = sorted(d.name for d in split_dir.iterdir() if d.is_dir())
        counts = {c: len(list((split_dir / c).glob("*"))) for c in classes}
        total = sum(counts.values())
        print(f"  {split:<5} {split_dir.relative_to(dest)}  —  {total:,} images")
        for name, n in counts.items():
            print(f"          {name:<12} {n:>7,}")
        missing = set(EXPECTED_CLASSES) - set(classes)
        if missing:
            print(f"  [!] missing expected classes: {sorted(missing)}")
            ok = False
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", default=str(REPO_ROOT / "data" / "intel"), help="where to put the data")
    parser.add_argument("--force", action="store_true", help="re-download even if the archive exists")
    parser.add_argument("--keep-archive", action="store_true", help="do not delete the zip after extracting")
    parser.add_argument("--verify-only", action="store_true", help="just check what is already on disk")
    args = parser.parse_args()

    dest = Path(args.dest)

    if args.verify_only:
        raise SystemExit(0 if verify(dest) else 1)

    archive = download(dest, force=args.force)
    extract(archive, dest)
    if not args.keep_archive:
        archive.unlink(missing_ok=True)

    print("\nDataset summary:")
    if verify(dest):
        print("\nReady. Next:  python scripts/train.py --config configs/default.yaml")
    else:
        print("\nSomething looks off — check the layout above against the docstring.")


if __name__ == "__main__":
    main()
