"""Dataset discovery, transforms and dataloaders.

The Intel Image Classification archive unzips to a slightly awkward nested layout::

    data/intel/
      seg_train/seg_train/{buildings,forest,glacier,mountain,sea,street}/*.jpg
      seg_test/seg_test/{...}/*.jpg
      seg_pred/seg_pred/*.jpg          # unlabelled, no class folders

`find_split_dir` walks down until it finds the directory that actually holds the
class folders, so the same code works for a plain ``train/`` + ``test/`` layout too.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Candidate folder names, in priority order, for each split.
_SPLIT_ALIASES = {
    "train": ("seg_train", "train", "training"),
    "test": ("seg_test", "test", "testing", "val", "valid", "validation"),
    "pred": ("seg_pred", "pred", "predict", "unlabeled", "unlabelled"),
}


def _has_class_folders(path: Path) -> bool:
    """True if `path` contains subdirectories that directly hold images."""
    subdirs = [d for d in path.iterdir() if d.is_dir()]
    if not subdirs:
        return False
    return any(
        any(f.suffix.lower() in IMAGE_EXTENSIONS for f in d.iterdir() if f.is_file())
        for d in subdirs
    )


def find_split_dir(root: str | Path, split: str = "train", max_depth: int = 3) -> Path:
    """Locate the class-folder directory for a split beneath `root`."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"Data root not found: {root}\n"
            "Download the dataset first:  python scripts/download_data.py"
        )

    aliases = _SPLIT_ALIASES.get(split, (split,))

    # Breadth-first search for a directory whose name matches an alias.
    frontier = [(root, 0)]
    candidates: list[Path] = []
    while frontier:
        current, depth = frontier.pop(0)
        if depth > max_depth:
            continue
        for child in sorted(p for p in current.iterdir() if p.is_dir()):
            if child.name.lower() in aliases:
                candidates.append(child)
            frontier.append((child, depth + 1))

    # Prefer the deepest matching folder that actually contains class folders.
    for candidate in sorted(candidates, key=lambda p: -len(p.parts)):
        if _has_class_folders(candidate):
            return candidate

    # Fall back to the root itself (plain <root>/<class>/*.jpg layout).
    if _has_class_folders(root):
        return root

    raise FileNotFoundError(
        f"Could not find a '{split}' split with class subfolders under {root}. "
        "Expected something like <root>/seg_train/seg_train/<class>/*.jpg"
    )


def build_transforms(cfg, train: bool = True):
    """Compose the augmentation / preprocessing pipeline for one split."""
    size = int(cfg.data.image_size)
    mean, std = list(cfg.data.mean), list(cfg.data.std)
    aug = cfg.get_path("augment", {}) or {}

    if not train:
        return transforms.Compose(
            [
                transforms.Resize(int(size * 1.14)),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    steps: list = []
    if aug.get("random_resized_crop", True):
        scale = tuple(aug.get("crop_scale", (0.7, 1.0)))
        steps.append(transforms.RandomResizedCrop(size, scale=scale))
    else:
        steps += [transforms.Resize(int(size * 1.14)), transforms.CenterCrop(size)]

    if aug.get("horizontal_flip", 0):
        steps.append(transforms.RandomHorizontalFlip(p=float(aug["horizontal_flip"])))
    if aug.get("rotation_degrees", 0):
        steps.append(transforms.RandomRotation(float(aug["rotation_degrees"])))
    if aug.get("color_jitter", 0):
        j = float(aug["color_jitter"])
        steps.append(transforms.ColorJitter(brightness=j, contrast=j, saturation=j))

    steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]

    if aug.get("random_erasing", 0):
        steps.append(transforms.RandomErasing(p=float(aug["random_erasing"])))

    return transforms.Compose(steps)


def stratified_indices(
    targets: Sequence[int], val_split: float, seed: int = 42
) -> tuple[list[int], list[int]]:
    """Split indices per class so every class keeps its proportion in both halves."""
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)
    train_idx: list[int] = []
    val_idx: list[int] = []

    for label in np.unique(targets):
        idx = np.where(targets == label)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_split))) if val_split > 0 else 0
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


class UnlabeledImageDataset(Dataset):
    """Flat folder of images with no labels — used for the seg_pred split."""

    def __init__(self, folder: str | Path, transform=None) -> None:
        self.folder = Path(folder)
        self.transform = transform
        self.paths = sorted(
            p for p in self.folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found in {folder}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, str(path)


def build_datasets(cfg) -> dict:
    """Return {'train','val','test'} datasets plus the class list.

    Validation is carved out of the training folder (stratified) so the official
    test folder stays untouched until the final evaluation.
    """
    root = cfg.data.root
    train_dir = cfg.get_path("data.train_dir") or find_split_dir(root, "train")
    test_dir = cfg.get_path("data.test_dir") or find_split_dir(root, "test")

    train_tf = build_transforms(cfg, train=True)
    eval_tf = build_transforms(cfg, train=False)

    # Two views of the same folder: one augmented, one not.
    full_train = ImageFolder(str(train_dir), transform=train_tf)
    full_val = ImageFolder(str(train_dir), transform=eval_tf)

    val_split = float(cfg.get_path("data.val_split", 0.15) or 0.0)
    seed = int(cfg.get_path("seed", 42))
    if val_split > 0:
        tr_idx, va_idx = stratified_indices(full_train.targets, val_split, seed)
        train_ds: Dataset = Subset(full_train, tr_idx)
        # Same images as `train`, in the same order, but without augmentation —
        # so training data can be *scored* rather than trained on.
        train_eval_ds: Dataset = Subset(full_val, tr_idx)
        val_ds: Dataset | None = Subset(full_val, va_idx)
        train_targets = [full_train.targets[i] for i in tr_idx]
    else:
        train_ds, train_eval_ds, val_ds = full_train, full_val, None
        train_targets = list(full_train.targets)

    test_ds = ImageFolder(str(test_dir), transform=eval_tf)
    if test_ds.classes != full_train.classes:
        raise ValueError(
            "Train and test folders disagree on the class list, so the label indices would "
            f"not line up.\n  train ({train_dir}): {full_train.classes}\n"
            f"  test  ({test_dir}): {test_ds.classes}\n"
            "Remove any stray subfolders (e.g. .ipynb_checkpoints) or rename to match."
        )

    return {
        "train": train_ds,
        "train_eval": train_eval_ds,
        "val": val_ds,
        "test": test_ds,
        "classes": full_train.classes,
        "train_targets": train_targets,
        "train_dir": Path(train_dir),
        "test_dir": Path(test_dir),
    }


def build_dataloaders(cfg, datasets: dict | None = None) -> tuple[dict, list[str]]:
    """Wrap the datasets in DataLoaders configured from the config."""
    datasets = datasets or build_datasets(cfg)
    batch_size = int(cfg.data.batch_size)
    num_workers = int(cfg.get_path("data.num_workers", 4))
    pin = torch.cuda.is_available()
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )

    loaders = {
        "train": DataLoader(datasets["train"], shuffle=True, drop_last=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }
    # Evaluation loaders are never shuffled: downstream code lines predictions up with
    # dataset order to recover file paths.
    if datasets.get("train_eval") is not None:
        loaders["train_eval"] = DataLoader(datasets["train_eval"], shuffle=False, **common)
    if datasets.get("val") is not None:
        loaders["val"] = DataLoader(datasets["val"], shuffle=False, **common)

    return loaders, datasets["classes"]


def class_distribution(targets: Sequence[int], classes: Sequence[str]) -> dict[str, int]:
    counts = Counter(int(t) for t in targets)
    return {classes[label]: counts.get(label, 0) for label in range(len(classes))}


def compute_class_weights(targets: Sequence[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights, normalised to mean 1.0 — for imbalanced data."""
    counts = np.bincount(np.asarray(targets, dtype=int), minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def denormalize(tensor: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    """Undo Normalize so a tensor can be shown with matplotlib."""
    mean_t = torch.tensor(mean).view(-1, 1, 1)
    std_t = torch.tensor(std).view(-1, 1, 1)
    return (tensor.detach().cpu() * std_t + mean_t).clamp(0, 1)
