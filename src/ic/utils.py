"""Reproducibility, device selection, logging and checkpoint helpers."""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed every RNG we rely on so a run can be reproduced."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def resolve_device(preference: str = "auto") -> torch.device:
    """Pick the best available device, honouring an explicit preference."""
    preference = (preference or "auto").lower()
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_logger(name: str = "ic", logfile: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(logfile, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def make_run_dir(base: str | Path, run_name: str | None, model_name: str = "model") -> Path:
    """Create outputs/<run_name>/ with checkpoints/ and figures/ subfolders."""
    if not run_name:
        run_name = f"{model_name}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(base) / run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    return run_dir


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def save_checkpoint(path: str | Path, model, classes, cfg, epoch: int, metrics: dict) -> None:
    """Save weights plus everything needed to rebuild the model for inference."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": list(classes),
            "config": cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg),
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\nTrain a model first: python scripts/train.py"
        )
    # weights_only=False because we also store the class list and config dict.
    return torch.load(path, map_location=map_location, weights_only=False)


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


@dataclass
class AverageMeter:
    """Running average of a scalar (loss, accuracy, ...)."""

    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0

    def reset(self) -> None:
        self.total, self.count = 0.0, 0


@dataclass
class History:
    """Per-epoch metric history, easy to hand to pandas or matplotlib."""

    records: list[dict] = field(default_factory=list)

    def append(self, **kwargs) -> None:
        self.records.append(kwargs)

    def series(self, key: str) -> list:
        return [r[key] for r in self.records if key in r]

    def to_dict(self) -> list[dict]:
        return self.records

    def save(self, path: str | Path) -> None:
        save_json(path, self.records)


class EarlyStopping:
    """Stop when the monitored metric has not improved for `patience` epochs."""

    def __init__(self, patience: int = 5, mode: str = "max", min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best: float | None = None
        self.counter = 0
        self.should_stop = False

    def _is_better(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def step(self, value: float) -> bool:
        """Return True if this value is a new best."""
        if self._is_better(value):
            self.best = value
            self.counter = 0
            return True
        self.counter += 1
        if self.patience and self.counter >= self.patience:
            self.should_stop = True
        return False


def format_seconds(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"
