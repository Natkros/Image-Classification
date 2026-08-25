"""Loading a trained checkpoint and predicting on new images."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .data import IMAGE_EXTENSIONS
from .model import model_from_checkpoint
from .utils import load_checkpoint, resolve_device


class Predictor:
    """Wraps a checkpoint into a simple `predict(image) -> ranked classes` object."""

    def __init__(self, checkpoint_path: str | Path, device: str = "auto") -> None:
        self.device = resolve_device(device)
        checkpoint = load_checkpoint(checkpoint_path, map_location=self.device)
        self.model, self.classes = model_from_checkpoint(checkpoint, self.device)
        self.config = checkpoint.get("config", {})
        self.epoch = checkpoint.get("epoch")
        self.train_metrics = checkpoint.get("metrics", {})

        data_cfg = self.config.get("data", {}) if isinstance(self.config, dict) else {}
        size = int(data_cfg.get("image_size", 224))
        self.mean = list(data_cfg.get("mean", [0.485, 0.456, 0.406]))
        self.std = list(data_cfg.get("std", [0.229, 0.224, 0.225]))
        self.transform = transforms.Compose(
            [
                transforms.Resize(int(size * 1.14)),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

    # -- single image ------------------------------------------------------ #
    def _to_tensor(self, image: str | Path | Image.Image) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        return self.transform(image.convert("RGB"))

    @torch.no_grad()
    def predict(self, image: str | Path | Image.Image, top_k: int = 3) -> list[tuple[str, float]]:
        """Return [(class_name, probability), ...] sorted high to low."""
        tensor = self._to_tensor(image).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(tensor).float(), dim=1)[0].cpu().numpy()
        order = np.argsort(probs)[::-1][: max(1, min(top_k, len(self.classes)))]
        return [(self.classes[i], float(probs[i])) for i in order]

    @torch.no_grad()
    def predict_proba(self, image: str | Path | Image.Image) -> np.ndarray:
        tensor = self._to_tensor(image).unsqueeze(0).to(self.device)
        return torch.softmax(self.model(tensor).float(), dim=1)[0].cpu().numpy()

    # -- many images ------------------------------------------------------- #
    @torch.no_grad()
    def predict_batch(self, paths: Sequence[str | Path], batch_size: int = 32, top_k: int = 1) -> list[dict]:
        results: list[dict] = []
        paths = [Path(p) for p in paths]
        for start in range(0, len(paths), batch_size):
            chunk = paths[start : start + batch_size]
            tensors, valid = [], []
            for path in chunk:
                try:
                    tensors.append(self._to_tensor(path))
                    valid.append(path)
                except Exception as exc:  # unreadable / corrupt file
                    results.append({"path": str(path), "error": str(exc)})
            if not tensors:
                continue
            batch = torch.stack(tensors).to(self.device)
            probs = torch.softmax(self.model(batch).float(), dim=1).cpu().numpy()
            for path, row in zip(valid, probs):
                order = np.argsort(row)[::-1][:top_k]
                results.append(
                    {
                        "path": str(path),
                        "prediction": self.classes[int(order[0])],
                        "confidence": float(row[int(order[0])]),
                        "top_k": [(self.classes[int(i)], float(row[int(i)])) for i in order],
                    }
                )
        return results

    def gradcam(self, image: str | Path | Image.Image, class_index: int | None = None):
        """Return (rgb_image, heatmap) for a single image."""
        from .data import denormalize
        from .gradcam import GradCAM

        tensor = self._to_tensor(image).to(self.device)
        with GradCAM(self.model) as cam_fn:
            cam = cam_fn(tensor.unsqueeze(0), class_index)
        rgb = denormalize(tensor, self.mean, self.std).permute(1, 2, 0).numpy()
        return rgb, cam


def collect_images(target: str | Path) -> list[Path]:
    """Accept a file or a folder and return the image paths inside it."""
    target = Path(target)
    if target.is_file():
        return [target]
    if target.is_dir():
        return sorted(p for p in target.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    raise FileNotFoundError(f"No such file or directory: {target}")
