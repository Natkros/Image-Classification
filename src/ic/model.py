"""Model factory for transfer learning.

Every backbone comes from torchvision with ImageNet weights. We replace the final
classifier with a small head sized for our number of classes, and expose helpers
to freeze / unfreeze the backbone so training can run in two stages:

  stage 1  frozen backbone, train the head only  -> fast, stable, no catastrophic forgetting
  stage 2  unfreeze, fine-tune everything at a much smaller LR -> squeezes out the last few points
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torchvision import models

# name -> (constructor, default-weights enum attribute)
_BACKBONES: dict[str, tuple[Callable, str]] = {
    "resnet18": (models.resnet18, "ResNet18_Weights"),
    "resnet34": (models.resnet34, "ResNet34_Weights"),
    "resnet50": (models.resnet50, "ResNet50_Weights"),
    "efficientnet_b0": (models.efficientnet_b0, "EfficientNet_B0_Weights"),
    "efficientnet_b1": (models.efficientnet_b1, "EfficientNet_B1_Weights"),
    "mobilenet_v3_large": (models.mobilenet_v3_large, "MobileNet_V3_Large_Weights"),
    "convnext_tiny": (models.convnext_tiny, "ConvNeXt_Tiny_Weights"),
}

AVAILABLE_MODELS = tuple(_BACKBONES)


def _default_weights(weights_attr: str):
    enum = getattr(models, weights_attr, None)
    return getattr(enum, "DEFAULT", None) if enum is not None else None


def _replace_classifier(model: nn.Module, name: str, num_classes: int, dropout: float) -> nn.Module:
    """Swap the ImageNet head (1000 classes) for one sized to our dataset."""
    if name.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    elif name.startswith("efficientnet"):
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    elif name.startswith("mobilenet"):
        in_features = model.classifier[-1].in_features
        head = list(model.classifier[:-1])          # keep Linear + Hardswish
        head += [nn.Dropout(dropout), nn.Linear(in_features, num_classes)]
        model.classifier = nn.Sequential(*head)
    elif name.startswith("convnext"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:  # pragma: no cover - guarded by build_model
        raise ValueError(f"Do not know how to replace the head of {name!r}")
    return model


def build_model(
    name: str = "resnet18",
    num_classes: int = 6,
    pretrained: bool = True,
    dropout: float = 0.2,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Create a backbone with a fresh classification head."""
    name = name.lower()
    if name not in _BACKBONES:
        raise ValueError(f"Unknown model {name!r}. Available: {', '.join(AVAILABLE_MODELS)}")

    ctor, weights_attr = _BACKBONES[name]
    weights = _default_weights(weights_attr) if pretrained else None
    model = ctor(weights=weights)
    model = _replace_classifier(model, name, num_classes, dropout)
    model.model_name = name  # remembered so inference can rebuild the same architecture

    if freeze_backbone:
        set_backbone_frozen(model, True)
    return model


def classifier_module(model: nn.Module) -> nn.Module:
    """Return the head submodule, whatever the backbone calls it."""
    for attr in ("fc", "classifier", "head"):
        if hasattr(model, attr):
            return getattr(model, attr)
    raise AttributeError("Could not locate the classifier head on this model")


def set_backbone_frozen(model: nn.Module, frozen: bool = True) -> None:
    """Freeze or unfreeze everything except the classification head."""
    head = classifier_module(model)
    head_params = {id(p) for p in head.parameters()}
    for param in model.parameters():
        if id(param) not in head_params:
            param.requires_grad = not frozen
    for param in head.parameters():
        param.requires_grad = True


def param_groups(model: nn.Module, head_lr: float, backbone_lr: float) -> list[dict]:
    """Discriminative learning rates: small for pretrained weights, larger for the new head."""
    head = classifier_module(model)
    head_ids = {id(p) for p in head.parameters()}
    backbone = [p for p in model.parameters() if id(p) not in head_ids and p.requires_grad]
    head_params = [p for p in head.parameters() if p.requires_grad]

    groups = [{"params": head_params, "lr": head_lr, "name": "head"}]
    if backbone:
        groups.append({"params": backbone, "lr": backbone_lr, "name": "backbone"})
    return groups


def last_conv_layer(model: nn.Module) -> nn.Module:
    """Deepest convolutional layer — the target for Grad-CAM."""
    conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            conv = module
    if conv is None:  # pragma: no cover
        raise ValueError("No Conv2d layer found in this model")
    return conv


def model_from_checkpoint(checkpoint: dict, device: torch.device | str = "cpu") -> tuple[nn.Module, list[str]]:
    """Rebuild the exact architecture stored in a checkpoint and load its weights."""
    cfg = checkpoint.get("config", {})
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    classes = list(checkpoint["classes"])

    model = build_model(
        name=model_cfg.get("name", "resnet18"),
        num_classes=len(classes),
        pretrained=False,  # weights come from the checkpoint
        dropout=float(model_cfg.get("dropout", 0.2)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, classes
