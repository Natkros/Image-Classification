"""
Final Project: Image Classification
Author: Kritarth Saxena
GitHub: @Natkros

Model definitions:
1. CustomCNN: 4-stage convolutional network built from scratch.
2. TransferLearningClassifier: Pretrained MobileNetV2 with custom classification head.
"""
import torch
import torch.nn as nn
import torchvision.models as models


class ConvBlock(nn.Module):
    """
    Standard double convolution block with batch norm, ReLU, and max pooling.
    Added spatial dropout to help prevent early co-adaptation of features.
    """
    def __init__(self, in_c: int, out_c: int, drop: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(drop)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomCNN(nn.Module):
    """
    4-stage CNN architecture.
    Progressively increases channels (32 -> 64 -> 128 -> 256) while reducing spatial resolution.
    Uses AdaptiveAvgPool2d so it works cleanly regardless of minor input resolution changes.
    """
    def __init__(self, num_classes: int = 2, in_c: int = 3, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_c, 32, drop=0.1),
            ConvBlock(32, 64, drop=0.15),
            ConvBlock(64, 128, drop=0.2),
            ConvBlock(128, 256, drop=0.25)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        return self.classifier(x)


class TransferLearningClassifier(nn.Module):
    """
    Transfer learning wrapper around MobileNetV2 / ResNet-18.
    Replaces the default 1000-class head with our custom 2-class classifier head.
    """
    def __init__(self, backbone: str = "mobilenet_v2", num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        if backbone == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            self.model = models.mobilenet_v2(weights=weights)
            in_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )
        elif backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.model = models.resnet18(weights=weights)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def get_model(name: str = "custom_cnn", num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name in ["custom_cnn", "cnn", "custom"]:
        return CustomCNN(num_classes=num_classes)
    backbone = "resnet18" if "resnet" in name else "mobilenet_v2"
    return TransferLearningClassifier(backbone=backbone, num_classes=num_classes, pretrained=pretrained)


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable, "frozen_params": total - trainable}
