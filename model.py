"""Two classifiers: a CNN built from scratch and a fine-tuned MobileNetV2. Both expose `.features`."""
import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = {"custom_cnn": "Custom CNN", "mobilenet_v2": "MobileNetV2"}


def conv_block(c_in: int, c_out: int, drop: float) -> nn.Sequential:
    """Two 3x3 convs (BatchNorm + ReLU), then 2x2 max-pool and spatial dropout."""
    layers = []
    for a, b in [(c_in, c_out), (c_out, c_out)]:
        layers += [nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers, nn.MaxPool2d(2), nn.Dropout2d(drop))


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        stem = [nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*stem, conv_block(32, 64, .1), conv_block(64, 128, .15), conv_block(128, 256, .2))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),
            nn.Linear(256 * 16, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(.3),
            nn.Linear(512, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def get_model(name: str, pretrained: bool = True) -> nn.Module:
    if name == "custom_cnn":
        return CustomCNN()
    net = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT if pretrained else None)
    net.classifier = nn.Sequential(nn.Dropout(.3), nn.Linear(net.last_channel, 2))
    return net


def load_trained(name: str, save_dir: str = "saved_models") -> nn.Module:
    model = get_model(name, pretrained=False)
    model.load_state_dict(torch.load(f"{save_dir}/best_{name}.pth", map_location=DEVICE))
    return model.to(DEVICE).eval()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
