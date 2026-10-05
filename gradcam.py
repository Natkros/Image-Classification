"""Grad-CAM: highlights the image regions that pushed a model towards its prediction."""
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F


def explain(model, x: torch.Tensor):
    """x: (1, 3, H, W). Returns (class probabilities, heat map in [0, 1] of shape H x W)."""
    store = {}

    def hook(_, __, out):
        store["act"] = out
        out.register_hook(lambda g: store.update(grad=g))

    handle = model.features.register_forward_hook(hook)
    try:
        with torch.enable_grad():
            logits = model(x)
            model.zero_grad()
            logits[0, logits.argmax()].backward()
    finally:
        handle.remove()

    weights = store["grad"].mean((2, 3), keepdim=True)
    cam = F.relu((weights * store["act"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, x.shape[-2:], mode="bilinear", align_corners=False)[0, 0].detach()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return logits.softmax(1)[0].detach().cpu().numpy(), cam.cpu().numpy()


def overlay(image: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a heat map onto an HxWx3 float image in [0, 1]."""
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    return (1 - alpha) * image + alpha * heat
