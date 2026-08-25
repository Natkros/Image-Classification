"""Minimal Grad-CAM — shows which pixels drove a prediction.

Hooks the last conv layer, weights its feature maps by the gradient of the target
logit, and returns a heatmap the size of the input image.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .model import last_conv_layer


class GradCAM:
    def __init__(self, model, target_layer=None) -> None:
        self.model = model.eval()
        self.target_layer = target_layer or last_conv_layer(model)
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [
            self.target_layer.register_forward_hook(self._save_activation),
            self.target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, _module, _inp, output):
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, input_tensor: torch.Tensor, class_index: int | None = None) -> np.ndarray:
        """Return an HxW heatmap in [0, 1] for a single 1x3xHxW input."""
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        # The backward hook only fires if something in the graph requires grad. With a
        # frozen backbone nothing does, so we make the input itself require grad.
        input_tensor = input_tensor.detach().requires_grad_(True)

        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(input_tensor)
            if class_index is None:
                class_index = int(logits.argmax(1).item())
            logits[0, class_index].backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "Grad-CAM captured no gradients — the target layer did not take part in "
                "the backward pass. Check that `target_layer` is inside the model."
            )

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)      # global-average-pool the grads
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = cam - cam.min()
        denom = cam.max()
        if denom > 0:
            cam = cam / denom
        return cam.cpu().numpy()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def overlay_heatmap(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a Grad-CAM heatmap over an RGB image in [0, 1]."""
    from .viz import BLUE_CMAP

    heat = BLUE_CMAP(cam)[..., :3]
    return np.clip((1 - alpha) * image_rgb + alpha * heat, 0, 1)
