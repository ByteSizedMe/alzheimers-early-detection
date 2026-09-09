"""Grad-CAM: turning a prediction into a picture a clinician can check.

Selvaraju et al., 2017. The gradient of the target class score with respect to
the last convolutional feature map, global-average-pooled per channel, gives a
weight for each channel. The ReLU of the weighted channel sum is a coarse map
of "what raised this score" -- for our model, where the atrophy the network
keyed on actually is.

This is the difference between a number a radiologist has to take on faith and
a claim they can accept or reject. A model that reaches 92% by keying on a scanner
watermark looks identical to a good one until you see the heatmap.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.datasets import IMAGENET_MEAN, IMAGENET_STD


class GradCAM:
    """Grad-CAM for a single target layer.

    ``target_layer`` must be a *post-activation* map. On VGG19 that is
    ``features.35`` (the ReLU), not ``features.34`` (the conv it follows):
    raw conv output contains large negative values, the channel-weighted sum
    goes negative everywhere, and the final ReLU flattens the CAM to zero.

    Register once and reuse; call :meth:`release` (or use it as a context
    manager) to detach the hooks, otherwise the layer keeps a reference to the
    stored activations for the lifetime of the model.
    """

    def __init__(self, model: nn.Module, target_layer: str = "features.35"):
        self.model = model
        self.target_layer_name = target_layer
        self.layer = self._resolve_layer(model, target_layer)

        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        # A forward hook plus a hook on the output *tensor*, rather than
        # register_full_backward_hook: the latter is incompatible with the
        # in-place ReLUs a torchvision VGG is built from, because autograd
        # refuses to let a custom-Function output be modified in place.
        self._handles = [self.layer.register_forward_hook(self._save_activations)]
        self._tensor_handle = None

    # ------------------------------------------------------------------ hooks

    @staticmethod
    def _resolve_layer(model: nn.Module, dotted: str) -> nn.Module:
        node: nn.Module = model
        for part in dotted.split("."):
            node = node[int(part)] if part.isdigit() else getattr(node, part)
        return node

    def _save_activations(self, _module, _inputs, output) -> None:
        self.activations = output.detach().clone()
        if output.requires_grad:
            self._tensor_handle = output.register_hook(self._save_gradients)

    def _save_gradients(self, grad) -> None:
        self.gradients = grad.detach()

    def release(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        if self._tensor_handle is not None:
            self._tensor_handle.remove()
            self._tensor_handle = None

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *_exc) -> None:
        self.release()

    # -------------------------------------------------------------- inference

    def __call__(
        self, image_tensor: torch.Tensor, class_index: int | None = None
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """Compute the CAM for one image.

        Parameters
        ----------
        image_tensor:
            Normalised ``(1, 3, H, W)`` or ``(3, H, W)`` tensor.
        class_index:
            Class to explain. Defaults to the model's own prediction, which is
            what you want when auditing a diagnosis; pass an explicit index to
            ask "what would have made this AD?".

        Returns
        -------
        ``(cam, class_index, probabilities)`` where ``cam`` is float32 in [0, 1]
        at the input's spatial size.
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        if image_tensor.size(0) != 1:
            raise ValueError("GradCAM handles one image at a time")

        was_training = self.model.training
        self.model.eval()

        # Gradients are needed even at inference time -- that is the whole method.
        image_tensor = image_tensor.clone().requires_grad_(True)
        logits = self.model(image_tensor)
        probabilities = F.softmax(logits, dim=1).detach().cpu().numpy()[0]

        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())

        self.model.zero_grad(set_to_none=True)
        logits[0, class_index].backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                f"no activations captured at {self.target_layer_name!r}; "
                "check the layer name against the model's architecture"
            )

        # Channel importance = global average pooled gradient.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False
        )

        cam = cam.squeeze().cpu().numpy()
        span = cam.max() - cam.min()
        # A dead map (uniform activation) normalises to NaN otherwise.
        cam = (cam - cam.min()) / span if span > 1e-8 else np.zeros_like(cam)

        if was_training:
            self.model.train()
        return cam.astype(np.float32), class_index, probabilities


def denormalize(image_tensor: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalisation, returning uint8 RGB ``(H, W, 3)``."""
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]
    array = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    array = array * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return (np.clip(array, 0, 1) * 255).astype(np.uint8)


def overlay_cam(
    image: np.ndarray, cam: np.ndarray, alpha: float = 0.45, colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """Blend a CAM over the source slice as a uint8 RGB image."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if cam.shape[:2] != image.shape[:2]:
        cam = cv2.resize(cam, (image.shape[1], image.shape[0]))

    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = (1 - alpha) * image.astype(np.float32) + alpha * heatmap.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def cam_focus_summary(cam: np.ndarray, threshold: float = 0.5) -> dict:
    """Coarse, quotable description of where the model looked.

    Not anatomy -- it is a quadrant breakdown of CAM mass. Enough to flag the
    failure mode that matters: a model attending to image corners or background
    rather than to central brain structure.
    """
    height, width = cam.shape
    mid_y, mid_x = height // 2, width // 2
    quadrants = {
        "upper_left": cam[:mid_y, :mid_x],
        "upper_right": cam[:mid_y, mid_x:],
        "lower_left": cam[mid_y:, :mid_x],
        "lower_right": cam[mid_y:, mid_x:],
    }
    total = cam.sum()
    shares = {
        name: float(region.sum() / total) if total > 0 else 0.0
        for name, region in quadrants.items()
    }

    # Central half of the frame -- where brain tissue sits in a registered slice.
    centre = cam[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
    return {
        "quadrant_share": {k: round(v, 3) for k, v in shares.items()},
        "dominant_quadrant": max(shares, key=shares.get) if total > 0 else None,
        "central_share": round(float(centre.sum() / total), 3) if total > 0 else 0.0,
        "activated_fraction": round(float((cam >= threshold).mean()), 3),
    }
