"""Single-image prediction with optional Grad-CAM explanation.

Shared by the FastAPI service and the Streamlit dashboard so both surfaces
return identical numbers -- a dashboard that disagrees with the API is worse
than no dashboard.

The predictor loads the checkpoint once and keeps the model resident; loading
VGG19 per request costs about a second and dominates the actual inference.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import CLASS_DESCRIPTIONS
from .config import Config
from .data.datasets import build_transforms
from .models.gradcam import GradCAM, cam_focus_summary, denormalize, overlay_cam
from .models.vgg19 import load_checkpoint


class Predictor:
    """Loaded model plus everything needed to explain one prediction."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        config: Config | None = None,
        device: str | None = None,
    ):
        self.cfg = config or Config.load()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.checkpoint_path = Path(checkpoint) if checkpoint else self.cfg.path("api.checkpoint")

        self.model, self.metadata = load_checkpoint(self.checkpoint_path, self.device)
        self.classes = tuple(self.metadata.get("classes", self.cfg["classes"]))
        self.transform = build_transforms(self.cfg, train=False)

    # ------------------------------------------------------------------ core

    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

    def predict(self, image: Image.Image) -> dict:
        """Class probabilities for one slice."""
        tensor = self._to_tensor(image)
        with torch.no_grad():
            probabilities = torch.softmax(self.model(tensor), dim=1).cpu().numpy()[0]
        return self._format(probabilities)

    def explain(self, image: Image.Image, class_index: int | None = None) -> dict:
        """Prediction plus a base64 PNG Grad-CAM overlay.

        ``class_index`` lets a clinician ask a counterfactual -- "show me what
        would have made this AD" -- rather than only seeing the winning class.
        """
        tensor = self._to_tensor(image)
        with GradCAM(self.model, str(self.cfg["gradcam.target_layer"])) as explainer:
            cam, explained_index, probabilities = explainer(tensor, class_index)

        original = denormalize(tensor)
        blended = overlay_cam(original, cam, alpha=float(self.cfg["gradcam.alpha"]))

        result = self._format(probabilities)
        result["explained_class"] = self.classes[explained_index]
        result["overlay_png_base64"] = _encode_png(blended)
        result["heatmap_png_base64"] = _encode_png(
            (np.stack([cam] * 3, axis=-1) * 255).astype(np.uint8)
        )
        result["focus"] = cam_focus_summary(cam)
        return result

    # --------------------------------------------------------------- helpers

    def _format(self, probabilities: np.ndarray) -> dict:
        index = int(probabilities.argmax())
        ranked = sorted(
            (
                {
                    "class": name,
                    "description": CLASS_DESCRIPTIONS.get(name, name),
                    "probability": round(float(p), 4),
                }
                for name, p in zip(self.classes, probabilities)
            ),
            key=lambda item: -item["probability"],
        )
        return {
            "predicted_class": self.classes[index],
            "predicted_description": CLASS_DESCRIPTIONS.get(self.classes[index], ""),
            "confidence": round(float(probabilities[index]), 4),
            "probabilities": ranked,
            "impairment_probability": round(
                float(sum(p for name, p in zip(self.classes, probabilities) if name != "NC")),
                4,
            ),
            "checkpoint": self.checkpoint_path.name,
        }


def _encode_png(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def load_image_bytes(payload: bytes) -> Image.Image:
    """Decode uploaded bytes, raising a clear error for non-images."""
    try:
        return Image.open(io.BytesIO(payload))
    except Exception as exc:  # noqa: BLE001 - surfaced to the API caller
        raise ValueError(f"could not decode image: {exc}") from exc
