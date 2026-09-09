"""VGG19 backbone with a fresh 4-class head.

The deck's argument for transfer learning holds up: ADNI-scale datasets are far
too small to fit 143M parameters from scratch, but the early ImageNet
convolutions already encode the edge and texture detectors that ventricular
enlargement and cortical thinning show up as. So we freeze most of the
backbone, unfreeze the last conv block, and train a small new head.

VGG19 is also a deliberate choice for explainability: its final feature map is
14x14 at 224px input, coarse enough for stable Grad-CAM gradients and fine
enough to localise to a lobe. Grad-CAM reads ``features.35``, the ReLU after
the last conv.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from .. import CLASSES

# torchvision's VGG19 features block; indices of each MaxPool mark block ends.
_VGG19_BLOCK_ENDS = (4, 9, 18, 27, 36)


def build_model(
    num_classes: int = len(CLASSES),
    pretrained: bool = True,
    dropout: float = 0.5,
    trainable_blocks: int = 1,
    hidden_dim: int = 512,
) -> nn.Module:
    """Return a VGG19 adapted to MRI stage classification.

    Parameters
    ----------
    pretrained:
        Load ImageNet weights. Downloads ~550 MB on first use; set False for
        offline smoke tests (the architecture is identical, only worse).
    trainable_blocks:
        How many trailing conv blocks to unfreeze, 0-5. ``1`` fine-tunes only
        the deepest block, which is the right trade-off under a few thousand
        training slices.
    """
    if not 0 <= trainable_blocks <= len(_VGG19_BLOCK_ENDS):
        raise ValueError(f"trainable_blocks must be 0..{len(_VGG19_BLOCK_ENDS)}")

    weights = models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vgg19(weights=weights)

    _freeze_backbone(model, trainable_blocks)

    # torchvision builds VGG with in-place ReLUs, which overwrite the conv
    # activations Grad-CAM needs to read. The extra memory is a few hundred MB
    # at batch 16 -- cheap next to losing explainability, which is the whole
    # reason this project picked VGG19.
    for index, layer in enumerate(model.features):
        if isinstance(layer, nn.ReLU):
            model.features[index] = nn.ReLU(inplace=False)

    # Replace the 1000-class ImageNet head. Two dropout layers, because a small
    # medical dataset overfits a 25088-wide feature vector almost immediately.
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout / 2),
        nn.Linear(hidden_dim // 2, num_classes),
    )
    return model


def _freeze_backbone(model: nn.Module, trainable_blocks: int) -> None:
    """Freeze all conv layers except the last ``trainable_blocks`` blocks."""
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    if trainable_blocks == 0:
        return

    first_trainable = _VGG19_BLOCK_ENDS[-trainable_blocks - 1] if trainable_blocks < len(
        _VGG19_BLOCK_ENDS
    ) else 0
    for index, layer in enumerate(model.features):
        if index >= first_trainable:
            for parameter in layer.parameters():
                parameter.requires_grad = True


def parameter_groups(model: nn.Module, lr: float, head_lr_multiplier: float) -> list[dict]:
    """Discriminative learning rates: the fresh head moves faster than the backbone."""
    backbone = [p for p in model.features.parameters() if p.requires_grad]
    head = list(model.classifier.parameters())
    groups = [{"params": head, "lr": lr * head_lr_multiplier}]
    if backbone:
        groups.append({"params": backbone, "lr": lr})
    return groups


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """``(trainable, total)`` parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def save_checkpoint(path: str | Path, model: nn.Module, metadata: dict) -> None:
    """Persist weights alongside the metadata needed to rebuild the model."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict]:
    """Rebuild a model from a checkpoint written by :func:`save_checkpoint`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {path}\nTrain one with: python -m alzheimers.training.train"
        )

    payload = torch.load(path, map_location=device, weights_only=False)
    metadata = payload.get("metadata", {})
    model = build_model(
        num_classes=len(metadata.get("classes", CLASSES)),
        pretrained=False,  # weights come from the checkpoint, not ImageNet
        dropout=float(metadata.get("dropout", 0.5)),
        trainable_blocks=int(metadata.get("trainable_blocks", 1)),
        hidden_dim=int(metadata.get("hidden_dim", 512)),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, metadata
