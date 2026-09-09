"""MRI slice datasets and dataloaders.

Images are expected as a manifest CSV with ``path`` and ``label`` columns,
produced by ``python -m alzheimers.data.prepare``. Working from a manifest
rather than walking directories at train time keeps the train/val/test split
reproducible and makes it obvious which scans a checkpoint was fitted on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .. import CLASSES

# ImageNet statistics -- required because the VGG19 backbone is pretrained.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class MRISliceDataset(Dataset):
    """Labelled 2-D MRI slices read from a manifest."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        transform: transforms.Compose,
        classes: tuple[str, ...] = CLASSES,
    ):
        missing = set(manifest["label"]) - set(classes)
        if missing:
            raise ValueError(f"manifest contains unknown labels: {sorted(missing)}")

        self.manifest = manifest.reset_index(drop=True)
        self.transform = transform
        self.classes = tuple(classes)
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.targets = [self.class_to_idx[label] for label in self.manifest["label"]]

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.manifest.iloc[index]
        # MRI slices are single-channel; VGG19 wants 3, so replicate on load.
        image = Image.open(row["path"]).convert("RGB")
        return self.transform(image), self.targets[index]

    def class_counts(self) -> dict[str, int]:
        counts = self.manifest["label"].value_counts().to_dict()
        return {name: int(counts.get(name, 0)) for name in self.classes}


def build_transforms(cfg, train: bool) -> transforms.Compose:
    """Training transforms include augmentation; eval transforms never do."""
    size = int(cfg["data.image_size"])

    if not train:
        return transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    aug = cfg["data.augment"]
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=float(aug["horizontal_flip"])),
            transforms.RandomAffine(
                degrees=float(aug["rotation_degrees"]),
                translate=(float(aug["translate"]), float(aug["translate"])),
            ),
            transforms.ColorJitter(
                brightness=float(aug["brightness"]), contrast=float(aug["contrast"])
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def scan_image_directory(root: Path, classes: tuple[str, ...] = CLASSES) -> pd.DataFrame:
    """Build a manifest from ``root/<CLASS>/*.png`` style folders."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"image directory not found: {root}\n"
            "Run scripts/make_synthetic_data.py for a runnable stand-in, or point "
            "paths.raw_dir at your ADNI export."
        )

    rows = []
    for label in classes:
        class_dir = root / label
        if not class_dir.is_dir():
            continue
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() in IMAGE_SUFFIXES:
                rows.append({"path": str(image_path), "label": label})

    if not rows:
        raise FileNotFoundError(
            f"no images under {root}; expected class subfolders {list(classes)}"
        )
    return pd.DataFrame(rows)


def stratified_split(
    manifest: pd.DataFrame,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Split into train/val/test, preserving the class ratio in each part."""
    if not 0 < val_fraction + test_fraction < 1:
        raise ValueError("val_fraction + test_fraction must be between 0 and 1")

    train_df, holdout = train_test_split(
        manifest,
        test_size=val_fraction + test_fraction,
        stratify=manifest["label"],
        random_state=seed,
    )
    # Re-normalise so test_fraction stays a fraction of the *full* dataset.
    test_share = test_fraction / (val_fraction + test_fraction)
    val_df, test_df = train_test_split(
        holdout,
        test_size=test_share,
        stratify=holdout["label"],
        random_state=seed,
    )
    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def build_dataloaders(cfg, splits: dict[str, pd.DataFrame]) -> dict[str, DataLoader]:
    """One DataLoader per split; only the training loader shuffles."""
    loaders = {}
    for name, frame in splits.items():
        is_train = name == "train"
        dataset = MRISliceDataset(frame, build_transforms(cfg, train=is_train))
        loaders[name] = DataLoader(
            dataset,
            batch_size=int(cfg["data.batch_size"]),
            shuffle=is_train,
            num_workers=int(cfg["data.num_workers"]),
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
    return loaders


def compute_class_weights(manifest: pd.DataFrame, classes: tuple[str, ...] = CLASSES) -> torch.Tensor:
    """Inverse-frequency weights for the loss, normalised to mean 1.

    This is the image-side counterpart to ADASYN: rather than synthesising MRI
    slices (which would fabricate anatomy), we make the rare classes cost more
    when the model gets them wrong.
    """
    counts = manifest["label"].value_counts()
    frequencies = np.array([max(int(counts.get(name, 0)), 1) for name in classes], dtype=float)
    weights = frequencies.sum() / (len(classes) * frequencies)
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)
