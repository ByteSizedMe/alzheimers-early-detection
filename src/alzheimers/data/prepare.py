"""Phase 1 entry point: build the processed dataset.

    python -m alzheimers.data.prepare --config configs/default.yaml

Does three things and writes everything it produces to ``paths.processed_dir``:

1. Scans the MRI directory and writes a stratified train/val/test manifest.
2. Hot-deck imputes the clinical table (fitted on the training split only, so
   validation and test records cannot borrow values from their own rows).
3. ADASYN-resamples the *training* clinical features to balance the classes.

Validation and test sets are never resampled -- doing so would inflate metrics
against synthetic rows the model has effectively already seen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import CLASSES
from ..config import Config, set_seed
from .balancing import adasyn, class_distribution, imbalance_ratio
from .datasets import scan_image_directory, stratified_split
from .imputation import HotDeckImputer, missingness_report

LABEL_COLUMN = "label"
ID_COLUMNS = ("subject_id", "image_id", "path")


def prepare_images(cfg: Config, out_dir: Path) -> dict[str, pd.DataFrame]:
    manifest = scan_image_directory(cfg.path("paths.raw_dir"), CLASSES)
    print(f"[images] found {len(manifest)} slices: {class_distribution(manifest['label'])}")

    splits = stratified_split(
        manifest,
        val_fraction=float(cfg["data.val_fraction"]),
        test_fraction=float(cfg["data.test_fraction"]),
        seed=int(cfg["seed"]),
    )
    for name, frame in splits.items():
        destination = out_dir / f"{name}_manifest.csv"
        frame.to_csv(destination, index=False)
        print(f"[images] {name:<5} n={len(frame):<5} -> {destination.name}")
    return splits


def prepare_clinical(cfg: Config, out_dir: Path) -> dict | None:
    clinical_path = cfg.path("paths.clinical_csv")
    if not clinical_path.exists():
        print(f"[clinical] {clinical_path.name} not found -- skipping tabular stage")
        return None

    frame = pd.read_csv(clinical_path)
    if LABEL_COLUMN not in frame.columns:
        raise ValueError(f"{clinical_path} must contain a '{LABEL_COLUMN}' column")

    feature_columns = [
        column
        for column in frame.columns
        if column != LABEL_COLUMN and column not in ID_COLUMNS
    ]
    features = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    labels = frame[LABEL_COLUMN]

    report = missingness_report(features)
    print(f"[clinical] {len(frame)} records, {len(feature_columns)} features")
    print(f"[clinical] missing values: {int(report['missing'].sum())} cells")
    for column, row in report[report["missing"] > 0].iterrows():
        print(f"           {column:<22} {int(row['missing']):>5}  ({row['rate']:.1%})")

    # Split first so the imputer never sees held-out rows.
    indices = np.arange(len(frame))
    rng = np.random.default_rng(int(cfg["seed"]))
    rng.shuffle(indices)
    n_holdout = int(len(frame) * (float(cfg["data.val_fraction"]) + float(cfg["data.test_fraction"])))
    holdout_idx, train_idx = indices[:n_holdout], indices[n_holdout:]

    imputer = HotDeckImputer(
        n_donors=int(cfg["imputation.n_donors"]),
        stratify_by_class=cfg["imputation.donor_pool"] == "class",
        scale=bool(cfg["imputation.scale"]),
        random_state=int(cfg["seed"]),
    )
    train_features = imputer.fit_transform(
        features.iloc[train_idx], labels.iloc[train_idx]
    )
    # No labels at transform time for held-out rows: that is what inference sees.
    holdout_features = imputer.transform(features.iloc[holdout_idx])
    print(f"[clinical] hot-deck imputed; residual NaN: {int(train_features.isna().sum().sum())}")

    before = class_distribution(labels.iloc[train_idx])
    print(f"[clinical] train distribution {before} (imbalance {imbalance_ratio(labels.iloc[train_idx]):.2f}x)")

    if cfg["balancing.method"] == "adasyn":
        resampled_x, resampled_y = adasyn(
            train_features.to_numpy(),
            labels.iloc[train_idx].to_numpy(),
            n_neighbors=int(cfg["balancing.n_neighbors"]),
            beta=float(cfg["balancing.beta"]),
            random_state=int(cfg["seed"]),
        )
        after = class_distribution(resampled_y)
        print(f"[clinical] ADASYN -> {after} (+{len(resampled_y) - len(train_idx)} synthetic)")
    else:
        resampled_x, resampled_y = train_features.to_numpy(), labels.iloc[train_idx].to_numpy()
        after = before

    balanced = pd.DataFrame(resampled_x, columns=feature_columns)
    balanced[LABEL_COLUMN] = resampled_y
    balanced.to_csv(out_dir / "clinical_train_balanced.csv", index=False)

    holdout = holdout_features.copy()
    holdout[LABEL_COLUMN] = labels.iloc[holdout_idx].to_numpy()
    holdout.to_csv(out_dir / "clinical_holdout.csv", index=False)

    return {
        "features": feature_columns,
        "n_train": int(len(train_idx)),
        "n_holdout": int(len(holdout_idx)),
        "distribution_before": {str(k): int(v) for k, v in before.items()},
        "distribution_after": {str(k): int(v) for k, v in after.items()},
        "missing_cells": int(report["missing"].sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MRI + clinical data")
    parser.add_argument("--config", default=None, help="path to a YAML config")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    set_seed(int(cfg["seed"]))

    out_dir = cfg.path("paths.processed_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = prepare_images(cfg, out_dir)
    clinical_summary = prepare_clinical(cfg, out_dir)

    summary = {
        "config": str(cfg.source),
        "classes": list(CLASSES),
        "images": {name: len(frame) for name, frame in splits.items()},
        "image_distribution": {
            name: {str(k): int(v) for k, v in class_distribution(frame["label"]).items()}
            for name, frame in splits.items()
        },
        "clinical": clinical_summary,
    }
    summary_path = out_dir / "prepare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
