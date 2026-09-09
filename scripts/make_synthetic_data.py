"""Generate a synthetic stand-in cohort so the pipeline runs end to end.

    python scripts/make_synthetic_data.py --n-per-class 200

ADNI access needs an approved application, which takes weeks. This script
produces data with the same *shape* as the real thing so every stage --
imputation, ADASYN, training, Grad-CAM, the API, the dashboard -- can be
developed and tested today:

* **Images**: phantom axial slices. Progressive stages get enlarged ventricles,
  a thinner cortical band, and a smaller hippocampal region -- the same coarse
  atrophy signal a CNN keys on, minus the anatomy.
* **Clinical table**: MMSE / CDR / ADAS-Cog / hippocampal volume / APOE4 with
  realistic stage-dependent distributions, deliberate class imbalance, and
  values missing at higher rates in the later stages (which is what makes
  hot-deck imputation worth doing rather than dropping rows).

These images are NOT brain scans and a model trained on them tells you nothing
clinical. They exist to prove the plumbing works before the real data lands.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("NC", "EMCI", "LMCI", "AD")

# Per stage: (ventricle scale, cortical thickness, hippocampus scale, cohort share)
STAGE_PROFILE = {
    "NC":   {"ventricle": 0.16, "cortex": 0.085, "hippocampus": 1.00, "share": 0.42},
    "EMCI": {"ventricle": 0.21, "cortex": 0.072, "hippocampus": 0.90, "share": 0.26},
    "LMCI": {"ventricle": 0.27, "cortex": 0.058, "hippocampus": 0.78, "share": 0.19},
    "AD":   {"ventricle": 0.35, "cortex": 0.044, "hippocampus": 0.62, "share": 0.13},
}

# Per stage: mean and sd for each clinical measure.
CLINICAL_PROFILE = {
    "NC":   {"mmse": (29.0, 1.0), "cdr": (0.02, 0.05), "adas": (8.0, 3.0),
             "hippo_vol": (7300, 550), "apoe4": 0.25, "age": (73, 6)},
    "EMCI": {"mmse": (28.0, 1.4), "cdr": (0.45, 0.12), "adas": (12.0, 4.0),
             "hippo_vol": (6700, 600), "apoe4": 0.42, "age": (74, 6)},
    "LMCI": {"mmse": (26.5, 2.0), "cdr": (0.55, 0.18), "adas": (17.0, 5.0),
             "hippo_vol": (6100, 650), "apoe4": 0.55, "age": (75, 7)},
    "AD":   {"mmse": (21.5, 3.4), "cdr": (0.95, 0.30), "adas": (28.0, 8.0),
             "hippo_vol": (5300, 700), "apoe4": 0.68, "age": (76, 7)},
}

# Missingness rates. CSF biomarkers are the classic sparse column, and dropout
# is worse in the impaired groups -- exactly the pattern that biases a model if
# you handle it by deleting incomplete rows.
BASE_MISSING = {"adas": 0.10, "hippo_vol": 0.22, "apoe4": 0.30, "csf_abeta": 0.45,
                "mmse": 0.02, "cdr": 0.02, "age": 0.0}
MISSING_STAGE_PENALTY = {"NC": 0.0, "EMCI": 0.03, "LMCI": 0.06, "AD": 0.10}


def make_slice(rng: np.random.Generator, profile: dict, size: int = 224) -> np.ndarray:
    """One phantom axial slice for a given stage profile."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cy = cx = size / 2
    radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (size / 2)

    jitter = lambda scale: float(rng.normal(0, scale))  # noqa: E731 - local shorthand

    # Skull ring and brain parenchyma.
    skull_outer = 0.92 + jitter(0.015)
    skull_inner = skull_outer - (profile["cortex"] + jitter(0.006))
    image = np.zeros((size, size), dtype=np.float32)
    image[radius < skull_outer] = 0.55                       # skull / scalp
    image[radius < skull_inner] = 0.34                       # grey matter
    image[radius < skull_inner - 0.10] = 0.46                # white matter

    # Lateral ventricles: two ellipses that widen with disease stage.
    v = profile["ventricle"] + jitter(0.012)
    for sign in (-1, 1):
        ex = (xx - (cx + sign * size * 0.085)) / (size * v * 0.42)
        ey = (yy - (cy - size * 0.02)) / (size * v * 0.95)
        image[(ex ** 2 + ey ** 2) < 1.0] = 0.06              # CSF is dark on T1

    # Hippocampal regions: shrink with stage.
    h = profile["hippocampus"]
    for sign in (-1, 1):
        ex = (xx - (cx + sign * size * 0.155)) / (size * 0.052 * h)
        ey = (yy - (cy + size * 0.085)) / (size * 0.034 * h)
        image[(ex ** 2 + ey ** 2) < 1.0] = 0.30

    # Sulcal widening: faint dark radial lines, more of them in later stages.
    n_sulci = int(6 + (1 - profile["hippocampus"]) * 26)
    for angle in rng.uniform(0, 2 * np.pi, n_sulci):
        depth = rng.uniform(0.10, 0.30)
        for t in np.linspace(skull_inner - depth, skull_inner, 26):
            px, py = int(cx + t * cx * np.cos(angle)), int(cy + t * cy * np.sin(angle))
            if 1 <= px < size - 1 and 1 <= py < size - 1:
                image[py - 1 : py + 1, px - 1 : px + 1] *= 0.72

    # Acquisition noise and a mild intensity bias field.
    image += rng.normal(0, 0.022, image.shape)
    bias = 1.0 + 0.06 * ((xx - cx) / size + (yy - cy) / size)
    image = np.clip(image * bias, 0, 1)

    return (image * 255).astype(np.uint8)


def generate_images(out_dir: Path, counts: dict[str, int], seed: int, size: int) -> int:
    rng = np.random.default_rng(seed)
    total = 0
    for label, n in counts.items():
        class_dir = out_dir / label
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            array = make_slice(rng, STAGE_PROFILE[label], size)
            Image.fromarray(array).save(class_dir / f"{label}_{i:05d}.png")
        print(f"[images] {label:<5} {n:>5} slices -> {class_dir}")
        total += n
    return total


def generate_clinical(out_path: Path, counts: dict[str, int], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    rows = []

    for label, n in counts.items():
        profile = CLINICAL_PROFILE[label]
        for i in range(n):
            rows.append(
                {
                    "subject_id": f"{label}_{i:05d}",
                    "age": round(float(rng.normal(*profile["age"])), 1),
                    "mmse": float(np.clip(rng.normal(*profile["mmse"]), 0, 30).round(1)),
                    "cdr": float(np.clip(rng.normal(*profile["cdr"]), 0, 3).round(2)),
                    "adas_cog": float(np.clip(rng.normal(*profile["adas"]), 0, 70).round(1)),
                    "hippo_vol": round(float(rng.normal(*profile["hippo_vol"]))),
                    "apoe4": int(rng.random() < profile["apoe4"]),
                    # Amyloid-beta drops as pathology advances.
                    "csf_abeta": float(
                        np.clip(rng.normal(1100 - 190 * CLASSES.index(label), 190), 200, 2000).round(0)
                    ),
                    "label": label,
                }
            )

    frame = pd.DataFrame(rows)

    # Punch holes: missingness rate rises with stage severity.
    column_map = {"adas": "adas_cog", "hippo_vol": "hippo_vol", "apoe4": "apoe4",
                  "csf_abeta": "csf_abeta", "mmse": "mmse", "cdr": "cdr", "age": "age"}
    for key, rate in BASE_MISSING.items():
        column = column_map[key]
        for label in CLASSES:
            mask = frame["label"] == label
            effective = min(rate + MISSING_STAGE_PENALTY[label], 0.85)
            drop = mask & (rng.random(len(frame)) < effective)
            frame.loc[drop, column] = np.nan

    # Hot-deck needs at least some fully observed donors per class; guarantee a
    # complete core so the imputer never falls back to medians everywhere.
    feature_columns = [c for c in frame.columns if c not in ("subject_id", "label")]
    for label in CLASSES:
        idx = frame.index[frame["label"] == label][: max(int(counts[label] * 0.25), 12)]
        complete = generate_clinical_rows(rng, label, len(idx))
        frame.loc[idx, feature_columns] = complete[feature_columns].to_numpy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)

    missing = int(frame[feature_columns].isna().sum().sum())
    cells = len(frame) * len(feature_columns)
    print(f"[clinical] {len(frame)} records -> {out_path}")
    print(f"[clinical] {missing}/{cells} cells missing ({missing / cells:.1%})")
    return frame


def generate_clinical_rows(rng: np.random.Generator, label: str, n: int) -> pd.DataFrame:
    """n complete (no-NaN) clinical rows for one stage."""
    profile = CLINICAL_PROFILE[label]
    return pd.DataFrame(
        {
            "age": rng.normal(*profile["age"], size=n).round(1),
            "mmse": np.clip(rng.normal(*profile["mmse"], size=n), 0, 30).round(1),
            "cdr": np.clip(rng.normal(*profile["cdr"], size=n), 0, 3).round(2),
            "adas_cog": np.clip(rng.normal(*profile["adas"], size=n), 0, 70).round(1),
            "hippo_vol": rng.normal(*profile["hippo_vol"], size=n).round(0),
            "apoe4": (rng.random(n) < profile["apoe4"]).astype(int),
            "csf_abeta": np.clip(
                rng.normal(1100 - 190 * CLASSES.index(label), 190, size=n), 200, 2000
            ).round(0),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="data/raw", help="output directory")
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=150,
        help="slices for the largest class; smaller classes scale by cohort share",
    )
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="equal counts per class (defeats the point of the ADASYN stage)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    if args.balanced:
        counts = {label: args.n_per_class for label in CLASSES}
    else:
        largest = max(p["share"] for p in STAGE_PROFILE.values())
        counts = {
            label: max(int(args.n_per_class * STAGE_PROFILE[label]["share"] / largest), 8)
            for label in CLASSES
        }

    print(f"Cohort: {counts}  (imbalance {max(counts.values()) / min(counts.values()):.2f}x)\n")
    total = generate_images(out_dir, counts, args.seed, args.size)
    generate_clinical(out_dir / "clinical.csv", counts, args.seed)

    print(f"\nDone: {total} images + clinical table in {out_dir}")
    print("Next: python -m alzheimers.data.prepare")


if __name__ == "__main__":
    main()
