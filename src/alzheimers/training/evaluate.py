"""Phase 3 entry point: evaluate a checkpoint and generate Grad-CAM evidence.

    python -m alzheimers.training.evaluate --split test --gradcam-samples 8

Writes to ``artifacts/``:

* ``evaluation_<split>.json``  -- metrics, confusion matrix, per-class breakdown
* ``confusion_matrix_<split>.png``
* ``gradcam/``                 -- side-by-side slice / heatmap / overlay panels

The Grad-CAM panels are the deliverable that makes the model reviewable. Sample
both correct and incorrect predictions -- the misclassifications are where you
find out whether the model is reading anatomy or an artefact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display in Docker or CI
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from .. import CLASSES  # noqa: E402
from ..config import Config, set_seed  # noqa: E402
from ..data.datasets import MRISliceDataset, build_transforms  # noqa: E402
from ..models.gradcam import (  # noqa: E402
    GradCAM,
    cam_focus_summary,
    denormalize,
    overlay_cam,
)
from ..models.vgg19 import load_checkpoint  # noqa: E402
from .metrics import compute_metrics, early_detection_recall, format_confusion_matrix  # noqa: E402


def predict_split(model, dataset, device, batch_size=16):
    """Return ``(y_true, y_pred, y_prob)`` over a dataset."""
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    targets, preds, probs = [], [], []

    model.eval()
    with torch.no_grad():
        for images, batch_targets in loader:
            logits = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            targets.append(batch_targets.numpy())
            probs.append(probabilities)
            preds.append(probabilities.argmax(axis=1))

    return np.concatenate(targets), np.concatenate(preds), np.concatenate(probs)


def plot_confusion_matrix(matrix, classes, out_path: Path, title: str) -> None:
    matrix = np.asarray(matrix)
    # Row-normalised: raw counts hide poor recall on a small class.
    normalized = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(6, 5.2))
    image = ax.imshow(normalized, cmap="BuPu", vmin=0, vmax=1)

    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted stage")
    ax.set_ylabel("True stage")
    ax.set_title(title)

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j, i, f"{int(matrix[i, j])}\n{normalized[i, j]:.0%}",
                ha="center", va="center", fontsize=9,
                color="white" if normalized[i, j] > 0.55 else "#222222",
            )

    fig.colorbar(image, ax=ax, label="share of true class")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_gradcam_panels(
    model, dataset, indices, device, out_dir: Path, cfg: Config
) -> list[dict]:
    """Write a slice / heatmap / overlay panel per sampled index."""
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    with GradCAM(model, str(cfg["gradcam.target_layer"])) as explainer:
        for index in indices:
            image_tensor, target = dataset[index]
            cam, predicted, probabilities = explainer(image_tensor.to(device))

            original = denormalize(image_tensor)
            blended = overlay_cam(original, cam, alpha=float(cfg["gradcam.alpha"]))

            true_name, pred_name = CLASSES[target], CLASSES[predicted]
            correct = target == predicted

            fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))
            for ax, (image, caption) in zip(
                axes,
                [
                    (original, "MRI slice"),
                    (cam, "Grad-CAM"),
                    (blended, "Overlay"),
                ],
            ):
                ax.imshow(image, cmap="inferno" if caption == "Grad-CAM" else None)
                ax.set_title(caption, fontsize=10)
                ax.axis("off")

            verdict = "correct" if correct else "MISCLASSIFIED"
            fig.suptitle(
                f"true {true_name}  |  predicted {pred_name} "
                f"({probabilities[predicted]:.1%})  --  {verdict}",
                fontsize=11,
                color="#1a7f37" if correct else "#b3261e",
            )
            fig.tight_layout()

            filename = f"{index:05d}_{true_name}_as_{pred_name}.png"
            fig.savefig(out_dir / filename, dpi=140, bbox_inches="tight")
            plt.close(fig)

            records.append(
                {
                    "index": int(index),
                    "source": dataset.manifest.iloc[index]["path"],
                    "true": true_name,
                    "predicted": pred_name,
                    "confidence": round(float(probabilities[predicted]), 4),
                    "correct": bool(correct),
                    "panel": filename,
                    "focus": cam_focus_summary(cam),
                }
            )

    return records


def sample_indices(y_true, y_pred, n: int, seed: int) -> list[int]:
    """Half correct, half wrong -- the errors are the informative half."""
    rng = np.random.default_rng(seed)
    correct = np.where(y_true == y_pred)[0]
    wrong = np.where(y_true != y_pred)[0]

    n_wrong = min(len(wrong), n // 2)
    n_correct = min(len(correct), n - n_wrong)
    chosen = np.concatenate(
        [
            rng.choice(wrong, n_wrong, replace=False),
            rng.choice(correct, n_correct, replace=False),
        ]
    )
    return sorted(int(i) for i in chosen)


def evaluate(cfg: Config, split: str, gradcam_samples: int, device_str: str | None) -> dict:
    set_seed(int(cfg["seed"]))
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))

    manifest_path = cfg.path("paths.processed_dir") / f"{split}_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} missing. Run: python -m alzheimers.data.prepare"
        )

    dataset = MRISliceDataset(pd.read_csv(manifest_path), build_transforms(cfg, train=False))
    model, metadata = load_checkpoint(cfg.path("api.checkpoint"), device)
    print(f"[eval] {split}: {len(dataset)} slices, checkpoint epoch {metadata.get('epoch', '?')}")

    y_true, y_pred, y_prob = predict_split(model, dataset, device, int(cfg["data.batch_size"]))
    metrics = compute_metrics(y_true, y_pred, y_prob, CLASSES)
    metrics["early_detection_recall"] = early_detection_recall(y_true, y_pred, CLASSES)
    metrics["split"] = split
    metrics["n_samples"] = len(dataset)

    print(f"\naccuracy {metrics['accuracy']:.4f}   macro F1 {metrics['macro_f1']:.4f}   "
          f"quadratic kappa {metrics['quadratic_kappa']:.4f}")
    print(f"impaired-case recall {metrics['early_detection_recall']:.4f}\n")
    print(format_confusion_matrix(metrics["confusion_matrix"], CLASSES))
    print()
    for name, scores in metrics["per_class"].items():
        print(
            f"  {name:<5} precision {scores['precision']:.3f}  "
            f"recall {scores['recall']:.3f}  f1 {scores['f1']:.3f}  (n={scores['support']})"
        )

    artifacts = cfg.path("paths.artifacts_dir")
    artifacts.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(
        metrics["confusion_matrix"], CLASSES,
        artifacts / f"confusion_matrix_{split}.png",
        f"Confusion matrix -- {split} split",
    )

    if gradcam_samples > 0:
        indices = sample_indices(y_true, y_pred, gradcam_samples, int(cfg["seed"]))
        metrics["gradcam"] = save_gradcam_panels(
            model, dataset, indices, device, artifacts / "gradcam", cfg
        )
        print(f"\n[eval] wrote {len(indices)} Grad-CAM panels to {artifacts / 'gradcam'}")

    report_path = artifacts / f"evaluation_{split}.json"
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[eval] wrote {report_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--gradcam-samples", type=int, default=8)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    evaluate(Config.load(args.config), args.split, args.gradcam_samples, args.device)


if __name__ == "__main__":
    main()
