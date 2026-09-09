"""Classification metrics for a 4-stage ordinal diagnosis.

Accuracy is the wrong headline number here. With a cohort that is mostly normal
controls, a model that never predicts AD still scores well. Macro F1 weights
every stage equally, and the confusion matrix is what actually tells you whether
EMCI is being caught or quietly folded into NC.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None, classes: tuple[str, ...]
) -> dict:
    """Headline metrics plus per-class breakdown."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        # Quadratic weights: mistaking AD for NC should cost far more than
        # mistaking EMCI for LMCI, and the stages are ordered.
        "quadratic_kappa": float(
            cohen_kappa_score(y_true, y_pred, weights="quadratic")
            if len(np.unique(y_true)) > 1
            else 0.0
        ),
    }

    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["macro_auc"] = float(
                roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="macro",
                    labels=list(range(len(classes))),
                )
            )
        except ValueError:
            # A split missing a class entirely makes one-vs-rest AUC undefined.
            metrics["macro_auc"] = float("nan")

    per_class = classification_report(
        y_true, y_pred,
        labels=list(range(len(classes))), target_names=list(classes),
        output_dict=True, zero_division=0,
    )
    metrics["per_class"] = {
        name: {
            "precision": round(float(per_class[name]["precision"]), 4),
            "recall": round(float(per_class[name]["recall"]), 4),
            "f1": round(float(per_class[name]["f1-score"]), 4),
            "support": int(per_class[name]["support"]),
        }
        for name in classes
    }
    metrics["confusion_matrix"] = confusion_matrix(
        y_true, y_pred, labels=list(range(len(classes)))
    ).tolist()
    return metrics


def early_detection_recall(
    y_true: np.ndarray, y_pred: np.ndarray, classes: tuple[str, ...]
) -> float:
    """Fraction of any-impairment cases (EMCI/LMCI/AD) not called normal.

    The clinically load-bearing number: a screening model's job is to not send
    an impaired patient home. Collapses the 4-way problem to impaired vs not.
    """
    healthy_index = classes.index("NC") if "NC" in classes else 0
    impaired = np.asarray(y_true) != healthy_index
    if not impaired.any():
        return float("nan")
    return float((np.asarray(y_pred)[impaired] != healthy_index).mean())


def format_confusion_matrix(matrix, classes: tuple[str, ...]) -> str:
    """Readable confusion matrix; rows are true, columns predicted."""
    matrix = np.asarray(matrix)
    width = max(max(len(c) for c in classes), 5) + 2
    header = " " * width + "".join(f"{c:>{width}}" for c in classes)
    lines = [f"{'':<{width}}{'-- predicted --':>{width * len(classes)}}", header]
    for name, row in zip(classes, matrix):
        lines.append(f"{name:<{width}}" + "".join(f"{int(v):>{width}}" for v in row))
    return "\n".join(lines)
