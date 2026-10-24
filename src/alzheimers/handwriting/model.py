"""Handwriting risk model: logistic regression on kinematic features.

A linear model is a deliberate choice, not a stand-in. Each prediction splits
exactly into per-feature contributions (coefficient x standardised value), so the
interface can say *which* aspects of the drawing moved the score -- the same job
Grad-CAM does for the MRI branch.

Trained on simulated drawings (see ``synthetic.py``); the cross-validated figures
stored with the model describe that simulator, not patients.

    python -m alzheimers.handwriting.model          # retrain and save
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import REPO_ROOT
from .features import FEATURE_INFO, FEATURE_NAMES, extract_features, speed_trace
from .synthetic import CANVAS, make_cohort

MODEL_PATH = REPO_ROOT / "artifacts" / "handwriting_model.joblib"

# Right-skewed positives: model their logs so one very slow drawing does not
# dominate the scaler.
LOG_FEATURES = {"duration_s", "mean_speed", "speed_peaks", "tremor_amp", "spiral_irregularity"}

TIERS = ((0.35, "Low"), (0.65, "Moderate"), (1.01, "High"))


def _transform(matrix: np.ndarray) -> np.ndarray:
    out = np.array(matrix, dtype=float, copy=True)
    for i, name in enumerate(FEATURE_NAMES):
        if name in LOG_FEATURES:
            out[:, i] = np.log(np.maximum(out[:, i], 1e-4))
    return out


def train(n_per_class: int = 300, seed: int = 0) -> dict:
    """Fit on a fresh simulated cohort and report cross-validated performance."""
    X, y, _ = make_cohort(n_per_class, seed)
    Xt = _transform(X)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=0.03, class_weight="balanced", max_iter=1000))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    proba = cross_val_predict(pipe, Xt, y, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "n_samples": int(len(y)),
        "cv_auc": float(roc_auc_score(y, proba)),
        "cv_accuracy": float(np.mean(pred == y)),
        "cv_sensitivity": float(np.mean(pred[y == 1] == 1)),
        "cv_specificity": float(np.mean(pred[y == 0] == 0)),
        "data": "simulated drawings; not patient data",
    }
    pipe.fit(Xt, y)
    return {"pipeline": pipe, "metrics": metrics, "features": list(FEATURE_NAMES)}


def save(bundle: dict, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    path.with_suffix(".json").write_text(json.dumps(bundle["metrics"], indent=2))


class HandwritingAnalyzer:
    """Loads (or, if absent, trains and caches) the model and scores drawings."""

    def __init__(self, path: Path = MODEL_PATH):
        if path.exists():
            bundle = joblib.load(path)
        else:
            bundle = train()
            try:
                save(bundle, path)
            except OSError:  # read-only filesystem: keep the in-memory model
                pass
        self.pipeline = bundle["pipeline"]
        self.metrics = bundle["metrics"]

    def analyze(self, strokes, canvas_size=CANVAS) -> dict:
        feats = extract_features(strokes, canvas_size)
        raw = np.array([[feats[name] for name in FEATURE_NAMES]])
        scaler = self.pipeline.named_steps["standardscaler"]
        clf = self.pipeline.named_steps["logisticregression"]
        z = scaler.transform(_transform(raw))[0]
        contributions = clf.coef_[0] * z
        probability = float(self.pipeline.predict_proba(_transform(raw))[0, 1])
        tier = next(label for limit, label in TIERS if probability < limit)

        rows = []
        for i, name in enumerate(FEATURE_NAMES):
            info = FEATURE_INFO[name]
            value = feats[name]
            shown = value * 100 if info.unit in {"%", "% of canvas"} else value
            rows.append(
                {
                    "name": name,
                    "label": info.label,
                    "unit": info.unit,
                    "meaning": info.meaning,
                    "value": round(float(shown), 3),
                    "contribution": round(float(contributions[i]), 3),
                }
            )
        rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)

        return {
            "probability": round(probability, 4),
            "tier": tier,
            "features": rows,
            "trace": speed_trace(strokes, canvas_size),
            "canvas": {"width": canvas_size[0], "height": canvas_size[1]},
            "model": {
                "type": "logistic regression",
                "trained_on": self.metrics["data"],
                "cv_auc": round(self.metrics["cv_auc"], 3),
            },
            "disclaimer": (
                "Research prototype trained on simulated drawings. Not a medical device "
                "and not a diagnosis."
            ),
        }


if __name__ == "__main__":
    bundle = train()
    save(bundle)
    print(json.dumps(bundle["metrics"], indent=2))
    print(f"saved to {MODEL_PATH}")
