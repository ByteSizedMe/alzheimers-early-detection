"""ADASYN oversampling for imbalanced diagnostic cohorts.

ADNI-like cohorts are dominated by normal controls and stable MCI; the AD and
LMCI classes -- the ones a screening tool exists to catch -- are the minority.
A model trained on the raw distribution reaches high accuracy by predicting the
majority class and is clinically useless.

ADASYN (He et al., 2008, "ADASYN: Adaptive Synthetic Sampling Approach for
Imbalanced Learning") fixes this by generating synthetic minority samples, but
unlike SMOTE it allocates them *adaptively*: minority points sitting in
majority-dense neighbourhoods -- the hard, near-boundary cases -- receive more
synthetic neighbours than ones deep inside a safe minority cluster. That pushes
the decision boundary towards the region that actually matters.

Implemented directly against scikit-learn's NearestNeighbors so the project has
no imbalanced-learn dependency, and extended to the multi-class case by
resampling each minority class against the majority class in turn.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.neighbors import NearestNeighbors


def adasyn(
    features: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int = 5,
    beta: float = 1.0,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Adaptively oversample every minority class.

    Parameters
    ----------
    features:
        ``(n_samples, n_features)`` numeric matrix. Must not contain NaN --
        run :class:`~alzheimers.data.imputation.HotDeckImputer` first.
    labels:
        ``(n_samples,)`` class labels of any hashable dtype.
    n_neighbors:
        Neighbourhood size for both the difficulty ratio and the interpolation
        partner. Clipped down when a class has fewer members than that.
    beta:
        Degree of balancing. ``1.0`` brings every minority class up to the
        majority count; ``0.5`` closes half the gap.
    random_state:
        Seed for the interpolation draws.

    Returns
    -------
    ``(features_resampled, labels_resampled)`` -- the originals with synthetic
    rows appended. Row order is original-then-synthetic; shuffle downstream.
    """
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels)

    if features.ndim != 2:
        raise ValueError(f"features must be 2-D, got shape {features.shape}")
    if len(features) != len(labels):
        raise ValueError("features and labels must have the same length")
    if np.isnan(features).any():
        raise ValueError("ADASYN cannot run on data containing NaN; impute first")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")

    rng = np.random.default_rng(random_state)
    counts = Counter(labels.tolist())
    majority_label, majority_count = counts.most_common(1)[0]

    synth_features: list[np.ndarray] = []
    synth_labels: list[object] = []

    for label, count in counts.items():
        if label == majority_label:
            continue

        n_to_generate = int(round((majority_count - count) * beta))
        if n_to_generate <= 0 or count < 2:
            # A class with a single member has no partner to interpolate with;
            # duplicating it would only add noise-free copies.
            continue

        minority = features[labels == label]

        # --- Step 1: difficulty ratio r_i = (majority neighbours) / k --------
        k_global = min(n_neighbors, len(features) - 1)
        global_nn = NearestNeighbors(n_neighbors=k_global + 1).fit(features)
        _, global_idx = global_nn.kneighbors(minority)
        global_idx = global_idx[:, 1:]  # drop the point itself
        majority_fraction = (labels[global_idx] != label).mean(axis=1)

        total = majority_fraction.sum()
        if total == 0:
            # Perfectly separated minority cluster: nothing is "hard", so fall
            # back to uniform allocation rather than skipping the class.
            weights = np.full(len(minority), 1.0 / len(minority))
        else:
            weights = majority_fraction / total

        allocation = _largest_remainder(weights, n_to_generate)

        # --- Step 2: interpolate towards a same-class neighbour --------------
        k_local = min(n_neighbors, len(minority) - 1)
        local_nn = NearestNeighbors(n_neighbors=k_local + 1).fit(minority)
        _, local_idx = local_nn.kneighbors(minority)
        local_idx = local_idx[:, 1:]

        for i, n_i in enumerate(allocation):
            if n_i == 0:
                continue
            partners = minority[rng.choice(local_idx[i], size=n_i)]
            gaps = rng.random((n_i, 1))
            synth_features.append(minority[i] + gaps * (partners - minority[i]))
            synth_labels.extend([label] * n_i)

    if not synth_features:
        return features, labels

    return (
        np.vstack([features, np.vstack(synth_features)]),
        np.concatenate([labels, np.asarray(synth_labels, dtype=labels.dtype)]),
    )


def _largest_remainder(weights: np.ndarray, total: int) -> np.ndarray:
    """Apportion ``total`` integer slots across ``weights`` summing exactly.

    Plain ``round(w * total)`` loses or gains a few samples; the largest
    remainder method distributes the leftovers to the biggest fractional parts
    so the generated count matches the requested count exactly.
    """
    exact = weights * total
    base = np.floor(exact).astype(int)
    shortfall = total - base.sum()
    if shortfall > 0:
        order = np.argsort(-(exact - base))
        base[order[:shortfall]] += 1
    return base


def class_distribution(labels) -> dict:
    """Ordered ``{label: count}`` for logging before/after resampling."""
    counts = Counter(np.asarray(labels).tolist())
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0]))))


def imbalance_ratio(labels) -> float:
    """Majority-to-minority ratio; 1.0 means perfectly balanced."""
    counts = class_distribution(labels)
    if not counts:
        return 1.0
    values = list(counts.values())
    return max(values) / max(min(values), 1)
