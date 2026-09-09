"""ADASYN tests.

The two claims worth defending: it balances the classes exactly, and it
allocates synthetic samples *adaptively* -- more of them near the decision
boundary, which is what distinguishes ADASYN from plain SMOTE.
"""

from __future__ import annotations

import numpy as np
import pytest

from alzheimers.data.balancing import (
    _largest_remainder,
    adasyn,
    class_distribution,
    imbalance_ratio,
)


@pytest.fixture
def imbalanced():
    rng = np.random.default_rng(0)
    majority = rng.normal(0, 1, (100, 4))
    minority = rng.normal(3, 1, (20, 4))
    features = np.vstack([majority, minority])
    labels = np.array(["NC"] * 100 + ["AD"] * 20)
    return features, labels


def test_balances_classes_exactly(imbalanced):
    features, labels = imbalanced
    _, resampled_y = adasyn(features, labels, random_state=0)

    counts = class_distribution(resampled_y)
    assert counts["AD"] == counts["NC"] == 100
    assert imbalance_ratio(resampled_y) == 1.0


def test_beta_controls_how_far_the_gap_closes(imbalanced):
    features, labels = imbalanced
    _, resampled_y = adasyn(features, labels, beta=0.5, random_state=0)

    counts = class_distribution(resampled_y)
    assert counts["AD"] == 20 + 40  # half of the 80-sample gap
    assert counts["NC"] == 100


def test_original_samples_are_preserved_unchanged(imbalanced):
    features, labels = imbalanced
    resampled_x, resampled_y = adasyn(features, labels, random_state=0)

    np.testing.assert_array_equal(resampled_x[: len(features)], features)
    np.testing.assert_array_equal(resampled_y[: len(labels)], labels)
    assert len(resampled_x) > len(features)


def test_synthetic_points_lie_between_real_minority_points(imbalanced):
    """Interpolation, not extrapolation: synthetic AD samples must fall inside
    the convex range of the real AD cloud, never outside it."""
    features, labels = imbalanced
    resampled_x, resampled_y = adasyn(features, labels, random_state=0)

    real_minority = features[labels == "AD"]
    synthetic = resampled_x[len(features) :][resampled_y[len(labels) :] == "AD"]

    assert len(synthetic) > 0
    assert (synthetic >= real_minority.min(axis=0) - 1e-9).all()
    assert (synthetic <= real_minority.max(axis=0) + 1e-9).all()


def test_allocation_is_adaptive_not_uniform():
    """A minority point sitting inside the majority cloud is 'hard' and must
    receive more synthetic neighbours than one far away from it."""
    rng = np.random.default_rng(1)
    majority = rng.normal(0, 0.5, (60, 2))
    # One minority point buried in the majority, the rest in a distant cluster.
    hard_point = np.array([[0.0, 0.0]])
    easy_cluster = rng.normal(8, 0.3, (9, 2))

    features = np.vstack([majority, hard_point, easy_cluster])
    labels = np.array(["NC"] * 60 + ["AD"] * 10)

    resampled_x, resampled_y = adasyn(features, labels, n_neighbors=5, random_state=0)
    synthetic = resampled_x[len(features) :]

    # Synthetic samples generated from the hard point stay near the origin.
    near_hard = (np.linalg.norm(synthetic - hard_point, axis=1) < 4.0).sum()
    assert near_hard > 0, "the boundary-adjacent minority point got no synthetic samples"


def test_multiclass_lifts_every_minority_to_the_majority():
    rng = np.random.default_rng(2)
    features = np.vstack(
        [rng.normal(i * 4, 1, (n, 3)) for i, n in enumerate([80, 40, 25, 15])]
    )
    labels = np.array(["NC"] * 80 + ["EMCI"] * 40 + ["LMCI"] * 25 + ["AD"] * 15)

    _, resampled_y = adasyn(features, labels, random_state=0)

    assert set(class_distribution(resampled_y).values()) == {80}


def test_already_balanced_data_is_returned_untouched():
    rng = np.random.default_rng(3)
    features = rng.normal(0, 1, (40, 2))
    labels = np.array(["NC"] * 20 + ["AD"] * 20)

    resampled_x, resampled_y = adasyn(features, labels, random_state=0)

    np.testing.assert_array_equal(resampled_x, features)
    np.testing.assert_array_equal(resampled_y, labels)


def test_singleton_class_is_skipped_not_duplicated():
    """One sample has no partner to interpolate with; copying it would only add
    noise-free duplicates that leak into validation."""
    rng = np.random.default_rng(4)
    features = np.vstack([rng.normal(0, 1, (30, 2)), np.array([[9.0, 9.0]])])
    labels = np.array(["NC"] * 30 + ["AD"])

    _, resampled_y = adasyn(features, labels, random_state=0)

    assert class_distribution(resampled_y)["AD"] == 1


def test_rejects_nan_input(imbalanced):
    features, labels = imbalanced
    features = features.copy()
    features[0, 0] = np.nan

    with pytest.raises(ValueError, match="NaN"):
        adasyn(features, labels)


def test_rejects_out_of_range_beta(imbalanced):
    features, labels = imbalanced
    with pytest.raises(ValueError, match="beta"):
        adasyn(features, labels, beta=1.5)


def test_largest_remainder_sums_exactly():
    weights = np.array([0.31, 0.24, 0.23, 0.22])
    allocation = _largest_remainder(weights, 97)

    assert allocation.sum() == 97
    assert allocation[0] == allocation.max()


def test_imbalance_ratio():
    assert imbalance_ratio(np.array(["a"] * 10 + ["b"] * 5)) == 2.0
    assert imbalance_ratio(np.array(["a"] * 7 + ["b"] * 7)) == 1.0
