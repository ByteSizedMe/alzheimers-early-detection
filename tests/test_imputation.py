"""Hot-deck imputation tests.

The property that matters clinically: imputed values must be *real observed
values from similar patients*, not invented averages. These tests pin that down.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alzheimers.data.imputation import HotDeckImputer, missingness_report


@pytest.fixture
def cohort() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = 120
    labels = pd.Series(["NC"] * 60 + ["AD"] * 60)
    frame = pd.DataFrame(
        {
            "mmse": np.concatenate([rng.normal(29, 1, 60), rng.normal(21, 3, 60)]),
            "hippo_vol": np.concatenate([rng.normal(7300, 400, 60), rng.normal(5300, 500, 60)]),
            "age": rng.normal(74, 6, n),
        }
    )
    return frame, labels


def test_fills_every_missing_value(cohort):
    frame, labels = cohort
    holed = frame.copy()
    holed.loc[[5, 12, 40, 77, 90], "hippo_vol"] = np.nan
    holed.loc[[8, 33], "mmse"] = np.nan

    filled = HotDeckImputer(random_state=0).fit_transform(holed, labels)

    assert filled.isna().sum().sum() == 0
    assert filled.shape == holed.shape
    assert list(filled.columns) == list(holed.columns)


def test_imputed_values_are_real_donor_values(cohort):
    """Every filled value must exist verbatim in the donor pool -- that is what
    separates hot-deck from mean/median imputation."""
    frame, labels = cohort
    holed = frame.copy()
    missing_rows = [3, 17, 55, 88, 101]
    holed.loc[missing_rows, "hippo_vol"] = np.nan

    filled = HotDeckImputer(random_state=1).fit_transform(holed, labels)

    observed = set(frame["hippo_vol"].round(9))
    for row in missing_rows:
        assert round(filled.loc[row, "hippo_vol"], 9) in observed


def test_observed_values_are_untouched(cohort):
    frame, labels = cohort
    holed = frame.copy()
    holed.loc[[10, 20], "mmse"] = np.nan

    filled = HotDeckImputer(random_state=0).fit_transform(holed, labels)

    intact = holed.drop(index=[10, 20])
    pd.testing.assert_frame_equal(filled.drop(index=[10, 20]), intact, check_dtype=False)


def test_class_stratified_donors_respect_the_label(cohort):
    """An AD record must not inherit a healthy control's hippocampal volume."""
    frame, labels = cohort
    holed = frame.copy()
    ad_rows = [65, 70, 75, 80, 85]
    holed.loc[ad_rows, "hippo_vol"] = np.nan

    filled = HotDeckImputer(stratify_by_class=True, random_state=2).fit_transform(holed, labels)

    ad_observed = set(frame.loc[labels == "AD", "hippo_vol"].round(9))
    for row in ad_rows:
        assert round(filled.loc[row, "hippo_vol"], 9) in ad_observed


def test_preserves_variance_better_than_mean_imputation(cohort):
    """Mean imputation shrinks the spread; hot-deck should not, by much."""
    frame, labels = cohort
    rng = np.random.default_rng(3)
    holed = frame.copy()
    holed.loc[rng.choice(len(frame), 40, replace=False), "hippo_vol"] = np.nan

    hot_deck = HotDeckImputer(random_state=4).fit_transform(holed, labels)
    mean_filled = holed.fillna(holed.mean())

    truth = frame["hippo_vol"].std()
    assert abs(hot_deck["hippo_vol"].std() - truth) < abs(mean_filled["hippo_vol"].std() - truth)


def test_transform_without_labels_uses_global_pool(cohort):
    """Inference has no label, so transform must still work."""
    frame, labels = cohort
    holed = frame.copy()
    holed.loc[[2, 4], "age"] = np.nan

    imputer = HotDeckImputer(random_state=0).fit(holed, labels)
    filled = imputer.transform(holed)  # no labels

    assert filled.isna().sum().sum() == 0


def test_rejects_column_mismatch(cohort):
    frame, labels = cohort
    imputer = HotDeckImputer().fit(frame, labels)
    with pytest.raises(ValueError, match="column mismatch"):
        imputer.transform(frame.rename(columns={"age": "years"}))


def test_requires_at_least_one_complete_record():
    frame = pd.DataFrame({"a": [1.0, np.nan], "b": [np.nan, 2.0]})
    with pytest.raises(ValueError, match="at least one fully observed"):
        HotDeckImputer().fit(frame)


def test_transform_before_fit_raises(cohort):
    frame, _ = cohort
    with pytest.raises(RuntimeError, match="call fit"):
        HotDeckImputer().transform(frame)


def test_missingness_report_counts_and_orders(cohort):
    frame, _ = cohort
    holed = frame.copy()
    holed.loc[0:9, "mmse"] = np.nan
    holed.loc[0:2, "age"] = np.nan

    report = missingness_report(holed)

    assert report.loc["mmse", "missing"] == 10
    assert report.loc["age", "missing"] == 3
    assert report.index[0] == "mmse"  # worst first
    assert report.loc["mmse", "rate"] == pytest.approx(10 / len(frame), abs=1e-4)
