"""Hot-deck imputation for incomplete clinical records.

ADNI-style clinical tables are missing values non-uniformly: MMSE and CDR are
almost always present, while CSF biomarkers and APOE genotype are often blank.
Mean imputation flattens exactly the between-group variance the classifier
needs, so we use *hot deck* imputation instead: each incomplete record borrows
its missing values from a real, complete record ("donor") that looks like it.

Donor matching is nearest-neighbour on the standardised observed features. The
donor is drawn at random from the k nearest complete records rather than always
taking the single closest one, which stops one donor from being copied into
dozens of recipients and collapsing the imputed variance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class HotDeckImputer:
    """Sklearn-style hot-deck imputer for numeric columns.

    Parameters
    ----------
    n_donors:
        Size of the candidate donor pool; the actual donor is sampled uniformly
        from these k nearest complete records.
    stratify_by_class:
        When True, donors are restricted to records with the same label. This is
        the right default for a diagnostic table -- an AD patient should not
        inherit a healthy control's hippocampal volume -- but it means the label
        must be available at transform time, so it is only usable on training
        data. Inference falls back to the global pool automatically.
    scale:
        Standardise features before computing distances so that a column
        measured in thousands does not dominate the match.
    """

    def __init__(
        self,
        n_donors: int = 5,
        stratify_by_class: bool = True,
        scale: bool = True,
        random_state: int | None = None,
    ):
        if n_donors < 1:
            raise ValueError("n_donors must be >= 1")
        self.n_donors = n_donors
        self.stratify_by_class = stratify_by_class
        self.scale = scale
        self.rng = np.random.default_rng(random_state)

        self.columns_: list[str] = []
        self.scaler_: StandardScaler | None = None
        self.donors_: dict[object, np.ndarray] = {}
        self.column_medians_: np.ndarray | None = None

    # ------------------------------------------------------------------ fit

    def fit(self, frame: pd.DataFrame, labels: pd.Series | None = None) -> "HotDeckImputer":
        """Record the pool of complete donor rows, optionally per class."""
        self.columns_ = list(frame.columns)
        values = frame.to_numpy(dtype=float)

        complete_mask = ~np.isnan(values).any(axis=1)
        if not complete_mask.any():
            raise ValueError(
                "hot-deck imputation needs at least one fully observed record; "
                "drop the sparsest columns or fall back to median imputation"
            )

        complete = values[complete_mask]

        if self.scale:
            self.scaler_ = StandardScaler().fit(complete)

        # Median fallback for the rare case where a class has no complete donor.
        self.column_medians_ = np.nanmedian(values, axis=0)

        self.donors_ = {"__global__": complete}
        if self.stratify_by_class and labels is not None:
            complete_labels = np.asarray(labels)[complete_mask]
            for label in np.unique(complete_labels):
                self.donors_[label] = complete[complete_labels == label]

        return self

    # ------------------------------------------------------------- transform

    def transform(self, frame: pd.DataFrame, labels: pd.Series | None = None) -> pd.DataFrame:
        """Return a copy of ``frame`` with every NaN replaced by a donor value."""
        if not self.donors_:
            raise RuntimeError("call fit() before transform()")
        if list(frame.columns) != self.columns_:
            raise ValueError(
                f"column mismatch: fitted on {self.columns_}, got {list(frame.columns)}"
            )

        values = frame.to_numpy(dtype=float).copy()
        label_array = np.asarray(labels) if labels is not None else None

        for row_idx in np.where(np.isnan(values).any(axis=1))[0]:
            row = values[row_idx]
            missing = np.isnan(row)
            label = label_array[row_idx] if label_array is not None else None
            pool = self._donor_pool(label)

            if pool is None or len(pool) == 0:
                row[missing] = self.column_medians_[missing]
                continue

            donor = pool[self._pick_donor(row, ~missing, pool)]
            row[missing] = donor[missing]

        return pd.DataFrame(values, columns=self.columns_, index=frame.index)

    def fit_transform(
        self, frame: pd.DataFrame, labels: pd.Series | None = None
    ) -> pd.DataFrame:
        return self.fit(frame, labels).transform(frame, labels)

    # ---------------------------------------------------------------- helpers

    def _donor_pool(self, label: object | None) -> np.ndarray | None:
        if label is not None and label in self.donors_:
            pool = self.donors_[label]
            # A class with too few complete records gives degenerate matches;
            # widen to the global pool rather than copying the same donor.
            if len(pool) >= self.n_donors:
                return pool
        return self.donors_.get("__global__")

    def _pick_donor(self, row: np.ndarray, observed: np.ndarray, pool: np.ndarray) -> int:
        """Index of a donor, chosen among the k nearest on the observed columns."""
        if not observed.any():
            # Nothing to match on -- any donor is as good as another.
            return int(self.rng.integers(len(pool)))

        if self.scaler_ is not None:
            row_s = (row - self.scaler_.mean_) / self.scaler_.scale_
            pool_s = (pool - self.scaler_.mean_) / self.scaler_.scale_
        else:
            row_s, pool_s = row, pool

        distances = np.linalg.norm(pool_s[:, observed] - row_s[observed], axis=1)
        k = min(self.n_donors, len(pool))
        nearest = np.argpartition(distances, k - 1)[:k]
        return int(self.rng.choice(nearest))


def missingness_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing counts and rates, sorted worst-first."""
    missing = frame.isna().sum()
    report = pd.DataFrame(
        {"missing": missing, "rate": (missing / max(len(frame), 1)).round(4)}
    )
    return report.sort_values("missing", ascending=False)
