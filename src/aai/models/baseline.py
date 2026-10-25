"""Statistical baseline: how far is the current point from its recent past, robustly?"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class RobustZBaseline:
    """Anomaly score = the largest ``|robust z-score|`` over the configured windows.

    The robust z-scores are already computed by the feature step (distance from the median of
    the previous window, in MAD units), so there is nothing to learn. It is the bar every
    learned model has to clear.
    """

    name = "robust_z"

    def __init__(self, feature_names: Sequence[str]) -> None:
        self.columns = [i for i, n in enumerate(feature_names) if n.startswith("robust_z_")]
        self.column_names = [feature_names[i] for i in self.columns]
        if not self.columns:
            raise ValueError("no robust_z_* features found")

    def fit(self, X: np.ndarray) -> RobustZBaseline:
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return np.abs(np.asarray(X, dtype=float)[:, self.columns]).max(axis=1)

    def contributions(self, X: np.ndarray) -> np.ndarray:
        """Per-feature share of the score: only the robust z-score columns are non-zero."""
        X = np.asarray(X, dtype=float)
        out = np.zeros_like(X)
        out[:, self.columns] = np.abs(X[:, self.columns])
        return out
