"""Isolation Forest on the engineered features."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from aai.models.common import keep_columns


class IsolationForestModel:
    """Anomaly score = ``-score_samples`` of scikit-learn's Isolation Forest.

    scikit-learn returns the negated score from the original paper, so negating it again gives
    a number where higher means more anomalous. The model scores each row independently, so it
    cannot use information from neighbouring time steps.
    """

    name = "isolation_forest"

    def __init__(self, n_estimators: int = 200, seed: int = 0, drop_constant: bool = False) -> None:
        self.n_estimators = n_estimators
        self.seed = seed
        self.drop_constant = drop_constant
        self._keep: np.ndarray | None = None
        self._center: np.ndarray | None = None
        self._spread: np.ndarray | None = None
        self._model: IsolationForest | None = None

    def fit(self, X: np.ndarray) -> IsolationForestModel:
        X = np.asarray(X, dtype=float)
        self._keep = keep_columns(X, self.drop_constant)
        self._center = np.median(X, axis=0)
        self._spread = np.maximum(1.4826 * np.median(np.abs(X - self._center), axis=0), 1e-3)
        self._model = IsolationForest(
            n_estimators=self.n_estimators, random_state=self.seed, n_jobs=1
        ).fit(X[:, self._keep])
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("model is not fitted")
        return -self._model.score_samples(np.asarray(X, dtype=float)[:, self._keep])

    def contributions(self, X: np.ndarray) -> np.ndarray:
        """Proxy attribution: each feature's robust distance from its fit-data median.

        Isolation Forest has no per-feature explanation of its own, so this is not what the
        forest used; it only says which features look unusual compared with the fit data.
        """
        if self._center is None or self._spread is None:
            raise RuntimeError("model is not fitted")
        return np.abs((np.asarray(X, dtype=float) - self._center) / self._spread)
