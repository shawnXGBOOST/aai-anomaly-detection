"""Helpers shared by the learned models."""

from __future__ import annotations

import numpy as np


def keep_columns(X: np.ndarray, drop_constant: bool) -> np.ndarray:
    """Indices of the features a model should use.

    A feature that never changes in the fit data says nothing about normal behaviour, and
    standardizing it divides by (almost) zero, so any later change becomes an enormous score.
    With ``drop_constant`` such features are ignored. If every feature is constant, all are kept.
    """
    if not drop_constant:
        return np.arange(X.shape[1])
    keep = np.flatnonzero(X.std(axis=0) > 1e-9)
    return keep if keep.size else np.arange(X.shape[1])
