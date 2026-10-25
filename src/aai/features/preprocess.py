"""Cleaning and scaling.

Everything here either uses only values that came earlier in time (forward fill) or is fit on
the training series alone (scaler). Nothing looks at the test series.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def clean_nonfinite(
    arr: np.ndarray, fill: np.ndarray, previous: np.ndarray | None = None
) -> tuple[np.ndarray, int]:
    """Replace NaN and +-inf by the last finite value above them (causal forward fill).

    A gap at the very start uses ``previous`` (the row just before this array, if known) and
    otherwise ``fill`` (one value per column, computed from train). Returns the cleaned copy and
    the number of replaced values.
    """
    out = np.array(arr, dtype=float, copy=True)
    bad = ~np.isfinite(out)
    n_bad = int(bad.sum())
    if n_bad == 0:
        return out, 0

    out[bad] = np.nan
    seed = np.asarray(fill if previous is None else previous, dtype=float)
    for col in np.flatnonzero(bad.any(axis=0)):
        series = out[:, col]
        if np.isnan(series[0]):
            series[0] = seed[col]
        idx = np.where(np.isnan(series), 0, np.arange(len(series)))
        np.maximum.accumulate(idx, out=idx)
        out[:, col] = series[idx]

    if not np.isfinite(out).all():
        raise ValueError("cannot clean: a column has no finite value to fill from")
    return out, n_bad


@dataclass(frozen=True)
class TelemetryScaler:
    """Standardizes the telemetry column with statistics from the training series.

    ``scale`` has a floor so that channels whose training telemetry is constant do not divide by
    zero. On such channels any later deviation becomes a large scaled value, which is intended.
    """

    center: float
    scale: float

    @classmethod
    def fit(cls, train_telemetry: np.ndarray, min_scale: float) -> TelemetryScaler:
        x = np.asarray(train_telemetry, dtype=float)
        if x.size == 0 or not np.isfinite(x).all():
            raise ValueError("scaler needs a non-empty, finite training series")
        return cls(center=float(x.mean()), scale=float(max(x.std(), min_scale)))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=float) - self.center) / self.scale

    def to_dict(self) -> dict[str, float]:
        return {"center": self.center, "scale": self.scale}

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> TelemetryScaler:
        return cls(center=float(data["center"]), scale=float(data["scale"]))
