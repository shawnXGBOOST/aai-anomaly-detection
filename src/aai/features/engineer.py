"""Feature engineering for one channel.

Every feature at time ``t`` uses only the points at ``t`` and earlier, so features can be
computed the same way at training time and at inference time. The first ``warmup`` rows of a
series do not have a full history and are NaN; callers drop them or provide history.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import yaml
from numpy.lib.stride_tricks import sliding_window_view

MAD_TO_STD = 1.4826  # makes the median absolute deviation comparable to a standard deviation


@dataclass(frozen=True)
class FeatureConfig:
    windows: tuple[int, ...] = (10, 50)  # rolling window lengths, in time steps
    min_scale: float = 0.05  # floor for the telemetry scale (in raw telemetry units)
    mad_floor: float = 0.1  # floor for the robust spread (in scaled units)

    def __post_init__(self) -> None:
        object.__setattr__(self, "windows", tuple(int(w) for w in self.windows))
        if not self.windows or min(self.windows) < 3:
            raise ValueError("windows must be a non-empty list of integers >= 3")
        if self.min_scale <= 0 or self.mad_floor <= 0:
            raise ValueError("min_scale and mad_floor must be positive")

    @property
    def warmup(self) -> int:
        """Rows at the start of a series without a full history."""
        return max(self.windows)

    def to_dict(self) -> dict[str, object]:
        return {
            "windows": list(self.windows),
            "min_scale": self.min_scale,
            "mad_floor": self.mad_floor,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> FeatureConfig:
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown feature config keys: {sorted(unknown)}")
        return cls(**data)  # type: ignore[arg-type]

    @classmethod
    def from_yaml(cls, path: str | Path) -> FeatureConfig:
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle) or {})


def _rolling_mean_std(x: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std over the window ending at ``t`` (current point included)."""
    mean = np.full(len(x), np.nan)
    std = np.full(len(x), np.nan)
    if len(x) >= w:
        windows = sliding_window_view(x, w)  # row i covers x[i : i + w], ends at t = i + w - 1
        mean[w - 1 :] = windows.mean(axis=1)
        std[w - 1 :] = windows.std(axis=1)
    return mean, std


def _robust_z(x: np.ndarray, w: int, mad_floor: float) -> np.ndarray:
    """Deviation of ``x[t]`` from the median of the *previous* ``w`` points, in MAD units.

    The current point is excluded from its own baseline, so a spike cannot hide itself.
    """
    z = np.full(len(x), np.nan)
    if len(x) > w:
        previous = sliding_window_view(x, w)[
            :-1
        ]  # row i covers x[i : i + w], baseline for t = i + w
        median = np.median(previous, axis=1)
        mad = np.median(np.abs(previous - median[:, None]), axis=1)
        z[w:] = (x[w:] - median) / np.maximum(MAD_TO_STD * mad, mad_floor)
    return z


def build_features(
    scaled: np.ndarray, flags: np.ndarray, cfg: FeatureConfig
) -> tuple[np.ndarray, list[str]]:
    """Feature matrix (n, d) and column names for one scaled telemetry series.

    ``scaled`` is the standardized telemetry (n,), ``flags`` the command-flag columns (n, k).
    """
    scaled = np.asarray(scaled, dtype=float)
    flags = np.asarray(flags, dtype=float)
    if scaled.ndim != 1 or flags.ndim != 2 or len(scaled) != len(flags):
        raise ValueError(f"expected (n,) and (n, k), got {scaled.shape} and {flags.shape}")
    n = len(scaled)

    diff = np.full(n, np.nan)
    diff[1:] = np.diff(scaled)
    columns, names = [scaled, diff], ["value", "diff1"]
    for w in cfg.windows:
        mean, std = _rolling_mean_std(scaled, w)
        columns += [mean, std, _robust_z(scaled, w, cfg.mad_floor)]
        names += [f"mean_{w}", f"std_{w}", f"robust_z_{w}"]

    changed = np.full(n, np.nan)
    changed[1:] = np.abs(np.diff(flags, axis=0)).sum(axis=1)
    columns += [flags.sum(axis=1), changed]
    names += ["flags_active", "flags_changed"]
    return np.column_stack(columns), names
