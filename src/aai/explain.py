"""Explaining alarms: which features drove a score, and what kind of event it looks like.

Two parts, both deliberately simple:

* :func:`top_features` turns a model's per-feature contributions over an alarm into a ranked list
  of shares. For the baseline and the autoencoder these are real parts of the score; for
  Isolation Forest they are a proxy (see ``IsolationForestModel.contributions``).
* :func:`characterize_event` labels an alarm segment as a ``spike`` or a ``sustained shift`` from
  the scaled telemetry alone. This is a heuristic with fixed rules, not a learned classifier, and
  it says nothing about *why* the telemetry moved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAD_TO_STD = 1.4826


def top_features(
    contributions: np.ndarray, feature_names: list[str], k: int = 3
) -> list[tuple[str, float]]:
    """Features ranked by their share of the total contribution over the given rows."""
    contributions = np.asarray(contributions, dtype=float)
    if contributions.ndim == 1:
        contributions = contributions[None, :]
    if contributions.shape[1] != len(feature_names):
        raise ValueError("contributions and feature_names disagree on the number of features")
    total = contributions.sum(axis=0)
    if total.sum() <= 0:
        return []
    share = total / total.sum()
    order = np.argsort(-share, kind="stable")[:k]
    return [(feature_names[i], float(share[i])) for i in order if share[i] > 0]


@dataclass(frozen=True)
class EventKind:
    label: str  # "spike" or "sustained shift"
    length: int
    shift_in_mad: float  # median of the event minus the median of the preceding history
    spread_ratio: float  # spread (MAD) of the event divided by that of the preceding history


def characterize_event(
    scaled_value: np.ndarray,
    start: int,
    end: int,
    history: int = 100,
    spike_max_length: int = 5,
    shift_min: float = 3.0,
    ratio_min: float = 2.0,
) -> EventKind:
    """Label the alarm segment ``[start, end]`` (inclusive) using only the telemetry.

    * ``spike``: the segment is at most ``spike_max_length`` steps long.
    * ``variance change``: longer, with the level close to before (|shift| < ``shift_min`` MADs)
      but a spread at least ``ratio_min`` times the earlier spread.
    * ``sustained shift``: longer, with the median at least ``shift_min`` MADs away from the
      median of the ``history`` steps before it.
    * ``no clear change``: everything else, e.g. a change in a command flag or a pattern that the
      value alone does not reveal.
    """
    scaled_value = np.asarray(scaled_value, dtype=float)
    if not 0 <= start <= end < len(scaled_value):
        raise ValueError("segment is outside the series")
    length = end - start + 1
    before = scaled_value[max(0, start - history) : start]
    seg = scaled_value[start : end + 1]
    if before.size == 0:
        return EventKind(
            "spike" if length <= spike_max_length else "sustained shift", length, 0.0, 1.0
        )
    center = float(np.median(before))
    floor = 0.1
    before_spread = max(MAD_TO_STD * float(np.median(np.abs(before - center))), floor)
    seg_center = float(np.median(seg))
    seg_spread = max(MAD_TO_STD * float(np.median(np.abs(seg - seg_center))), floor)
    shift, ratio = (seg_center - center) / before_spread, seg_spread / before_spread
    if length <= spike_max_length:
        label = "spike"
    elif abs(shift) < shift_min and ratio >= ratio_min:
        label = "variance change"
    elif abs(shift) >= shift_min:
        label = "sustained shift"
    else:
        label = "no clear change"  # the raw signal alone does not show what happened
    return EventKind(label, length, shift, ratio)
