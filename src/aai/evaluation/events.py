"""Turning boolean masks into events (runs of consecutive True values)."""

from __future__ import annotations

import numpy as np

from aai.data.labels import Interval


def mask_to_segments(mask: np.ndarray) -> list[Interval]:
    """Runs of consecutive True values as inclusive ``(start, end)`` pairs."""
    flags = np.asarray(mask, dtype=bool)
    if not flags.any():
        return []
    padded = np.concatenate(([False], flags, [False]))
    change = np.flatnonzero(padded[1:] != padded[:-1])
    starts, ends = change[0::2], change[1::2] - 1
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def overlaps(a: Interval, b: Interval) -> bool:
    """True if two inclusive intervals share at least one point.

    Intervals that merely touch (one ends at 9, the other starts at 10) do not overlap.
    """
    return a[0] <= b[1] and b[0] <= a[1]
