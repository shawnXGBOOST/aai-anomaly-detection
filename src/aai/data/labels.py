"""Parsing of the telemanom label file and conversion of intervals to boolean masks.

The label file has one row per channel. Each row lists the anomalous intervals of that channel's
*test* series as ``[[start, end], ...]``. Both ends are **inclusive** indices. The benchmark data
supports this: several intervals end exactly at the last index of their series and none go past it.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# An anomaly interval (start, end), both inclusive, indexing into the test series.
Interval = tuple[int, int]

REQUIRED_COLUMNS = ("chan_id", "spacecraft", "anomaly_sequences", "class", "num_values")


class LabelError(ValueError):
    """Raised when the label file or an interval is malformed."""


@dataclass(frozen=True)
class ChannelLabels:
    chan_id: str
    spacecraft: str
    intervals: tuple[Interval, ...]  # in file order
    classes: tuple[str, ...]  # "point" / "contextual", aligned with intervals
    num_values: int  # length of the test series according to the label file


@dataclass(frozen=True)
class LabelTable:
    usable: dict[str, ChannelLabels]  # channels with one unambiguous label row
    ambiguous: dict[str, str]  # chan_id -> why the labels cannot be trusted
    n_rows: int  # rows in the CSV, duplicates included


def parse_intervals(text: str) -> tuple[Interval, ...]:
    """Parse ``"[[5, 9], [20, 30]]"`` into ``((5, 9), (20, 30))`` and validate each pair."""
    try:
        intervals = tuple((int(a), int(b)) for a, b in ast.literal_eval(text))
    except (ValueError, SyntaxError, TypeError) as exc:
        raise LabelError(f"cannot parse anomaly_sequences: {text!r}") from exc
    for start, end in intervals:
        if start < 0 or end < start:
            raise LabelError(f"invalid interval ({start}, {end}) in {text!r}")
    return intervals


def parse_classes(text: str) -> tuple[str, ...]:
    """Parse ``"[point, contextual]"`` (not valid Python, the words are unquoted)."""
    inner = text.strip().strip("[]")
    return tuple(part.strip() for part in inner.split(",") if part.strip())


def load_labels(path: str | Path) -> LabelTable:
    """Read the label CSV.

    A channel that appears in several rows is kept only if all rows agree. Otherwise it is
    reported as ambiguous, because there is no principled way to pick the right row.
    """
    df = pd.read_csv(path, dtype={"chan_id": str, "spacecraft": str})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise LabelError(f"label file is missing columns: {missing}")

    usable: dict[str, ChannelLabels] = {}
    ambiguous: dict[str, str] = {}
    for chan_id, group in df.groupby("chan_id", sort=False):
        entries = [
            ChannelLabels(
                chan_id=str(chan_id),
                spacecraft=str(row["spacecraft"]),
                intervals=parse_intervals(row["anomaly_sequences"]),
                classes=parse_classes(row["class"]),
                num_values=int(row["num_values"]),
            )
            for row in group.to_dict("records")
        ]
        if all(entry == entries[0] for entry in entries[1:]):
            usable[str(chan_id)] = entries[0]
        else:
            details = "; ".join(str(list(entry.intervals)) for entry in entries)
            ambiguous[str(chan_id)] = f"{len(entries)} label rows disagree: {details}"
    return LabelTable(usable=usable, ambiguous=ambiguous, n_rows=len(df))


def intervals_to_mask(intervals: tuple[Interval, ...], length: int) -> np.ndarray:
    """Boolean mask of ``length`` points, True inside any interval (ends inclusive)."""
    mask = np.zeros(length, dtype=bool)
    for start, end in intervals:
        if end >= length:
            raise LabelError(f"interval ({start}, {end}) exceeds series length {length}")
        mask[start : end + 1] = True
    return mask
