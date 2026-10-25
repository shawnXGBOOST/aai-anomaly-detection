"""Event-level and point-level evaluation of one detector on one or many channels.

The exact matching rules are written down in ``docs/evaluation_protocol.md``. In short:

* A true interval is *detected* if at least one alarm point lies inside it.
* A predicted event (a run of alarm points) is a *hit* if it overlaps any true interval.
* Point-level metrics are computed on raw points, without "point-adjust".
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from aai.data.labels import Interval
from aai.evaluation.events import mask_to_segments, overlaps


@dataclass(frozen=True)
class ChannelResult:
    channel: str
    spacecraft: str
    n_points: int
    n_anomalous_points: int
    n_true_events: int
    n_detected_events: int
    n_pred_events: int
    n_pred_events_hit: int
    point_tp: int
    point_fp: int
    point_fn: int
    point_tn: int
    delays: tuple[int, ...]  # one per detected true interval, in time steps
    relative_delays: tuple[float, ...]  # delay divided by the interval length
    average_precision: float | None  # None when no scores were given or labels are one-class


def _check_inputs(mask: np.ndarray, alarms: np.ndarray, scores: np.ndarray | None) -> None:
    if mask.dtype != bool or alarms.dtype != bool:
        raise ValueError("mask and alarms must be boolean arrays")
    if mask.ndim != 1 or mask.shape != alarms.shape:
        raise ValueError(f"mask {mask.shape} and alarms {alarms.shape} must be equal 1-D shapes")
    if scores is not None:
        if scores.shape != mask.shape:
            raise ValueError(f"scores {scores.shape} must match mask {mask.shape}")
        if not np.isfinite(scores).all():
            raise ValueError("scores contain NaN or inf")


def evaluate_channel(
    channel: str,
    spacecraft: str,
    mask: np.ndarray,
    alarms: np.ndarray,
    scores: np.ndarray | None = None,
) -> ChannelResult:
    """Score one channel.

    ``mask`` is the ground truth, ``alarms`` the detector's final on/off decision per test
    step, and ``scores`` the optional raw anomaly score used for the threshold-free PR-AUC.
    """
    mask, alarms = np.asarray(mask), np.asarray(alarms)
    scores = None if scores is None else np.asarray(scores, dtype=float)
    _check_inputs(mask, alarms, scores)

    true_events: list[Interval] = mask_to_segments(mask)
    pred_events: list[Interval] = mask_to_segments(alarms)

    delays, relative_delays, detected = [], [], 0
    for start, end in true_events:
        inside = np.flatnonzero(alarms[start : end + 1])
        if inside.size:
            detected += 1
            delays.append(int(inside[0]))
            relative_delays.append(float(inside[0]) / (end - start + 1))
    hits = sum(any(overlaps(p, t) for t in true_events) for p in pred_events)

    ap = None
    if scores is not None and mask.any() and not mask.all():
        ap = float(average_precision_score(mask, scores))

    return ChannelResult(
        channel=channel,
        spacecraft=spacecraft,
        n_points=len(mask),
        n_anomalous_points=int(mask.sum()),
        n_true_events=len(true_events),
        n_detected_events=detected,
        n_pred_events=len(pred_events),
        n_pred_events_hit=hits,
        point_tp=int((alarms & mask).sum()),
        point_fp=int((alarms & ~mask).sum()),
        point_fn=int((~alarms & mask).sum()),
        point_tn=int((~alarms & ~mask).sum()),
        delays=tuple(delays),
        relative_delays=tuple(relative_delays),
        average_precision=ap,
    )


def _safe_div(numerator: float, denominator: float) -> float:
    """Ratio that is 0.0 when nothing was predicted (so a silent detector scores 0, not NaN)."""
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2 * precision * recall, precision + recall)


def summarize(results: Sequence[ChannelResult]) -> dict[str, float]:
    """Pool the counts of several channels into one set of headline metrics."""
    if not results:
        raise ValueError("no results to summarize")

    def total(attr: str) -> int:
        return sum(getattr(r, attr) for r in results)

    true_events, detected = total("n_true_events"), total("n_detected_events")
    pred_events, hits = total("n_pred_events"), total("n_pred_events_hit")
    tp, fp, fn = total("point_tp"), total("point_fp"), total("point_fn")
    normal_points = total("n_points") - total("n_anomalous_points")

    event_precision = _safe_div(hits, pred_events)
    event_recall = _safe_div(detected, true_events)
    point_precision = _safe_div(tp, tp + fp)
    point_recall = _safe_div(tp, tp + fn)
    delays = [d for r in results for d in r.delays]
    relative = [d for r in results for d in r.relative_delays]
    aps = [r.average_precision for r in results if r.average_precision is not None]

    return {
        "channels": len(results),
        "true_events": true_events,
        "detected_events": detected,
        "pred_events": pred_events,
        "false_alarm_events": pred_events - hits,
        "event_precision": event_precision,
        "event_recall": event_recall,
        "event_f1": _f1(event_precision, event_recall),
        "point_precision": point_precision,
        "point_recall": point_recall,
        "point_f1": _f1(point_precision, point_recall),
        "normal_time_flagged": _safe_div(fp, normal_points),
        "false_alarms_per_1k_normal": 1000 * _safe_div(pred_events - hits, normal_points),
        "median_delay": float(np.median(delays)) if delays else float("nan"),
        "median_relative_delay": float(np.median(relative)) if relative else float("nan"),
        "mean_pr_auc": float(np.mean(aps)) if aps else float("nan"),
        "mean_anomaly_rate": float(np.mean([r.n_anomalous_points / r.n_points for r in results])),
    }


def summary_table(results: Sequence[ChannelResult]) -> pd.DataFrame:
    """One row for all channels pooled, then one per spacecraft."""
    by_craft: dict[str, list[ChannelResult]] = defaultdict(list)
    for result in results:
        by_craft[result.spacecraft].append(result)
    rows = {"ALL": summarize(results)}
    rows.update({craft: summarize(group) for craft, group in sorted(by_craft.items())})
    return pd.DataFrame(rows).T


def channels_frame(results: Sequence[ChannelResult]) -> pd.DataFrame:
    """Per-channel results for inspection and export."""
    frame = pd.DataFrame([asdict(r) for r in results])
    frame["delays"] = frame["delays"].map(list)
    frame["relative_delays"] = frame["relative_delays"].map(list)
    return frame
