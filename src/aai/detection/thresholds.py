"""Turning anomaly scores into alarms: a fixed threshold, an adaptive one, and a causal hold.

Everything here is causal: the decision for time step ``t`` depends only on scores up to ``t``.

Two adaptive rules exist, and only one of them is used for the results:

``robust`` (used)
    Threshold = median + ``k`` * robust spread of a rolling buffer of recent scores, computed on
    ``log(score)``. ``k`` is calibrated once on the calibration scores so that the starting
    threshold matches the chosen quantile. The cutoff scales with the spread of the buffer, so it
    has a stable level when the data is stationary.

``quantile`` (kept as a documented failed experiment)
    Threshold = a high quantile of the buffer. With a freeze guard this erodes: flagged scores
    stay out of the buffer, so the buffer holds only the lower part of the distribution, its
    quantile is lower, more scores are flagged, and so on. See the failed experiments in the README.
"""

from __future__ import annotations

import math
from bisect import bisect_left, insort
from collections import deque
from dataclasses import dataclass

import numpy as np

MAD_TO_STD = 1.4826


def _check_quantile(q: float) -> None:
    if not 0.0 < q < 1.0:
        raise ValueError("quantile must be strictly between 0 and 1")


def fixed_threshold(calibration_scores: np.ndarray, quantile: float) -> float:
    """Threshold = a high quantile of scores observed on normal (calibration) data."""
    _check_quantile(quantile)
    scores = np.asarray(calibration_scores, dtype=float)
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("calibration scores must be non-empty and finite")
    return float(np.quantile(scores, quantile))


def hold_alarms(raw: np.ndarray, hold: int) -> np.ndarray:
    """Keep an alarm on for ``hold`` steps after the last threshold crossing.

    Unlike merging nearby alarm runs after the fact, this only looks backwards, so the alarm at
    step ``t`` never depends on a later score. It removes flicker around the threshold.
    """
    if hold < 0:
        raise ValueError("hold must be >= 0")
    raw = np.asarray(raw, dtype=bool)
    if hold == 0:
        return raw.copy()
    spread = np.convolve(raw.astype(np.int64), np.ones(hold + 1, dtype=np.int64))[: len(raw)]
    return spread > 0


@dataclass(frozen=True)
class AdaptiveConfig:
    quantile: float = 0.995  # false-alarm budget used to calibrate k (robust) or the cutoff
    rule: str = "robust"  # "robust" (median + k * spread) or "quantile" (failed experiment)
    window: int = 1000  # number of recent accepted scores the threshold is computed from
    guard: str = "freeze"  # "none": always accept scores; "freeze": do not accept flagged ones
    max_freeze: int | None = 200  # consecutive flagged steps after which scores are accepted again
    min_k: float = 3.0  # lower bound for the calibrated multiplier k
    min_spread: float = 0.1  # floor for the robust spread, in log-score units

    def __post_init__(self) -> None:
        _check_quantile(self.quantile)
        if self.rule not in ("robust", "quantile"):
            raise ValueError("rule must be 'robust' or 'quantile'")
        if self.window < 3:
            raise ValueError("window must be >= 3")
        if self.guard not in ("none", "freeze"):
            raise ValueError("guard must be 'none' or 'freeze'")
        if self.max_freeze is not None and self.max_freeze < 1:
            raise ValueError("max_freeze must be >= 1 or None")
        if self.min_k <= 0 or self.min_spread <= 0:
            raise ValueError("min_k and min_spread must be positive")


def _sorted_quantile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation quantile of an already sorted list (same rule as ``np.quantile``)."""
    pos = q * (len(sorted_values) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


class _QuantileBuffer:
    """Rolling buffer whose threshold is a quantile of its contents."""

    def __init__(self, values: np.ndarray, cfg: AdaptiveConfig) -> None:
        self._q, self._window = cfg.quantile, cfg.window
        self._fifo = deque(values[-cfg.window :].tolist())
        self._sorted = sorted(self._fifo)

    def threshold(self) -> float:
        return _sorted_quantile(self._sorted, self._q)

    def push(self, value: float) -> None:
        if len(self._fifo) == self._window:
            del self._sorted[bisect_left(self._sorted, self._fifo.popleft())]
        self._fifo.append(value)
        insort(self._sorted, value)


class _RobustBuffer:
    """Rolling buffer whose threshold is median + k * robust spread of its contents."""

    def __init__(self, values: np.ndarray, k: float, cfg: AdaptiveConfig) -> None:
        self._k, self._min_spread, self._window = k, cfg.min_spread, cfg.window
        self._values = values[-cfg.window :].astype(float).copy()
        self._next = 0  # index of the oldest value once the buffer is full

    def threshold(self) -> float:
        median = float(np.median(self._values))
        mad = float(np.median(np.abs(self._values - median)))
        return median + self._k * max(MAD_TO_STD * mad, self._min_spread)

    def push(self, value: float) -> None:
        if len(self._values) < self._window:
            self._values = np.append(self._values, value)
        else:
            self._values[self._next] = value
            self._next = (self._next + 1) % self._window


def calibrate_k(log_history: np.ndarray, cfg: AdaptiveConfig) -> float:
    """Multiplier k such that median + k * spread of the calibration data sits at the quantile."""
    median = float(np.median(log_history))
    mad = float(np.median(np.abs(log_history - median)))
    spread = max(MAD_TO_STD * mad, cfg.min_spread)
    level = float(np.quantile(log_history, cfg.quantile))
    return max((level - median) / spread, cfg.min_k)


def adaptive_threshold(
    history: np.ndarray, scores: np.ndarray, cfg: AdaptiveConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Alarms and per-step thresholds from a rolling buffer of recent *normal-looking* scores.

    ``history`` are scores from normal data that precede ``scores`` in time (the calibration
    part of train). The buffer starts from them. At each step the threshold is read from the
    buffer, the step is flagged if its score exceeds it, and then the score may or may not enter
    the buffer:

    * ``guard="none"``: every score enters, so a long anomaly drags the threshold up with it.
    * ``guard="freeze"``: flagged scores stay out, so anomalies cannot corrupt the baseline.
      After ``max_freeze`` consecutive flagged steps they are accepted again, because a permanent
      level shift would otherwise lock the detector in alarm forever.

    The threshold at step ``t`` is read before score ``t`` enters the buffer, so no future
    information is used. Thresholds are returned in score units.
    """
    history = np.asarray(history, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if history.size == 0:
        raise ValueError("history must not be empty")
    if not (np.isfinite(history).all() and np.isfinite(scores).all()):
        raise ValueError("scores must be finite")

    if cfg.rule == "robust":
        if (history < 0).any() or (scores < 0).any():
            raise ValueError("the robust rule works on log(score) and needs scores >= 0")
        eps = max(1e-12, 1e-3 * float(np.median(history)))
        log_history = np.log(history + eps)
        buffer = _RobustBuffer(log_history, calibrate_k(log_history, cfg), cfg)
        values = np.log(scores + eps)
    else:
        eps = 0.0
        buffer = _QuantileBuffer(history, cfg)
        values = scores

    alarms = np.zeros(len(values), dtype=bool)
    level = np.zeros(len(values))
    streak = 0
    for t, value in enumerate(values.tolist()):
        level[t] = buffer.threshold()
        flagged = value > level[t]
        alarms[t] = flagged
        streak = streak + 1 if flagged else 0
        accept = (
            True
            if not flagged or cfg.guard == "none"
            else cfg.max_freeze is not None and streak > cfg.max_freeze
        )
        if accept:
            buffer.push(value)

    if cfg.rule == "robust":
        with np.errstate(over="ignore"):  # a huge log-threshold simply means "never alarm"
            thresholds = np.exp(level) - eps
    else:
        thresholds = level
    return alarms, thresholds
