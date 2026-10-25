"""Inference cost of a scoring function: latency and peak Python/NumPy memory."""

from __future__ import annotations

import statistics
import time
import tracemalloc
from collections.abc import Callable

import numpy as np


def measure_scoring(score: Callable[[np.ndarray], np.ndarray], X: np.ndarray, repeats: int = 5):
    """Median latency per 1,000 rows and peak traced memory of one scoring call.

    Memory comes from ``tracemalloc``, so it counts Python and NumPy allocations only. Memory
    that PyTorch allocates natively is not included.
    """
    score(X)  # warm-up, so one-off setup cost does not count as latency
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        score(X)
        durations.append(time.perf_counter() - start)
    tracemalloc.start()
    try:
        score(X)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {
        "ms_per_1k_rows": 1000 * statistics.median(durations) / len(X) * 1000,
        "peak_mb": peak / 1e6,
    }
