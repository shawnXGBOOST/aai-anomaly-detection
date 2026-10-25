"""FastAPI service: loads saved detectors and scores raw telemetry windows.

Run with ``uvicorn --factory aai.serving.api:create_app``. Artifacts are read from the directory
in ``AAI_ARTIFACTS`` (default ``artifacts/models``) and never trained at request time.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from aai import __version__
from aai.serving.detector import ArtifactError, Detector, load_detector

log = logging.getLogger("aai.api")

MAX_ROWS = 100_000
NAME_PATTERN = r"^[A-Za-z0-9_-]{1,32}$"  # channel names become directory names, so be strict


class DetectRequest(BaseModel):
    channel: str = Field(pattern=NAME_PATTERN, examples=["A-1"])
    model: Literal["robust_z", "isolation_forest", "autoencoder"] = "robust_z"
    rows: list[list[float | None]] = Field(
        description="Telemetry rows in time order. Column 0 is the telemetry value, the rest are "
        "0/1 command flags. null is treated as missing.",
        max_length=MAX_ROWS,
    )
    include_scores: bool = True


class EventOut(BaseModel):
    start: int
    end: int
    length: int
    kind: str
    shift_in_mad: float
    top_features: list[tuple[str, float]]


class DetectResponse(BaseModel):
    channel: str
    model: str
    n_rows: int
    first_scored_row: int
    threshold: float
    n_replaced: int
    warnings: list[str]
    events: list[EventOut]
    scores: list[float] | None = None
    alarms: list[bool] | None = None


class _Metrics:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests = 0
        self.errors = 0
        self.events = 0
        self.latency_sum = 0.0

    def render(self) -> str:
        with self.lock:
            return (
                "# TYPE aai_requests_total counter\n"
                f"aai_requests_total {self.requests}\n"
                "# TYPE aai_errors_total counter\n"
                f"aai_errors_total {self.errors}\n"
                "# TYPE aai_events_total counter\n"
                f"aai_events_total {self.events}\n"
                "# TYPE aai_request_seconds_sum counter\n"
                f"aai_request_seconds_sum {self.latency_sum:.6f}\n"
            )


def create_app(artifacts_dir: str | Path | None = None) -> FastAPI:
    root = Path(artifacts_dir or os.environ.get("AAI_ARTIFACTS", "artifacts/models"))
    app = FastAPI(title="AAI anomaly detection", version=__version__)
    cache: dict[tuple[str, str], Detector] = {}
    cache_lock = threading.Lock()
    metrics = _Metrics()

    def get_detector(model: str, channel: str) -> Detector:
        key = (model, channel)
        with cache_lock:
            if key not in cache:
                cache[key] = load_detector(root, model, channel)
            return cache[key]

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "version": __version__, "artifacts_found": root.is_dir()}

    @app.get("/channels")
    def channels() -> dict[str, list[str]]:
        found: dict[str, list[str]] = {}
        if root.is_dir():
            for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
                found[model_dir.name] = sorted(
                    p.name for p in model_dir.iterdir() if (p / "meta.json").is_file()
                )
        return found

    @app.get("/metrics")
    def prometheus() -> Response:
        return Response(metrics.render(), media_type="text/plain; version=0.0.4")

    @app.post("/detect", response_model=DetectResponse)
    def detect(req: DetectRequest) -> DetectResponse:
        start = time.perf_counter()
        with metrics.lock:
            metrics.requests += 1
        try:
            detector = get_detector(req.model, req.channel)
        except ArtifactError as exc:
            _count_error(metrics)
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        rows = np.array(
            [[math.nan if v is None else v for v in row] for row in req.rows], dtype=float
        )
        try:
            result = detector.detect(rows, max_rows=MAX_ROWS)
        except ValueError as exc:
            _count_error(metrics)
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        warnings = []
        if detector.meta.get("flat_train"):
            warnings.append(
                "the training signal for this channel was flat, so its threshold is unreliable"
            )
        if result.n_replaced:
            warnings.append(f"{result.n_replaced} missing or non-finite values were filled")
        elapsed = time.perf_counter() - start
        with metrics.lock:
            metrics.events += len(result.events)
            metrics.latency_sum += elapsed
        log.info(
            json.dumps(
                {
                    "event": "detect",
                    "channel": req.channel,
                    "model": req.model,
                    "rows": len(rows),
                    "alarm_events": len(result.events),
                    "latency_ms": round(elapsed * 1000, 2),
                }
            )
        )
        return DetectResponse(
            channel=req.channel,
            model=req.model,
            n_rows=len(rows),
            first_scored_row=result.first_scored_row,
            threshold=result.threshold,
            n_replaced=result.n_replaced,
            warnings=warnings,
            events=[EventOut(**e.__dict__) for e in result.events],
            scores=result.scores.tolist() if req.include_scores else None,
            alarms=result.alarms.tolist() if req.include_scores else None,
        )

    return app


def _count_error(metrics: _Metrics) -> None:
    with metrics.lock:
        metrics.errors += 1
