"""A trained detector for one channel: raw telemetry window in, scores, alarms and reasons out."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aai import __version__
from aai.data.loader import Dataset
from aai.detection.thresholds import fixed_threshold, hold_alarms
from aai.evaluation.events import mask_to_segments
from aai.explain import characterize_event, top_features
from aai.features.engineer import FeatureConfig, build_features
from aai.features.pipeline import make_features
from aai.features.preprocess import TelemetryScaler, clean_nonfinite
from aai.models.baseline import RobustZBaseline
from aai.models.iforest import IsolationForestModel

log = logging.getLogger(__name__)

MODELS = ("robust_z", "isolation_forest", "autoencoder")
FLAT_TRAIN_STD = 0.1  # same rule as the experiment: scaled train signal std below this is flat


class ArtifactError(ValueError):
    """Raised when saved artifacts are missing or inconsistent."""


@dataclass
class Event:
    start: int  # row index in the submitted window, inclusive
    end: int
    length: int
    kind: str
    shift_in_mad: float
    top_features: list[tuple[str, float]]


@dataclass
class Detection:
    first_scored_row: int  # rows before this one lack the history needed for features
    scores: np.ndarray  # one per scored row
    alarms: np.ndarray  # boolean, one per scored row (after the alarm hold)
    threshold: float
    events: list[Event]
    n_replaced: int  # non-finite input values that were filled


class Detector:
    def __init__(self, meta: dict, model) -> None:
        self.meta = meta
        self.model = model
        self.cfg = FeatureConfig.from_dict(meta["features"])
        self.scaler = TelemetryScaler.from_dict(meta["scaler"])
        self.fill = np.asarray(meta["fill"], dtype=float)
        self.threshold = float(meta["threshold"])
        self.hold = int(meta["hold"])
        self.feature_names = list(meta["feature_names"])

    @property
    def n_features(self) -> int:
        return len(self.fill)

    @property
    def warmup(self) -> int:
        return self.cfg.warmup

    def detect(self, rows: np.ndarray, max_rows: int = 100_000) -> Detection:
        rows = np.asarray(rows, dtype=float)
        if rows.ndim != 2 or rows.shape[1] != self.n_features:
            raise ValueError(f"rows must have shape (n, {self.n_features})")
        if not self.warmup < len(rows) <= max_rows:
            raise ValueError(f"send between {self.warmup + 1} and {max_rows} rows")
        clean, n_replaced = clean_nonfinite(rows, self.fill)
        scaled = self.scaler.transform(clean[:, 0])
        features, _ = build_features(scaled, clean[:, 1:], self.cfg)
        X = features[self.warmup :]
        scores = self.model.score(X)
        alarms = hold_alarms(scores > self.threshold, self.hold)
        events = []
        for start, end in mask_to_segments(alarms):
            kind = characterize_event(scaled, start + self.warmup, end + self.warmup)
            top = top_features(self.model.contributions(X[start : end + 1]), self.feature_names)
            events.append(
                Event(
                    start + self.warmup,
                    end + self.warmup,
                    end - start + 1,
                    kind.label,
                    kind.shift_in_mad,
                    top,
                )
            )
        return Detection(self.warmup, scores, alarms, self.threshold, events, n_replaced)


def _make_model(name: str, feature_names: list[str], seed: int, drop_constant: bool):
    if name == "robust_z":
        return RobustZBaseline(feature_names)
    if name == "isolation_forest":
        return IsolationForestModel(seed=seed, drop_constant=drop_constant)
    if name == "autoencoder":
        from aai.models.autoencoder import AutoencoderModel

        return AutoencoderModel(seed=seed, drop_constant=drop_constant)
    raise ValueError(f"unknown model: {name}")


def train_detector(
    name: str,
    channel,
    feature_cfg: FeatureConfig,
    *,
    fit_fraction: float = 0.7,
    quantile: float = 0.995,
    hold: int = 20,
    min_threshold: float = 0.0,
    seed: int = 0,
    drop_constant: bool = True,
) -> Detector:
    """Fit on the first part of train, set the threshold on the rest. No labels are used."""
    feats = make_features(channel.name, channel.train, channel.test, feature_cfg)
    X = feats.train_X
    split = int(len(X) * fit_fraction)
    model = _make_model(name, feats.feature_names, seed, drop_constant).fit(X[:split])
    threshold = max(fixed_threshold(model.score(X[split:]), quantile), min_threshold)
    meta = {
        "aai_version": __version__,
        "channel": channel.name,
        "spacecraft": channel.spacecraft,
        "model": name,
        "features": feature_cfg.to_dict(),
        "feature_names": feats.feature_names,
        "scaler": feats.scaler.to_dict(),
        "fill": np.nanmedian(channel.train, axis=0).tolist(),
        "threshold": threshold,
        "quantile": quantile,
        "hold": hold,
        "min_threshold": min_threshold,
        "drop_constant": drop_constant,
        "seed": seed,
        "n_train_rows": int(len(X)),
        "flat_train": bool(X[:, 0].std() < FLAT_TRAIN_STD),
    }
    return Detector(meta, model)


def save_detector(detector: Detector, root: str | Path) -> Path:
    directory = Path(root) / detector.meta["model"] / detector.meta["channel"]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "meta.json").write_text(json.dumps(detector.meta, indent=2))
    name = detector.meta["model"]
    if name == "isolation_forest":
        import joblib

        joblib.dump(detector.model, directory / "model.joblib")
    elif name == "autoencoder":
        detector.model.save(directory)
    return directory


def load_detector(root: str | Path, model: str, channel: str) -> Detector:
    directory = Path(root) / model / channel
    meta_path = directory / "meta.json"
    if not meta_path.is_file():
        raise ArtifactError(f"no artifacts for model={model!r} channel={channel!r}")
    meta = json.loads(meta_path.read_text())
    if model == "robust_z":
        fitted = RobustZBaseline(meta["feature_names"])
    elif model == "isolation_forest":
        import joblib

        fitted = joblib.load(directory / "model.joblib")  # our own artifact; do not load others
    elif model == "autoencoder":
        from aai.models.autoencoder import AutoencoderModel

        fitted = AutoencoderModel.load(directory, len(meta["feature_names"]))
    else:
        raise ArtifactError(f"unknown model: {model!r}")
    return Detector(meta, fitted)


def train_all(
    dataset: Dataset,
    feature_cfg: FeatureConfig,
    root: str | Path,
    models: tuple[str, ...] = MODELS,
    **kwargs,
) -> int:
    count = 0
    for channel in dataset.channels.values():
        for name in models:
            floor = 3.5 if name == "robust_z" else 0.0
            det = train_detector(name, channel, feature_cfg, min_threshold=floor, **kwargs)
            save_detector(det, root)
            count += 1
        log.info("trained %s", channel.name)
    return count
