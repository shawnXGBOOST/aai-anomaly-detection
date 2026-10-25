"""From raw channel arrays to saved feature matrices, without leaking test information."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aai.data.loader import Dataset
from aai.features.engineer import FeatureConfig, build_features
from aai.features.preprocess import TelemetryScaler, clean_nonfinite

log = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class ChannelFeatures:
    name: str
    feature_names: list[str]
    train_X: np.ndarray  # row i is train time step ``warmup + i``
    test_X: np.ndarray  # row i is test time step ``i`` (no rows are dropped)
    warmup: int
    scaler: TelemetryScaler
    n_replaced_train: int  # non-finite values that had to be filled
    n_replaced_test: int


def make_features(
    name: str, train: np.ndarray, test: np.ndarray, cfg: FeatureConfig
) -> ChannelFeatures:
    """Fit the scaler on train, then build train and test features.

    * The scaler and the fill values come from the train series only.
    * Train features drop the first ``warmup`` rows, which have no full history.
    * Test features are computed on ``[end of train, test]`` and then cut back to the test rows,
      so the first test point already has a full window. This is exactly what an online system
      would see: the recent past plus the new points.
    """
    warm = cfg.warmup
    if len(train) <= warm:
        raise ValueError(f"{name}: train has {len(train)} rows, needs more than warmup {warm}")

    fill = np.nanmedian(train, axis=0)
    train_clean, n_train = clean_nonfinite(train, fill)
    test_clean, n_test = clean_nonfinite(test, fill, previous=train_clean[-1])

    scaler = TelemetryScaler.fit(train_clean[:, 0], cfg.min_scale)
    train_scaled = scaler.transform(train_clean[:, 0])
    test_scaled = scaler.transform(test_clean[:, 0])

    train_all, names = build_features(train_scaled, train_clean[:, 1:], cfg)
    joined_all, _ = build_features(
        np.concatenate([train_scaled[-warm:], test_scaled]),
        np.vstack([train_clean[-warm:, 1:], test_clean[:, 1:]]),
        cfg,
    )
    train_X, test_X = train_all[warm:], joined_all[warm:]
    if not (np.isfinite(train_X).all() and np.isfinite(test_X).all()):
        raise ValueError(f"{name}: non-finite feature values after warm-up")
    return ChannelFeatures(name, names, train_X, test_X, warm, scaler, n_train, n_test)


def save_features(features: ChannelFeatures, out_dir: str | Path) -> Path:
    """Save one channel's features. Labels are deliberately not stored here."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{features.name}.npz"
    np.savez_compressed(
        path,
        train_X=features.train_X,
        test_X=features.test_X,
        feature_names=np.array(features.feature_names),
        warmup=np.array(features.warmup),
        center=np.array(features.scaler.center),
        scale=np.array(features.scaler.scale),
        n_replaced=np.array([features.n_replaced_train, features.n_replaced_test]),
    )
    return path


def load_features(path: str | Path) -> ChannelFeatures:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        return ChannelFeatures(
            name=path.stem,
            feature_names=[str(n) for n in data["feature_names"]],
            train_X=data["train_X"],
            test_X=data["test_X"],
            warmup=int(data["warmup"]),
            scaler=TelemetryScaler(float(data["center"]), float(data["scale"])),
            n_replaced_train=int(data["n_replaced"][0]),
            n_replaced_test=int(data["n_replaced"][1]),
        )


def build_all(dataset: Dataset, cfg: FeatureConfig, out_dir: str | Path) -> dict[str, object]:
    """Build and save features for every scored channel, one channel at a time."""
    out_dir = Path(out_dir)
    channels = {}
    for name, channel in dataset.channels.items():
        features = make_features(name, channel.train, channel.test, cfg)
        save_features(features, out_dir)
        channels[name] = {
            "train_rows": len(features.train_X),
            "test_rows": len(features.test_X),
            "n_replaced": features.n_replaced_train + features.n_replaced_test,
        }
    manifest = {
        "config": cfg.to_dict(),
        "feature_names": features.feature_names if channels else [],
        "channels": channels,
        "excluded": dataset.excluded,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("built features for %d channels (%d excluded)", len(channels), len(dataset.excluded))
    return manifest
