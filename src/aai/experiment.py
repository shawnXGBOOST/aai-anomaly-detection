"""Run every model x threshold combination on every channel and write the results.

Protocol (fixed in ``configs/experiment.yaml`` before any test run):

* Per channel, the first ``fit_fraction`` of the train features fits the model; the remaining
  train rows are *calibration* rows (assumed normal, no labels) that set the thresholds.
* Test scores are computed once per model; thresholds and sensitivity variants are then applied
  to the stored scores, so no variant can see anything the others could not.
* Labels are used only inside :func:`aai.evaluation.metrics.evaluate_channel`.
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from aai.data.loader import Dataset
from aai.detection.thresholds import (
    AdaptiveConfig,
    adaptive_threshold,
    fixed_threshold,
    hold_alarms,
)
from aai.evaluation.metrics import ChannelResult, evaluate_channel, summary_table
from aai.evaluation.profiling import measure_scoring
from aai.features.pipeline import ChannelFeatures, load_features
from aai.models.baseline import RobustZBaseline
from aai.models.iforest import IsolationForestModel

log = logging.getLogger(__name__)

THRESHOLDS = ("fixed", "adaptive_none", "adaptive_freeze")


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 0
    fit_fraction: float = 0.7
    quantile: float = 0.995
    hold: int = 20
    rule: str = "robust"
    window: int = 1000
    max_freeze: int = 200
    models: tuple[str, ...] = ("robust_z", "isolation_forest", "autoencoder")
    thresholds: tuple[str, ...] = THRESHOLDS
    sens_quantiles: tuple[float, ...] = (0.99, 0.995, 0.999)
    sens_holds: tuple[int, ...] = (0, 20, 50)
    profiling_channels: int = 5
    drop_constant: bool = False  # models ignore features that are constant in their fit data
    min_thresholds: tuple[tuple[str, float], ...] = ()  # (model, lowest allowed threshold)
    abstain_constant_train: bool = False  # no alarms on channels whose train signal is flat

    def __post_init__(self) -> None:
        if not 0.3 <= self.fit_fraction < 1.0:
            raise ValueError("fit_fraction must be in [0.3, 1)")
        bad = set(self.thresholds) - set(THRESHOLDS)
        if bad:
            raise ValueError(f"unknown thresholds: {sorted(bad)}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        adaptive, sens = raw.get("adaptive", {}), raw.get("sensitivity", {})
        return cls(
            seed=raw.get("seed", 0),
            fit_fraction=raw.get("fit_fraction", 0.7),
            quantile=raw.get("quantile", 0.995),
            hold=raw.get("hold", 20),
            rule=adaptive.get("rule", "robust"),
            window=adaptive.get("window", 1000),
            max_freeze=adaptive.get("max_freeze", 200),
            models=tuple(raw.get("models", cls.models)),
            thresholds=tuple(raw.get("thresholds", cls.thresholds)),
            sens_quantiles=tuple(sens.get("quantiles", cls.sens_quantiles)),
            sens_holds=tuple(sens.get("holds", cls.sens_holds)),
            profiling_channels=raw.get("profiling_channels", 5),
            drop_constant=raw.get("drop_constant", False),
            min_thresholds=tuple(raw.get("min_thresholds", {}).items()),
            abstain_constant_train=raw.get("abstain_constant_train", False),
        )


def make_model(name: str, features: ChannelFeatures, cfg: ExperimentConfig):
    if name == "robust_z":
        return RobustZBaseline(features.feature_names)
    if name == "isolation_forest":
        return IsolationForestModel(seed=cfg.seed, drop_constant=cfg.drop_constant)
    if name == "autoencoder":
        from aai.models.autoencoder import AutoencoderModel

        return AutoencoderModel(seed=cfg.seed, drop_constant=cfg.drop_constant)
    raise ValueError(f"unknown model: {name}")


def score_channel(name: str, features: ChannelFeatures, cfg: ExperimentConfig):
    """Fit on the first part of train, return (model, calibration scores, test scores)."""
    X = features.train_X
    split = int(len(X) * cfg.fit_fraction)
    model = make_model(name, features, cfg).fit(X[:split])
    return model, model.score(X[split:]), model.score(features.test_X)


FLAT_TRAIN_STD = 0.1  # std of the scaled train signal below which a channel counts as flat


def guard_alarms(alarms: np.ndarray, scores: np.ndarray, floor: float, abstain: bool) -> np.ndarray:
    """Apply a minimum threshold and, optionally, switch a channel off entirely."""
    if abstain:
        return np.zeros_like(alarms)
    return alarms & (scores > floor)


def raw_alarms(
    kind: str, cal: np.ndarray, test: np.ndarray, cfg: ExperimentConfig, quantile: float
) -> np.ndarray:
    """Threshold crossings before the alarm hold is applied."""
    if kind == "fixed":
        return test > fixed_threshold(cal, quantile)
    guard = "none" if kind == "adaptive_none" else "freeze"
    acfg = AdaptiveConfig(
        quantile=quantile,
        rule=cfg.rule,
        window=cfg.window,
        guard=guard,
        max_freeze=cfg.max_freeze,
    )
    return adaptive_threshold(cal, test, acfg)[0]


def reference_alarms(kind: str, n: int, rate: float, seed: int) -> np.ndarray:
    if kind == "always_on":
        return np.ones(n, dtype=bool)
    if kind == "never_on":
        return np.zeros(n, dtype=bool)
    return np.random.default_rng(seed).random(n) < rate


def _evaluate(channels, scores, kind_of, quantile, hold, cfg):
    results: list[ChannelResult] = []
    for ch in channels:
        cal, test = scores[ch.name]
        alarms = kind_of(ch, cal, test, quantile, hold)
        results.append(
            evaluate_channel(ch.name, ch.spacecraft, ch.test_mask(), alarms, scores=test)
        )
    return results


def _load_or_score(model_name, channels, feats, cfg, scores_dir, profile_names):
    """Scores are cached per model, so a failed or repeated run never refits finished models."""
    path, cost_path = scores_dir / f"{model_name}.npz", scores_dir / f"{model_name}.cost.json"
    if path.exists() and cost_path.exists():
        with np.load(path) as z:
            scores = {c.name: (z[f"{c.name}__cal"], z[f"{c.name}__test"]) for c in channels}
        log.info("%s: loaded cached scores", model_name)
        return scores, json.loads(cost_path.read_text())
    scores, timings = {}, []
    for ch in channels:
        model, cal, test = score_channel(model_name, feats[ch.name], cfg)
        scores[ch.name] = (cal, test)
        if ch.name in profile_names:
            timings.append(measure_scoring(model.score, feats[ch.name].test_X))
    log.info("%s: scored %d channels", model_name, len(scores))
    np.savez_compressed(
        path,
        **{f"{n}__{k}": v for n, (c, t) in scores.items() for k, v in (("cal", c), ("test", t))},
    )
    timing = {
        "ms_per_1k_rows": float(np.median([t["ms_per_1k_rows"] for t in timings])),
        "peak_mb": float(np.max([t["peak_mb"] for t in timings])),
    }
    cost_path.write_text(json.dumps(timing))
    return scores, timing


def run_experiment(
    dataset: Dataset,
    features_dir: str | Path,
    cfg: ExperimentConfig,
    out_dir: str | Path,
    scores_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    if "autoencoder" in cfg.models:
        # Import torch before sklearn/NumPy spin up their thread pools; loading it late has
        # crashed with an MKL "cannot load libtorch_cpu" error in some environments.
        import torch  # noqa: F401

    channels = list(dataset.channels.values())
    out_dir, scores_dir = Path(out_dir), Path(scores_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)
    feats = {ch.name: load_features(Path(features_dir) / f"{ch.name}.npz") for ch in channels}

    summaries, per_channel, sensitivity, cost = {}, [], [], []
    flat = {n for n, f in feats.items() if f.train_X[:, 0].std() < FLAT_TRAIN_STD}
    log.info("%d channels have a flat train signal: %s", len(flat), sorted(flat))
    profile_names = [c.name for c in channels][:: max(1, len(channels) // cfg.profiling_channels)][
        : cfg.profiling_channels
    ]

    for model_name in cfg.models:
        scores, timing = _load_or_score(model_name, channels, feats, cfg, scores_dir, profile_names)
        cost.append({"model": model_name, **timing})

        for thr in cfg.thresholds:
            raw = {}  # quantile -> channel -> raw alarms, shared by all hold values
            for q in sorted({cfg.quantile, *cfg.sens_quantiles}):
                raw[q] = {
                    c.name: guard_alarms(
                        raw_alarms(thr, *scores[c.name], cfg, q),
                        scores[c.name][1],
                        floor=dict(cfg.min_thresholds).get(model_name, 0.0),
                        abstain=cfg.abstain_constant_train and c.name in flat,
                    )
                    for c in channels
                }
            for q, h in itertools.product(raw, sorted({cfg.hold, *cfg.sens_holds})):
                res = [
                    evaluate_channel(
                        c.name,
                        c.spacecraft,
                        c.test_mask(),
                        hold_alarms(raw[q][c.name], h),
                        scores=scores[c.name][1],
                    )
                    for c in channels
                ]
                table = summary_table(res)
                sensitivity.append(
                    {
                        "model": model_name,
                        "threshold": thr,
                        "quantile": q,
                        "hold": h,
                        **table.loc["ALL"].to_dict(),
                    }
                )
                if q == cfg.quantile and h == cfg.hold:
                    key = f"{model_name}/{thr}"
                    for label in ("ALL", "SMAP", "MSL"):
                        summaries[key if label == "ALL" else f"{key} [{label}]"] = table.loc[label]
                    per_channel.extend(
                        {"model": model_name, "threshold": thr, **r.__dict__} for r in res
                    )
            log.info("%s/%s done", model_name, thr)

    ref_rate = 1 - cfg.quantile
    any_scores = {ch.name: (np.zeros(1), np.zeros(len(ch.test))) for ch in channels}
    for kind in ("always_on", "never_on", "random"):

        def ref_fn(ch, cal, test, q, h, kind=kind):
            return reference_alarms(kind, len(test), ref_rate, cfg.seed)

        res = _evaluate(channels, any_scores, ref_fn, cfg.quantile, 0, cfg)
        summaries[f"reference/{kind}"] = summary_table(res).loc["ALL"]

    tables = {
        "summary": pd.DataFrame(summaries).T,
        "sensitivity": pd.DataFrame(sensitivity),
        "cost": pd.DataFrame(cost),
        "channels": pd.DataFrame(per_channel),
    }
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=name not in {"sensitivity", "cost", "channels"})
    (out_dir / "config.json").write_text(json.dumps(cfg.__dict__, indent=2, default=list))
    return tables
