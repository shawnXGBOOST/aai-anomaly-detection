"""Command-line interface: ``aai prepare-data``, ``inspect-data`` and ``build-features``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from aai.data.labels import load_labels
from aai.data.loader import LABELS_FILENAME, load_dataset
from aai.data.prepare import DataPrepError, prepare_data
from aai.data.report import channel_table, render_report
from aai.features.engineer import FeatureConfig
from aai.features.pipeline import build_all
from aai.serving.detector import ArtifactError

log = logging.getLogger("aai")

DEFAULT_DATA_DIR = Path("data/raw/telemanom")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aai", description="Adaptive Anomaly Intelligence")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare-data", help="extract the downloaded archive and fetch labels")
    prep.add_argument("--archive", type=Path, required=True, help="zip downloaded from Kaggle")
    prep.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    prep.add_argument("--labels", type=Path, help="local labeled_anomalies.csv (skips download)")

    insp = sub.add_parser("inspect-data", help="write the data report from the raw files")
    insp.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    insp.add_argument("--out", type=Path, default=Path("docs/data_report.md"))
    insp.add_argument("--csv", type=Path, default=Path("docs/channel_summary.csv"))

    feat = sub.add_parser("build-features", help="clean, scale and build features per channel")
    feat.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    feat.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    feat.add_argument("--config", type=Path, default=Path("configs/features.yaml"))
    exp = sub.add_parser("run-experiment", help="score all models and thresholds, write results")
    exp.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    exp.add_argument("--features-dir", type=Path, default=Path("data/processed"))
    exp.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    exp.add_argument("--out-dir", type=Path, default=Path("experiments/results"))
    exp.add_argument("--scores-dir", type=Path, default=Path("artifacts/scores"))
    trn = sub.add_parser("train", help="fit detectors per channel and save serving artifacts")
    trn.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    trn.add_argument("--config", type=Path, default=Path("configs/features.yaml"))
    trn.add_argument("--out-dir", type=Path, default=Path("artifacts/models"))
    trn.add_argument("--models", nargs="+", default=["robust_z", "isolation_forest", "autoencoder"])
    det = sub.add_parser("detect", help="score a .npy telemetry file with a saved detector")
    det.add_argument("--artifacts", type=Path, default=Path("artifacts/models"))
    det.add_argument("--channel", required=True)
    det.add_argument("--model", default="robust_z")
    det.add_argument("--input", type=Path, required=True, help=".npy array, column 0 = telemetry")
    srv = sub.add_parser("serve", help="run the HTTP API")
    srv.add_argument("--artifacts", type=Path, default=Path("artifacts/models"))
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare-data":
            result = prepare_data(args.archive, args.data_dir, args.labels)
            log.info("prepared %s channels in %s", result["channels"], args.data_dir)
        elif args.command == "inspect-data":
            table = load_labels(args.data_dir / LABELS_FILENAME)
            df = channel_table(args.data_dir, table)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(render_report(df, table), encoding="utf-8")
            df.to_csv(args.csv, index=False)
            log.info("wrote %s and %s", args.out, args.csv)
        elif args.command == "run-experiment":
            from aai.experiment import ExperimentConfig, run_experiment

            tables = run_experiment(
                load_dataset(args.data_dir),
                args.features_dir,
                ExperimentConfig.from_yaml(args.config),
                args.out_dir,
                args.scores_dir,
            )
            log.info("wrote %d result tables to %s", len(tables), args.out_dir)
        elif args.command == "train":
            from aai.serving.detector import train_all

            count = train_all(
                load_dataset(args.data_dir),
                FeatureConfig.from_yaml(args.config),
                args.out_dir,
                models=tuple(args.models),
            )
            log.info("saved %d detectors to %s", count, args.out_dir)
        elif args.command == "detect":
            import json

            import numpy as np

            from aai.serving.detector import load_detector

            detector = load_detector(args.artifacts, args.model, args.channel)
            result = detector.detect(np.load(args.input, allow_pickle=False))
            print(
                json.dumps(
                    {
                        "channel": args.channel,
                        "model": args.model,
                        "threshold": result.threshold,
                        "first_scored_row": result.first_scored_row,
                        "events": [e.__dict__ for e in result.events],
                    },
                    indent=2,
                )
            )
        elif args.command == "serve":
            import os

            import uvicorn

            os.environ["AAI_ARTIFACTS"] = str(args.artifacts)
            uvicorn.run("aai.serving.api:create_app", factory=True, host=args.host, port=args.port)
        else:
            cfg = FeatureConfig.from_yaml(args.config)
            manifest = build_all(load_dataset(args.data_dir), cfg, args.out_dir)
            log.info(
                "wrote features for %d channels to %s", len(manifest["channels"]), args.out_dir
            )
    except (DataPrepError, ValueError, OSError, yaml.YAMLError, ArtifactError) as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
