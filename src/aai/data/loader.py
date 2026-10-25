"""Loading and structural validation of the telemanom channel arrays.

Each channel has a ``train`` and a ``test`` array of shape (time, features). Column 0 is the
telemetry value, the remaining columns are 0/1 command flags. SMAP channels have 25 features
and MSL channels have 55.

The loader checks structure (shapes, file pairing, label consistency). Non-finite values are
*not* rejected here: they are counted by the inspection report and handled in preprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aai.data.labels import Interval, intervals_to_mask, load_labels

SPACECRAFT_BY_N_FEATURES = {25: "SMAP", 55: "MSL"}
LABELS_FILENAME = "labeled_anomalies.csv"


class DataValidationError(ValueError):
    """Raised when the raw data does not have the expected structure."""


@dataclass(frozen=True, eq=False)
class Channel:
    name: str
    spacecraft: str
    train: np.ndarray  # (n_train, n_features), assumed anomaly-free by the benchmark
    test: np.ndarray  # (n_test, n_features), contains the labeled anomalies
    intervals: tuple[Interval, ...]
    classes: tuple[str, ...]

    @property
    def n_features(self) -> int:
        return self.train.shape[1]

    def test_mask(self) -> np.ndarray:
        """Boolean ground truth for the test series."""
        return intervals_to_mask(self.intervals, len(self.test))


@dataclass(frozen=True)
class Dataset:
    channels: dict[str, Channel]  # channels with trustworthy labels (the scored set)
    excluded: dict[str, str]  # channel name -> reason it is not scored


def channel_names(data_dir: str | Path) -> list[str]:
    """Names of channels that have both a train and a test file."""
    data_dir = Path(data_dir)
    train = {p.stem for p in (data_dir / "train").glob("*.npy")}
    test = {p.stem for p in (data_dir / "test").glob("*.npy")}
    if train != test:
        raise DataValidationError(
            f"train/test files do not pair up: only in train {sorted(train - test)}, "
            f"only in test {sorted(test - train)}"
        )
    if not train:
        raise DataValidationError(f"no .npy files found under {data_dir}/train and /test")
    return sorted(train)


def load_arrays(data_dir: str | Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one channel's train and test arrays and check their shapes."""
    data_dir = Path(data_dir)
    arrays = []
    for split in ("train", "test"):
        # allow_pickle=False: the files come from an archive we do not control.
        arr = np.asarray(np.load(data_dir / split / f"{name}.npy", allow_pickle=False), dtype=float)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
            raise DataValidationError(f"{name} {split}: expected 2-D non-empty, got {arr.shape}")
        arrays.append(arr)
    train, test = arrays
    if train.shape[1] != test.shape[1]:
        raise DataValidationError(
            f"{name}: train has {train.shape[1]} features, test has {test.shape[1]}"
        )
    return train, test


def load_dataset(data_dir: str | Path, labels_path: str | Path | None = None) -> Dataset:
    """Load every channel with usable labels; list the others with the reason they are excluded."""
    data_dir = Path(data_dir)
    table = load_labels(labels_path or data_dir / LABELS_FILENAME)
    names = channel_names(data_dir)

    labeled_without_data = sorted(set(table.usable) - set(names))
    if labeled_without_data:
        raise DataValidationError(
            f"labels exist but data files are missing: {labeled_without_data}"
        )

    channels: dict[str, Channel] = {}
    excluded: dict[str, str] = {}
    for name in names:
        if name in table.ambiguous:
            excluded[name] = f"ambiguous labels ({table.ambiguous[name]})"
            continue
        entry = table.usable.get(name)
        if entry is None:
            excluded[name] = f"no row in {LABELS_FILENAME}"
            continue

        train, test = load_arrays(data_dir, name)
        if len(test) != entry.num_values:
            raise DataValidationError(
                f"{name}: test length {len(test)} != num_values {entry.num_values} in labels"
            )
        expected = SPACECRAFT_BY_N_FEATURES.get(train.shape[1])
        if expected != entry.spacecraft:
            raise DataValidationError(
                f"{name}: labeled {entry.spacecraft} but has {train.shape[1]} features"
            )
        channels[name] = Channel(
            name=name,
            spacecraft=entry.spacecraft,
            train=train,
            test=test,
            intervals=entry.intervals,
            classes=entry.classes,
        )
    return Dataset(channels=channels, excluded=excluded)
