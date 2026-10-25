"""Preparing the raw telemanom data from a downloaded archive.

The original download link (an S3 bucket) is no longer available. The data is distributed as a
Kaggle dataset, ``patrickfleith/nasa-anomaly-detection-dataset-smap-msl``, which needs a Kaggle
login, so the archive is downloaded by hand and handed to :func:`prepare_data`.

The archive also contains outputs of the original paper's pretrained models (``y_hat``,
``smoothed_errors``, ``models``). They are deliberately ignored: only ``train/*.npy`` and
``test/*.npy`` are extracted.
"""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from aai.data.loader import LABELS_FILENAME, channel_names

LABELS_URL = "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv"
SPLITS = ("train", "test")
MAX_UNCOMPRESSED_BYTES = 2 * 1024**3


class DataPrepError(RuntimeError):
    """Raised when the archive or the label download cannot be used."""


def _is_unsafe(name: str) -> bool:
    path = PurePosixPath(name)
    return path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name


def extract_arrays(archive: str | Path, out_dir: str | Path) -> dict[str, int]:
    """Extract only ``<anything>/train/X.npy`` and ``<anything>/test/X.npy`` into ``out_dir``.

    Members are copied one by one to a path built from the split and file name only, so the
    archive cannot write anywhere else. Returns the number of files extracted per split.
    """
    out_dir = Path(out_dir)
    counts = {split: 0 for split in SPLITS}
    seen: set[tuple[str, str]] = set()
    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        if sum(info.file_size for info in infos) > MAX_UNCOMPRESSED_BYTES:
            raise DataPrepError("archive is unexpectedly large, refusing to extract")
        for info in infos:
            if _is_unsafe(info.filename):
                raise DataPrepError(f"unsafe path in archive: {info.filename!r}")
            path = PurePosixPath(info.filename)
            if info.is_dir() or path.suffix != ".npy" or path.parent.name not in SPLITS:
                continue
            key = (path.parent.name, path.name)
            if key in seen:
                raise DataPrepError(f"archive contains {key[0]}/{key[1]} more than once")
            seen.add(key)
            target = out_dir / key[0] / key[1]
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            counts[key[0]] += 1
    return counts


def download_labels(dest: str | Path, url: str = LABELS_URL, timeout: float = 30.0) -> Path:
    """Download the label CSV from the original telemanom repository."""
    if not url.startswith("https://"):
        raise DataPrepError(f"refusing non-HTTPS url: {url}")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            dest.write_bytes(response.read())
    except OSError as exc:
        raise DataPrepError(
            f"could not download labels from {url}: {exc}. "
            "Download it manually and pass it with --labels."
        ) from exc
    return dest


def prepare_data(
    archive: str | Path, data_dir: str | Path, labels: str | Path | None = None
) -> dict[str, object]:
    """Extract the archive into ``data_dir`` and make sure the label file is there."""
    archive, data_dir = Path(archive), Path(data_dir)
    if not archive.is_file():
        raise DataPrepError(f"archive not found: {archive}")

    counts = extract_arrays(archive, data_dir)
    if counts["train"] == 0 and counts["test"] == 0:
        raise DataPrepError("no train/test .npy files found in the archive")
    n_channels = len(channel_names(data_dir))  # raises if train and test do not pair up

    labels_dest = data_dir / LABELS_FILENAME
    if labels is not None:
        shutil.copyfile(labels, labels_dest)
    elif not labels_dest.exists():
        download_labels(labels_dest)
    return {"channels": n_channels, "labels_file": str(labels_dest)}
