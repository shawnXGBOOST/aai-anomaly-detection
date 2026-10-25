import io
import zipfile

import numpy as np
import pytest

from aai.data.loader import DataValidationError
from aai.data.prepare import DataPrepError, extract_arrays, prepare_data


def npy_bytes() -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.zeros((4, 3)))
    return buffer.getvalue()


def make_zip(path, members: dict[str, bytes]):
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return path


def test_extracts_only_train_and_test_arrays(tmp_path):
    archive = make_zip(
        tmp_path / "a.zip",
        {
            "data/data/train/A-1.npy": npy_bytes(),
            "data/data/test/A-1.npy": npy_bytes(),
            "data/data/2018-05-19_15.00.10/y_hat/A-1.npy": npy_bytes(),  # pretrained outputs
            "data/data/2018-05-19_15.00.10/models/A-1.h5": b"x",
        },
    )
    out = tmp_path / "out"
    assert extract_arrays(archive, out) == {"train": 1, "test": 1}
    found = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert found == ["test/A-1.npy", "train/A-1.npy"]


@pytest.mark.parametrize("name", ["../evil.npy", "/abs/train/x.npy", "C:/evil/train/x.npy"])
def test_unsafe_paths_are_rejected(tmp_path, name):
    archive = make_zip(tmp_path / "a.zip", {name: b"x"})
    with pytest.raises(DataPrepError, match="unsafe"):
        extract_arrays(archive, tmp_path / "out")


def test_duplicate_targets_are_rejected(tmp_path):
    archive = make_zip(
        tmp_path / "a.zip",
        {"one/train/A-1.npy": npy_bytes(), "two/train/A-1.npy": npy_bytes()},
    )
    with pytest.raises(DataPrepError, match="more than once"):
        extract_arrays(archive, tmp_path / "out")


def test_prepare_data_uses_local_labels_without_network(tmp_path):
    archive = make_zip(
        tmp_path / "a.zip",
        {"d/train/A-1.npy": npy_bytes(), "d/test/A-1.npy": npy_bytes()},
    )
    labels = tmp_path / "labels.csv"
    labels.write_text("chan_id,spacecraft,anomaly_sequences,class,num_values\n")
    result = prepare_data(archive, tmp_path / "out", labels=labels)
    assert result["channels"] == 1
    assert (tmp_path / "out" / "labeled_anomalies.csv").is_file()


def test_prepare_data_requires_paired_files(tmp_path):
    archive = make_zip(tmp_path / "a.zip", {"d/train/A-1.npy": npy_bytes()})
    with pytest.raises(DataValidationError, match="pair up"):
        prepare_data(archive, tmp_path / "out", labels=tmp_path / "unused.csv")


def test_prepare_data_missing_archive(tmp_path):
    with pytest.raises(DataPrepError, match="not found"):
        prepare_data(tmp_path / "nope.zip", tmp_path / "out")
