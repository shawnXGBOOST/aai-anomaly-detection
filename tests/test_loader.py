import numpy as np
import pytest

from aai.data.loader import DataValidationError, channel_names, load_dataset


def test_problem_channels_are_excluded_with_reason(synthetic_dir):
    dataset = load_dataset(synthetic_dir)
    assert set(dataset.channels) == {"A-1", "D-1", "M-1"}
    assert set(dataset.excluded) == {"P-2", "T-10"}
    assert "ambiguous" in dataset.excluded["P-2"]
    assert "no row" in dataset.excluded["T-10"]


def test_channel_contents(synthetic_dir):
    channel = load_dataset(synthetic_dir).channels["A-1"]
    assert channel.spacecraft == "SMAP"
    assert channel.n_features == 25
    assert channel.train.shape == (30, 25)
    assert channel.test.shape == (60, 25)
    assert channel.test_mask().sum() == 20  # two inclusive intervals of 10 points


def test_unpaired_files_rejected(synthetic_dir):
    (synthetic_dir / "train" / "M-1.npy").unlink()
    with pytest.raises(DataValidationError, match="pair up"):
        channel_names(synthetic_dir)


def test_feature_count_mismatch_between_train_and_test(synthetic_dir):
    np.save(synthetic_dir / "test" / "D-1.npy", np.zeros((50, 7)))
    with pytest.raises(DataValidationError, match="features"):
        load_dataset(synthetic_dir)


def test_label_length_must_match_test_length(synthetic_dir):
    np.save(synthetic_dir / "test" / "D-1.npy", np.zeros((49, 25)))
    with pytest.raises(DataValidationError, match="num_values"):
        load_dataset(synthetic_dir)


def test_spacecraft_must_match_feature_count(synthetic_dir):
    csv = synthetic_dir / "labeled_anomalies.csv"
    csv.write_text(csv.read_text().replace("M-1,MSL", "M-1,SMAP"))
    with pytest.raises(DataValidationError, match="features"):
        load_dataset(synthetic_dir)


def test_labeled_channel_without_data_is_an_error(synthetic_dir):
    for split in ("train", "test"):
        (synthetic_dir / split / "D-1.npy").unlink()
    with pytest.raises(DataValidationError, match="missing"):
        load_dataset(synthetic_dir)
