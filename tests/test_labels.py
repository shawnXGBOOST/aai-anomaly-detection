import pytest

from aai.data.labels import (
    LabelError,
    intervals_to_mask,
    load_labels,
    parse_classes,
    parse_intervals,
)


def test_parse_intervals_keeps_file_order():
    assert parse_intervals("[[1778, 1898], [1238, 1344]]") == ((1778, 1898), (1238, 1344))


@pytest.mark.parametrize("bad", ["not a list", "[[5, 3]]", "[[-1, 4]]", "[[1, 2, 3]]", "[1, 2]"])
def test_parse_intervals_rejects_malformed(bad):
    with pytest.raises(LabelError):
        parse_intervals(bad)


def test_parse_classes():
    assert parse_classes("[contextual, point]") == ("contextual", "point")
    assert parse_classes("[point]") == ("point",)


def test_mask_end_is_inclusive():
    mask = intervals_to_mask(((2, 4),), 10)
    assert mask.tolist() == [False, False, True, True, True] + [False] * 5


def test_mask_rejects_interval_past_end():
    with pytest.raises(LabelError):
        intervals_to_mask(((8, 10),), 10)
    assert intervals_to_mask(((8, 9),), 10)[-1]  # last valid index is allowed


def test_conflicting_duplicates_are_ambiguous(synthetic_dir):
    table = load_labels(synthetic_dir / "labeled_anomalies.csv")
    assert "P-2" in table.ambiguous
    assert "P-2" not in table.usable
    assert set(table.usable) == {"A-1", "D-1", "M-1"}
    assert table.n_rows == 5


def test_identical_duplicates_collapse(tmp_path):
    csv = tmp_path / "labels.csv"
    csv.write_text(
        "chan_id,spacecraft,anomaly_sequences,class,num_values\n"
        'X-1,SMAP,"[[1, 2]]",[point],10\n'
        'X-1,SMAP,"[[1, 2]]",[point],10\n'
    )
    table = load_labels(csv)
    assert set(table.usable) == {"X-1"}
    assert not table.ambiguous


def test_missing_columns_rejected(tmp_path):
    csv = tmp_path / "labels.csv"
    csv.write_text("chan_id,spacecraft\nX-1,SMAP\n")
    with pytest.raises(LabelError):
        load_labels(csv)
