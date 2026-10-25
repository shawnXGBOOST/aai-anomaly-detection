import numpy as np

from aai.cli import main
from aai.data.labels import load_labels
from aai.data.report import channel_table, render_report


def test_channel_table_flags(synthetic_dir):
    # Make D-1's train telemetry constant and push its test telemetry out of range.
    train = np.load(synthetic_dir / "train" / "D-1.npy")
    train[:, 0] = 1.0
    np.save(synthetic_dir / "train" / "D-1.npy", train)
    test = np.load(synthetic_dir / "test" / "D-1.npy")
    test[0, 0] = 5.0
    np.save(synthetic_dir / "test" / "D-1.npy", test)

    table = load_labels(synthetic_dir / "labeled_anomalies.csv")
    df = channel_table(synthetic_dir, table).set_index("channel")

    assert len(df) == 5
    assert df.loc["A-1", "test_len"] == 60
    assert df.loc["A-1", "anomalous_points"] == 20
    assert bool(df.loc["D-1", "train_constant_telemetry"])
    assert bool(df.loc["D-1", "test_outside_train_range"])
    assert df.loc["P-2", "label_status"] == "ambiguous"
    assert df.loc["T-10", "label_status"] == "unlabeled"
    assert df.loc["M-1", "spacecraft"] == "MSL"


def test_report_lists_exclusions_and_scored_count(synthetic_dir):
    table = load_labels(synthetic_dir / "labeled_anomalies.csv")
    report = render_report(channel_table(synthetic_dir, table), table)
    assert "**Scored set:** 3 channels" in report
    assert "`P-2`" in report
    assert "`T-10`" in report
    assert "inclusive" in report


def test_cli_inspect_data_writes_outputs(synthetic_dir, tmp_path):
    out, csv = tmp_path / "report.md", tmp_path / "summary.csv"
    code = main(
        ["inspect-data", "--data-dir", str(synthetic_dir), "--out", str(out), "--csv", str(csv)]
    )
    assert code == 0
    assert out.read_text(encoding="utf-8").startswith("# Data report")
    assert csv.is_file()


def test_cli_returns_error_code_on_bad_input(tmp_path):
    assert main(["prepare-data", "--archive", str(tmp_path / "missing.zip")]) == 1
