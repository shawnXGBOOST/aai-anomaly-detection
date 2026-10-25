import numpy as np
import pytest

from aai.cli import main
from aai.features.engineer import FeatureConfig, build_features
from aai.features.pipeline import load_features, make_features, save_features
from aai.features.preprocess import TelemetryScaler, clean_nonfinite

CFG = FeatureConfig(windows=(5, 10))


def make_channel(n_train=80, n_test=120, n_flags=4, seed=0):
    rng = np.random.default_rng(seed)
    arrays = []
    for n in (n_train, n_test):
        telemetry = np.sin(np.arange(n) / 7.0) + 0.1 * rng.normal(size=n)
        flags = (rng.random((n, n_flags)) > 0.8).astype(float)
        arrays.append(np.column_stack([telemetry, flags]))
    return arrays


# ---- cleaning -------------------------------------------------------------------------------


def test_forward_fill_uses_only_past_values():
    arr = np.array([[1.0], [np.nan], [np.inf], [4.0], [np.nan]])
    cleaned, n = clean_nonfinite(arr, fill=np.array([0.0]))
    assert cleaned[:, 0].tolist() == [1.0, 1.0, 1.0, 4.0, 4.0]
    assert n == 3


def test_leading_gap_uses_previous_row_then_fill():
    arr = np.array([[np.nan], [2.0]])
    assert clean_nonfinite(arr, fill=np.array([9.0]))[0][0, 0] == 9.0
    assert clean_nonfinite(arr, fill=np.array([9.0]), previous=np.array([5.0]))[0][0, 0] == 5.0


def test_clean_raises_when_nothing_to_fill_from():
    with pytest.raises(ValueError):
        clean_nonfinite(np.array([[np.nan]]), fill=np.array([np.nan]))


def test_clean_does_not_modify_its_input():
    arr = np.array([[1.0], [np.nan]])
    clean_nonfinite(arr, fill=np.array([0.0]))
    assert np.isnan(arr[1, 0])


# ---- scaling --------------------------------------------------------------------------------


def test_scaler_matches_train_statistics():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    scaler = TelemetryScaler.fit(x, min_scale=0.01)
    assert scaler.center == pytest.approx(2.5)
    assert scaler.scale == pytest.approx(x.std())


def test_scaler_floors_constant_channels():
    scaler = TelemetryScaler.fit(np.full(50, 0.3), min_scale=0.05)
    assert scaler.scale == 0.05
    assert np.isfinite(scaler.transform(np.array([0.3, 0.9]))).all()
    assert scaler.transform(np.array([0.9]))[0] == pytest.approx(12.0)


def test_scaler_dict_roundtrip():
    scaler = TelemetryScaler(1.5, 2.5)
    assert TelemetryScaler.from_dict(scaler.to_dict()) == scaler


# ---- features: causality --------------------------------------------------------------------


def test_features_never_use_future_values():
    rng = np.random.default_rng(3)
    scaled, flags = rng.normal(size=200), (rng.random((200, 4)) > 0.5).astype(float)
    t0 = 120
    future_scaled, future_flags = scaled.copy(), flags.copy()
    future_scaled[t0 + 1 :] += 100 * rng.normal(size=200 - t0 - 1)
    future_flags[t0 + 1 :] = 1 - future_flags[t0 + 1 :]

    original, _ = build_features(scaled, flags, CFG)
    perturbed, _ = build_features(future_scaled, future_flags, CFG)
    np.testing.assert_allclose(original[: t0 + 1], perturbed[: t0 + 1], equal_nan=True)


def test_robust_z_excludes_the_current_point():
    scaled = np.zeros(100)
    scaled[60] = 10.0  # one spike in an otherwise flat series
    X, names = build_features(scaled, np.zeros((100, 2)), FeatureConfig(windows=(5,)))
    z = X[:, names.index("robust_z_5")]
    assert z[60] == pytest.approx(10.0 / 0.1)  # baseline is 0 with the floor spread 0.1
    assert np.isnan(z[:5]).all()
    assert not np.isnan(z[5:]).any()


def test_flag_summaries():
    flags = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    X, names = build_features(np.zeros(4), flags, FeatureConfig(windows=(3,)))
    assert X[:, names.index("flags_active")].tolist() == [0, 1, 2, 1]
    changed = X[:, names.index("flags_changed")]
    assert np.isnan(changed[0])
    assert changed[1:].tolist() == [1, 1, 1]


def test_build_features_rejects_bad_shapes():
    with pytest.raises(ValueError):
        build_features(np.zeros(5), np.zeros((4, 2)), CFG)


# ---- pipeline: no leakage from test ----------------------------------------------------------


def test_shapes_and_no_nan():
    train, test = make_channel()
    feats = make_features("X-1", train, test, CFG)
    assert feats.warmup == 10
    assert feats.train_X.shape == (80 - 10, len(feats.feature_names))
    assert feats.test_X.shape == (120, len(feats.feature_names))
    assert np.isfinite(feats.train_X).all() and np.isfinite(feats.test_X).all()


def test_train_features_and_scaler_do_not_depend_on_test():
    train, test_a = make_channel(seed=1)
    test_b = test_a + 50.0
    a = make_features("X", train, test_a, CFG)
    b = make_features("X", train, test_b, CFG)
    np.testing.assert_array_equal(a.train_X, b.train_X)
    assert a.scaler == b.scaler


def test_test_features_match_one_continuous_computation():
    train, test = make_channel(seed=2)
    feats = make_features("X", train, test, CFG)
    scaler = TelemetryScaler.fit(train[:, 0], CFG.min_scale)
    both = np.vstack([train, test])
    expected, _ = build_features(scaler.transform(both[:, 0]), both[:, 1:], CFG)
    np.testing.assert_allclose(feats.test_X, expected[len(train) :])


def test_earlier_test_features_ignore_later_test_values():
    train, test = make_channel(seed=4)
    changed = test.copy()
    changed[80:, 0] += 5.0
    a = make_features("X", train, test, CFG)
    b = make_features("X", train, changed, CFG)
    np.testing.assert_array_equal(a.test_X[:80], b.test_X[:80])
    assert not np.array_equal(a.test_X[80:], b.test_X[80:])


def test_non_finite_values_are_filled_and_counted():
    train, test = make_channel(seed=5)
    test[10, 0] = np.nan
    test[11, 2] = np.inf
    feats = make_features("X", train, test, CFG)
    assert feats.n_replaced_test == 2
    assert feats.n_replaced_train == 0
    assert np.isfinite(feats.test_X).all()


def test_train_shorter_than_warmup_is_an_error():
    train, test = make_channel(n_train=10)
    with pytest.raises(ValueError, match="warmup"):
        make_features("X", train, test, CFG)


def test_save_load_roundtrip(tmp_path):
    train, test = make_channel()
    feats = make_features("X-1", train, test, CFG)
    loaded = load_features(save_features(feats, tmp_path))
    np.testing.assert_array_equal(loaded.train_X, feats.train_X)
    np.testing.assert_array_equal(loaded.test_X, feats.test_X)
    assert loaded.feature_names == feats.feature_names
    assert loaded.scaler == feats.scaler
    assert loaded.warmup == feats.warmup
    assert loaded.name == "X-1"


# ---- config and CLI -------------------------------------------------------------------------


def test_config_validation():
    with pytest.raises(ValueError):
        FeatureConfig(windows=())
    with pytest.raises(ValueError):
        FeatureConfig(windows=(2,))
    with pytest.raises(ValueError):
        FeatureConfig(min_scale=0)
    with pytest.raises(ValueError, match="unknown"):
        FeatureConfig.from_dict({"window": [5]})


def test_config_yaml_roundtrip(tmp_path):
    path = tmp_path / "f.yaml"
    path.write_text("windows: [4, 8]\nmin_scale: 0.1\nmad_floor: 0.2\n")
    cfg = FeatureConfig.from_yaml(path)
    assert cfg == FeatureConfig(windows=(4, 8), min_scale=0.1, mad_floor=0.2)
    assert cfg.warmup == 8
    assert FeatureConfig.from_dict(cfg.to_dict()) == cfg


def test_cli_build_features(synthetic_dir, tmp_path):
    config = tmp_path / "f.yaml"
    config.write_text("windows: [3, 5]\n")
    out = tmp_path / "processed"
    code = main(
        [
            "build-features",
            "--data-dir",
            str(synthetic_dir),
            "--out-dir",
            str(out),
            "--config",
            str(config),
        ]
    )
    assert code == 0
    assert sorted(p.name for p in out.glob("*.npz")) == ["A-1.npz", "D-1.npz", "M-1.npz"]
    assert (out / "manifest.json").is_file()


def test_cli_build_features_reports_bad_config(synthetic_dir, tmp_path):
    config = tmp_path / "f.yaml"
    config.write_text("windows: []\n")
    args = ["build-features", "--data-dir", str(synthetic_dir), "--config", str(config)]
    assert main(args + ["--out-dir", str(tmp_path / "o")]) == 1
