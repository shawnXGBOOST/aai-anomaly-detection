import numpy as np
import pytest

from aai.evaluation.profiling import measure_scoring
from aai.models.baseline import RobustZBaseline
from aai.models.iforest import IsolationForestModel

NAMES = ["value", "robust_z_10", "flags_active", "robust_z_50"]


def _normal(n=500, seed=0):
    return np.random.default_rng(seed).normal(size=(n, 4))


def test_baseline_uses_max_abs_robust_z():
    m = RobustZBaseline(NAMES).fit(np.zeros((1, 4)))
    X = np.array([[0, -3.0, 9, 1.0], [0, 1.0, 9, 2.0]])
    assert m.score(X).tolist() == [3.0, 2.0]
    with pytest.raises(ValueError):
        RobustZBaseline(["a", "b"])


def test_iforest_scores_outliers_higher_and_is_deterministic():
    X = _normal()
    m = IsolationForestModel(n_estimators=50, seed=0).fit(X)
    out = m.score(np.full((5, 4), 8.0)).mean()
    assert out > m.score(X[:50]).mean()
    assert np.allclose(m.score(X), IsolationForestModel(n_estimators=50, seed=0).fit(X).score(X))
    with pytest.raises(RuntimeError):
        IsolationForestModel().score(X)


def test_autoencoder_flags_outliers_and_attributes_features():
    pytest.importorskip("torch")
    from aai.models.autoencoder import AutoencoderModel

    X = _normal(800)
    m = AutoencoderModel(epochs=15, seed=0).fit(X)
    bad = np.zeros((5, 4))
    bad[:, 2] = 12.0
    assert m.score(bad).mean() > 5 * m.score(X).mean()
    assert m.feature_errors(bad).mean(axis=0).argmax() == 2
    again = AutoencoderModel(epochs=15, seed=0).fit(X)
    assert np.allclose(m.score(X[:20]), again.score(X[:20]), atol=1e-5)
    with pytest.raises(RuntimeError):
        AutoencoderModel().score(X)


def test_measure_scoring_returns_positive_numbers():
    X = _normal(200)
    r = measure_scoring(lambda a: np.abs(a).max(axis=1), X, repeats=2)
    assert r["ms_per_1k_rows"] > 0 and r["peak_mb"] >= 0
