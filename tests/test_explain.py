import numpy as np
import pytest

from aai.explain import characterize_event, top_features
from aai.models.baseline import RobustZBaseline
from aai.models.iforest import IsolationForestModel

NAMES = ["value", "robust_z_10", "flags_active", "robust_z_50"]


def test_top_features_ranks_and_sums_shares():
    c = np.array([[0.0, 3.0, 0.0, 1.0], [0.0, 3.0, 0.0, 1.0]])
    out = top_features(c, NAMES, k=2)
    assert [n for n, _ in out] == ["robust_z_10", "robust_z_50"]
    assert out[0][1] == pytest.approx(0.75)
    assert top_features(np.zeros((2, 4)), NAMES) == []
    with pytest.raises(ValueError):
        top_features(c, NAMES[:3])


def test_baseline_contributions_only_robust_z_columns():
    m = RobustZBaseline(NAMES)
    c = m.contributions(np.array([[5.0, -2.0, 9.0, 1.0]]))
    assert c.tolist() == [[0.0, 2.0, 0.0, 1.0]]


def test_iforest_proxy_points_at_the_odd_feature():
    X = np.random.default_rng(0).normal(size=(400, 4))
    m = IsolationForestModel(n_estimators=30).fit(X)
    odd = np.zeros((1, 4))
    odd[0, 2] = 10
    assert m.contributions(odd).argmax() == 2


def test_autoencoder_contributions_match_feature_errors():
    pytest.importorskip("torch")
    from aai.models.autoencoder import AutoencoderModel

    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    X[:, 1] = 0.0  # constant feature
    m = AutoencoderModel(epochs=5, drop_constant=True).fit(X)
    c = m.contributions(X[:10])
    assert c.shape == (10, 4)
    assert (c[:, 1] == 0).all()
    assert np.allclose(c.sum(axis=1) / 3, m.score(X[:10]), atol=1e-5)


def test_characterize_spike_and_shift():
    x = np.random.default_rng(1).normal(0, 0.1, 400)
    x[300:303] += 5
    x[350:] += 3
    spike = characterize_event(x, 300, 302)
    shift = characterize_event(x, 350, 399)
    assert spike.label == "spike" and spike.length == 3
    assert shift.label == "sustained shift" and shift.shift_in_mad > 5
    with pytest.raises(ValueError):
        characterize_event(x, 390, 500)


def test_characterize_variance_change():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 0.1, 600)
    x[400:500] = rng.normal(0, 0.8, 100)
    kind = characterize_event(x, 400, 499)
    assert kind.label == "variance change" and kind.spread_ratio > 2
