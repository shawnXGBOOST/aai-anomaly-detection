import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from aai.data.loader import Channel
from aai.features.engineer import FeatureConfig
from aai.serving.api import create_app
from aai.serving.detector import ArtifactError, load_detector, save_detector, train_detector

N_FEATURES = 4  # telemetry + 3 command flags


def _series(n, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    value = np.sin(t / 15) + rng.normal(0, 0.05, n)
    flags = (rng.random((n, N_FEATURES - 1)) < 0.05).astype(float)
    return np.column_stack([value, flags])


@pytest.fixture
def channel():
    return Channel("X-1", "SMAP", _series(1500, 0), _series(600, 1), [], [])


@pytest.fixture
def anomalous_test():
    test = _series(600, 1)
    test[300:340, 0] += 4.0  # a clear level shift
    return test


def _train(name, channel, **kw):
    return train_detector(name, channel, FeatureConfig(), **kw)


@pytest.mark.parametrize("name", ["robust_z", "isolation_forest"])
def test_roundtrip_gives_identical_scores(name, channel, anomalous_test, tmp_path):
    det = _train(name, channel)
    save_detector(det, tmp_path)
    loaded = load_detector(tmp_path, name, "X-1")
    a, b = det.detect(anomalous_test), loaded.detect(anomalous_test)
    assert np.array_equal(a.scores, b.scores)
    assert a.threshold == b.threshold


def test_autoencoder_roundtrip(channel, anomalous_test, tmp_path):
    pytest.importorskip("torch")
    det = _train("autoencoder", channel)
    save_detector(det, tmp_path)
    loaded = load_detector(tmp_path, "autoencoder", "X-1")
    assert np.allclose(
        det.detect(anomalous_test).scores, loaded.detect(anomalous_test).scores, atol=1e-6
    )


def test_detects_injected_shift_with_explanation(channel, anomalous_test):
    result = _train("robust_z", channel, min_threshold=3.5).detect(anomalous_test)
    hits = [e for e in result.events if e.start <= 340 and e.end >= 300]
    assert hits, "the injected level shift should raise an event"
    assert hits[0].top_features and hits[0].kind in {"sustained shift", "no clear change"}
    clean = _train("robust_z", channel, min_threshold=3.5).detect(_series(600, 1))
    assert len(clean.events) <= 3


def test_detect_validates_input(channel):
    det = _train("robust_z", channel)
    with pytest.raises(ValueError):
        det.detect(np.zeros((500, N_FEATURES + 1)))
    with pytest.raises(ValueError):
        det.detect(np.zeros((10, N_FEATURES)))


def test_missing_values_are_filled_and_counted(channel):
    test = _series(300, 3)
    test[100, 0] = np.nan
    result = _train("robust_z", channel).detect(test)
    assert result.n_replaced == 1 and np.isfinite(result.scores).all()


def test_missing_artifacts_raise(tmp_path):
    with pytest.raises(ArtifactError):
        load_detector(tmp_path, "robust_z", "nope")


@pytest.fixture
def client(channel, tmp_path):
    save_detector(_train("robust_z", channel, min_threshold=3.5), tmp_path)
    return TestClient(create_app(tmp_path))


def test_api_health_and_channels(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/channels").json() == {"robust_z": ["X-1"]}


def test_api_detect_flags_shift(client, anomalous_test):
    body = {"channel": "X-1", "rows": anomalous_test.tolist()}
    r = client.post("/detect", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["first_scored_row"] == 50 and len(data["scores"]) == 550
    assert any(e["start"] <= 340 and e["end"] >= 300 for e in data["events"])
    assert json.dumps(data)  # serialisable
    lean = client.post("/detect", json={**body, "include_scores": False}).json()
    assert lean["scores"] is None


def test_api_errors(client):
    rows = _series(200, 0).tolist()
    assert client.post("/detect", json={"channel": "NOPE", "rows": rows}).status_code == 404
    assert client.post("/detect", json={"channel": "../../etc", "rows": rows}).status_code == 422
    bad = [r + [0.0] for r in rows]
    assert client.post("/detect", json={"channel": "X-1", "rows": bad}).status_code == 422
    short = client.post("/detect", json={"channel": "X-1", "rows": rows[:10]})
    assert short.status_code == 422
    assert (
        client.post("/detect", json={"channel": "X-1", "model": "svm", "rows": rows}).status_code
        == 422
    )


def test_api_accepts_nulls_and_exposes_metrics(client):
    rows = _series(200, 0).tolist()
    rows[100][0] = None
    r = client.post("/detect", json={"channel": "X-1", "rows": rows})
    assert r.status_code == 200 and r.json()["n_replaced"] == 1
    text = client.get("/metrics").text
    assert "aai_requests_total 1" in text and "aai_errors_total 0" in text
