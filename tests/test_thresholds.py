import numpy as np
import pytest

from aai.detection.thresholds import (
    AdaptiveConfig,
    _sorted_quantile,
    adaptive_threshold,
    fixed_threshold,
    hold_alarms,
)


def _data(seed=0):
    rng = np.random.default_rng(seed)
    return np.abs(rng.normal(1, 0.2, 2000)), np.abs(rng.normal(1, 0.2, 2000))


def test_fixed_threshold_is_quantile_and_validates():
    x = np.arange(101.0)
    assert fixed_threshold(x, 0.9) == pytest.approx(90.0)
    with pytest.raises(ValueError):
        fixed_threshold(x, 1.0)
    with pytest.raises(ValueError):
        fixed_threshold(np.array([1.0, np.nan]), 0.5)


def test_hold_alarms_is_causal_and_extends_forward():
    raw = np.zeros(12, dtype=bool)
    raw[3] = True
    out = hold_alarms(raw, 3)
    assert out.tolist() == [False] * 3 + [True] * 4 + [False] * 5
    assert hold_alarms(raw, 0).tolist() == raw.tolist()
    with pytest.raises(ValueError):
        hold_alarms(raw, -1)


@pytest.mark.parametrize("q", [0.0, 0.3, 0.995, 1.0])
def test_sorted_quantile_matches_numpy(q):
    v = np.sort(np.random.default_rng(1).normal(size=57))
    assert _sorted_quantile(v.tolist(), q) == pytest.approx(np.quantile(v, q))


@pytest.mark.parametrize("rule", ["robust", "quantile"])
@pytest.mark.parametrize("guard", ["none", "freeze"])
def test_adaptive_is_causal(rule, guard):
    h, s = _data()
    cfg = AdaptiveConfig(rule=rule, guard=guard)
    a1, t1 = adaptive_threshold(h, s, cfg)
    s2 = s.copy()
    s2[1500:] += 10
    a2, t2 = adaptive_threshold(h, s2, cfg)
    assert np.array_equal(a1[:1500], a2[:1500])
    assert np.allclose(t1[:1500], t2[:1500])


def test_freeze_blocks_corruption_from_short_spike_none_does_not():
    h, s = _data()
    s[500:600] = 8.0
    probe = np.full(300, 3.0)
    s = np.concatenate([s[:700], probe])
    frozen, _ = adaptive_threshold(h, s, AdaptiveConfig(guard="freeze", max_freeze=200))
    open_, _ = adaptive_threshold(h, s, AdaptiveConfig(guard="none"))
    assert frozen[500:600].all() and open_[500:600].all()
    assert frozen[700:].all()


def test_level_shift_recovers_without_freeze_and_with_max_freeze():
    h, s = _data()
    s = s.copy()
    s[1000:] += 1.5
    none, _ = adaptive_threshold(h, s, AdaptiveConfig(guard="none"))
    frz, _ = adaptive_threshold(h, s, AdaptiveConfig(guard="freeze", max_freeze=50))
    assert none[1800:].mean() < 0.05
    assert frz[1000:1040].all()


def test_rejects_bad_input():
    h, s = _data()
    with pytest.raises(ValueError):
        adaptive_threshold(np.array([]), s, AdaptiveConfig())
    with pytest.raises(ValueError):
        adaptive_threshold(h, np.append(s, np.nan), AdaptiveConfig())
    with pytest.raises(ValueError):
        adaptive_threshold(h, -s, AdaptiveConfig())
    with pytest.raises(ValueError):
        AdaptiveConfig(guard="x")
    with pytest.raises(ValueError):
        AdaptiveConfig(rule="x")
