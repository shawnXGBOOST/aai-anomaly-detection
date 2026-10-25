import numpy as np
import pytest

from aai.data.labels import intervals_to_mask
from aai.evaluation.events import mask_to_segments, overlaps
from aai.evaluation.metrics import (
    channels_frame,
    evaluate_channel,
    summarize,
    summary_table,
)

N = 60
TRUTH = intervals_to_mask(((10, 19), (40, 49)), N)  # two anomalies of 10 points each


def alarms_from(*segments: tuple[int, int], n: int = N) -> np.ndarray:
    return intervals_to_mask(tuple(segments), n)


def evaluate(alarms, scores=None, mask=TRUTH, spacecraft="SMAP", name="X"):
    return evaluate_channel(name, spacecraft, mask, alarms, scores)


# ---- events ---------------------------------------------------------------------------------


def test_mask_to_segments():
    mask = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=bool)
    assert mask_to_segments(mask) == [(1, 2), (4, 4), (7, 8)]
    assert mask_to_segments(np.zeros(5, dtype=bool)) == []
    assert mask_to_segments(np.ones(5, dtype=bool)) == [(0, 4)]


def test_touching_intervals_do_not_overlap():
    assert overlaps((5, 10), (10, 12))  # share point 10
    assert not overlaps((5, 9), (10, 12))  # adjacent only


# ---- event-level matching -------------------------------------------------------------------


def test_one_hit_one_miss_one_false_alarm():
    r = evaluate(alarms_from((12, 14), (30, 32)))
    assert (r.n_true_events, r.n_detected_events) == (2, 1)
    assert (r.n_pred_events, r.n_pred_events_hit) == (2, 1)
    assert r.delays == (2,)  # first alarm at 12, interval starts at 10
    assert r.relative_delays == (0.2,)


def test_alarm_that_starts_early_and_runs_into_the_interval_has_zero_delay():
    r = evaluate(alarms_from((5, 12)))
    assert r.n_detected_events == 1
    assert r.delays == (0,)


def test_alarm_ending_just_before_the_interval_is_a_false_alarm():
    r = evaluate(alarms_from((5, 9)))
    assert r.n_detected_events == 0
    assert r.n_pred_events_hit == 0


def test_one_long_alarm_covering_two_intervals_detects_both_but_is_one_event():
    r = evaluate(alarms_from((8, 51)))
    assert r.n_detected_events == 2
    assert r.n_pred_events == 1 and r.n_pred_events_hit == 1


def test_no_alarms():
    r = evaluate(np.zeros(N, dtype=bool))
    assert r.n_detected_events == 0 and r.n_pred_events == 0
    assert r.delays == ()
    s = summarize([r])
    assert s["event_recall"] == 0 and s["event_precision"] == 0 and s["event_f1"] == 0
    assert np.isnan(s["median_delay"])


# ---- point-level ----------------------------------------------------------------------------


def test_point_counts_add_up():
    r = evaluate(alarms_from((12, 14), (30, 32)))
    assert (r.point_tp, r.point_fp, r.point_fn, r.point_tn) == (3, 3, 17, 37)
    assert r.point_tp + r.point_fp + r.point_fn + r.point_tn == N


def test_always_on_detector_is_exposed_by_point_level_metrics():
    s = summarize([evaluate(np.ones(N, dtype=bool))])
    assert s["event_recall"] == 1.0 and s["event_precision"] == 1.0  # the event-level loophole
    assert s["normal_time_flagged"] == 1.0  # ...which this column exposes
    assert s["point_precision"] == pytest.approx(20 / 60)


def test_never_on_detector():
    s = summarize([evaluate(np.zeros(N, dtype=bool))])
    assert s["event_recall"] == 0 and s["normal_time_flagged"] == 0


# ---- scores ---------------------------------------------------------------------------------


def test_pr_auc_perfect_and_uninformative():
    alarms = alarms_from((12, 14))
    perfect = evaluate(alarms, scores=TRUTH.astype(float))
    assert perfect.average_precision == pytest.approx(1.0)
    constant = evaluate(alarms, scores=np.zeros(N))
    assert constant.average_precision == pytest.approx(TRUTH.mean())  # the anomaly rate


def test_pr_auc_is_none_without_scores_or_with_one_class():
    assert evaluate(alarms_from((12, 14))).average_precision is None
    one_class = evaluate_channel(
        "X", "SMAP", np.zeros(N, dtype=bool), alarms_from((1, 2)), scores=np.zeros(N)
    )
    assert one_class.average_precision is None


# ---- aggregation ----------------------------------------------------------------------------


def test_summary_pools_counts_instead_of_averaging_ratios():
    a = evaluate(alarms_from((12, 14), (30, 32)), name="A")  # 1 of 2 detected, 1 of 2 hit
    b = evaluate(alarms_from((10, 19), (40, 49)), name="B", spacecraft="MSL")  # perfect
    s = summarize([a, b])
    assert (s["true_events"], s["detected_events"]) == (4, 3)
    assert (s["pred_events"], s["false_alarm_events"]) == (4, 1)
    assert s["event_recall"] == pytest.approx(3 / 4)
    assert s["event_precision"] == pytest.approx(3 / 4)
    assert s["median_delay"] == pytest.approx(0.0)  # delays are [2, 0, 0]


def test_summary_table_has_pooled_and_per_spacecraft_rows():
    a = evaluate(alarms_from((12, 14)), name="A")
    b = evaluate(alarms_from((10, 19)), name="B", spacecraft="MSL")
    table = summary_table([a, b])
    assert list(table.index) == ["ALL", "MSL", "SMAP"]
    assert table.loc["ALL", "channels"] == 2
    assert table.loc["SMAP", "channels"] == 1


def test_channels_frame_has_one_row_per_channel():
    frame = channels_frame([evaluate(alarms_from((12, 14)), name="A")])
    assert frame.loc[0, "channel"] == "A"
    assert frame.loc[0, "delays"] == [2]


# ---- input validation -----------------------------------------------------------------------


def test_rejects_wrong_dtype_shape_and_bad_scores():
    with pytest.raises(ValueError, match="boolean"):
        evaluate(np.zeros(N))
    with pytest.raises(ValueError, match="shapes"):
        evaluate(np.zeros(N - 1, dtype=bool))
    with pytest.raises(ValueError, match="NaN"):
        evaluate(alarms_from((12, 14)), scores=np.full(N, np.nan))
    with pytest.raises(ValueError, match="match"):
        evaluate(alarms_from((12, 14)), scores=np.zeros(N - 1))


def test_summarize_rejects_empty_input():
    with pytest.raises(ValueError):
        summarize([])
