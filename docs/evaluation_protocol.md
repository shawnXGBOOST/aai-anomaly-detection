# Evaluation protocol

This document fixes the evaluation rules **before** any model is run. It is implemented in
`src/aai/evaluation/` and covered by `tests/test_evaluation.py`.

## What is evaluated

For every scored channel (see `docs/data_report.md`) a detector produces, for each step of the
**test** series, a raw anomaly score and a final on/off alarm. Ground truth is the set of labeled
intervals of that test series (both ends inclusive).

## Data splits and tuning

* Models are fit on the train series only. Test data is never used for fitting, scaling,
  feature statistics, hyperparameter choice or threshold choice.
* Anything that needs tuning (thresholds, hyperparameters, adaptation settings) is tuned on a
  temporal hold-out: the **last part of the train series**, never a random split.
* The test series is scored once per final configuration, and every model is reported,
  including baselines that win.

## Event matching

* A **predicted event** is a run of consecutive alarm points.
* A **true interval** is *detected* if at least one alarm point lies inside it. A predicted event
  that starts earlier and runs into the interval counts. One that ends the step before it starts
  does not (adjacency is not overlap).
* A predicted event is a **hit** if it overlaps any true interval, otherwise a **false alarm**.
* **Event recall** = detected true intervals / true intervals.
* **Event precision** = hit predicted events / predicted events.
* **Event F1** = harmonic mean of the two. A detector that predicts nothing has precision 0 by
  convention, so its F1 is 0.
* **Detection delay** = steps from the interval start to the first alarm inside it (0 if the
  alarm was already on). Reported as a median in steps and as a fraction of the interval length.

## Point-level metrics

Precision, recall and F1 on individual test points. **No point-adjust** (counting a whole
interval as detected after one correct point), because it is known to inflate scores.

* **Normal time flagged** = alarm points outside any true interval / all points outside any true
  interval.
* **False alarms per 1,000 normal points** = false-alarm events normalised by normal time.

## Threshold-free metric

**PR-AUC** is the average precision of the raw scores against the point labels, computed per
channel and averaged over channels. It is shown next to the mean anomaly rate, which is what a
random scorer achieves.

## Aggregation

Counts are pooled across channels (events summed, not ratios averaged), then reported for all
channels together and for SMAP and MSL separately. Scores of different channels are on different
scales and are never compared across channels.

## Known weakness, and how it is handled

Event-level precision and recall alone can be gamed: an alarm that is always on is a single
event that overlaps every true interval, so it scores event recall 1.0 and event precision 1.0.
For that reason:

* **normal time flagged** and the **detection delay** are always reported next to event F1;
* the results table includes **always-on** and **never-on** reference rows, so the failure mode
  is visible rather than hidden.

## Not covered here

Inference latency and memory are measured at model level, not in this module.
