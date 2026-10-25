# Why the first results were weak, and what fixing it did

All numbers are pooled over the 80 scored channels from `experiments/results*/summary.csv`.
`results` is the first run, `results_v2` and `results_v3` are the two follow-ups described below.

## What the first run showed

The best event F1 was 0.449 (robust-z baseline, fixed threshold), but it flagged 7.5% of normal
time although the threshold was calibrated for 0.5%. The learned models were worse
(Isolation Forest 12.9%, autoencoder 26.0%).

## Cause

False alarms were concentrated, not spread out:

* Robust z: 10 of 80 channels produced 71% of all false-alarm points; 30 channels flagged more
  than 5% of their normal time.
* Autoencoder: 8 channels were flagged during 100% of their normal time; 56 channels flagged more
  than 5%.
* Several of these channels never change in train: C-2, D-12, A-1 and R-1 have a single value in
  train, P-14 has a standard deviation of 0.0001. Their test signal does vary, and the label file
  marks most of that as normal.
* For these channels the calibration scores are all about zero, so the fixed threshold is about
  zero and any change raises an alarm. In the autoencoder, features that never vary in the fit
  data are divided by a tiny standard deviation, so any change gives an enormous error.

So the 0.995 quantile did not fail by chance: train carries no information about normal behaviour
on these channels, and the test series is not drawn from the same distribution.

## Fixes (chosen from train data and convention, not from test labels)

* v2: models ignore features that are constant in their fit data; the robust-z score needs to
  exceed 3.5 (the usual robust z-score cutoff) to alarm.
* v3: v2 plus no alarms at all on channels with a flat train signal (21 channels by the rule
  "std of the scaled train signal < 0.1").

## What changed (fixed threshold, event F1 / normal time flagged / event recall)

| Model | first run | v2 | v3 |
|---|---|---|---|
| robust_z | 0.449 / 7.5% / 0.69 | 0.452 / 3.3% / 0.66 | 0.364 / 2.7% / 0.42 |
| isolation_forest | 0.296 / 12.9% / 0.66 | 0.299 / 12.6% / 0.66 | 0.305 / 9.4% / 0.58 |
| autoencoder | 0.255 / 26.0% / 0.94 | 0.255 / 20.8% / 0.81 | 0.275 / 12.7% / 0.70 |

## Reading

* The fixes cut false-alarm time a lot (robust z 7.5% to 3.3%, autoencoder 26% to 12.7% in v3).
* Event F1 barely moved. v2 left it unchanged and v3 lost recall because some flat-train channels
  do contain real anomalies. Point-level F1 for robust z also fell (0.378 to 0.222 in v2).
* The autoencoder's PR-AUC fell from 0.436 to 0.354 in v2: part of its first-run ranking score
  came from features that exploded on flat channels.
* The fixes were not tuned further, because every extra round on this test set would make it a
  tuning set. The 3.5 cutoff and the flat-train rule were set once.
* The diagnosis is a finding about the benchmark and the method: a threshold calibrated on
  train cannot be trusted on channels whose train data does not show normal variation.
