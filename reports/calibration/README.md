# Calibration and operating point: `finetune-e3-deep-balanced`

Rule: [`configs/experiments/operating_point.yaml`](../../configs/experiments/operating_point.yaml), committed before any calibrated policy-validation result was computed. Development partitions only; **the locked final test was not opened.**

## Decision

|  | Rule's threshold | Rule result | Released |
|---|---|---|---|
| Empty filter (every usable frame P(empty) >=) | 0.65 | enabled | **disabled** |
| Species acceptance (mean P(species) >=) | disabled | disabled | **disabled** |

**Deviation from the rule (2026-09-25):** automatic filtering released as disabled. The rule's empty filter (calibrated P(empty) >= 0.65) passed on the policy-validation cameras with a 95% upper bound of 1.99% against a 2% limit, but did not replicate on the other two unseen development cameras: it filters 40 of 724 animal events there (5.5%), 39 of 396 on camera 51 alone (9.8%). It would also remove only 38 of 939 events (4%) from review. Every event stays in review with its suggested label until a threshold holds up on new cameras.

A threshold is enabled only if the worst case of its 95% Wilson interval on policy validation meets the target: false-empty rate <= 2% of animal-containing events, accepted-label precision >= 95%. The lowest passing threshold on the grid is chosen; a disabled step sends those events to review.

At the rule's operating point on policy validation:

| Measure | Value |
|---|---|
| Animal events filtered as empty | 9 / 853 (1.06% [0.6, 2.0]) |
| Events filtered | 38 / 939 |
| Labels accepted automatically | 0 / 939 |
| Unsupported animals accepted as a known species | 0 |
| Events left for review | 901 (96.0%) |

## Replication on every development camera

The rule's thresholds applied to each unseen development camera. The calibration cameras were used to fit the temperature, not to choose thresholds, so they are an independent check of the operating point.

| Camera | Partition | Animal events filtered as empty (95% CI) | Filtered | Accepted |
|---|---|---|---|---|
| 51 | calibration | 39 / 396 (9.8% [7.3, 13.2]) | 112 / 522 | 0 |
| 108 | calibration | 1 / 328 (0.3% [0.1, 1.7]) | 6 / 386 | 0 |
| 90 | policy_validation | 5 / 297 (1.7% [0.7, 3.9]) | 20 / 350 | 0 |
| 125 | policy_validation | 4 / 556 (0.7% [0.3, 1.8]) | 18 / 589 | 0 |

## Calibration

Temperature scaling, fitted by NLL on the **calibration** partition only (2032 supported images): **T = 2.008**. T > 1 means the raw scores were overconfident. Temperature scaling never changes the predicted class, so macro-F1 and per-class recall are unchanged. Only the supported images enter NLL and ECE; unsupported animals count as not-empty in the P(empty) table.

| Partition | Role | NLL raw | NLL calibrated | ECE raw | ECE calibrated |
|---|---|---|---|---|---|
| calibration | fit | 1.713 | 1.440 | 0.210 | 0.034 |
| policy_validation | held out | 1.970 | 1.567 | 0.274 | 0.062 |

P(empty) reliability on policy validation (held out): how often images scored in each band are really empty. The empty filter relies on the top bands.

| P(empty) band | Images | Mean P(empty) raw | Empty (raw) | Images | Mean P(empty) calibrated | Empty (calibrated) |
|---|---|---|---|---|---|---|
| 0-0.5 | 1968 | 0.097 | 2.7% [2.1, 3.5] | 2407 | 0.169 | 3.9% [3.2, 4.8] |
| 0.5-0.7 | 253 | 0.604 | 9.5% [6.5, 13.7] | 252 | 0.575 | 17.5% [13.3, 22.6] |
| 0.7-0.8 | 143 | 0.756 | 11.9% [7.6, 18.2] | 43 | 0.736 | 41.9% [28.4, 56.7] |
| 0.8-0.9 | 208 | 0.846 | 13.0% [9.1, 18.2] | 19 | 0.842 | 57.9% [36.3, 76.9] |
| 0.9-0.95 | 86 | 0.926 | 18.6% [11.8, 28.1] | 6 | 0.925 | 66.7% [30.0, 90.3] |
| 0.95-0.99 | 50 | 0.969 | 46.0% [33.0, 59.6] | 5 | 0.959 | 100.0% [56.6, 100.0] |
| 0.99-1 | 24 | 0.996 | 70.8% [50.8, 85.1] | 0 | n/a | n/a |

## Review budget trade-off

Calibrated scores, policy validation, conservative policy. **Pass** means the threshold meets its target at the worst case of the 95% interval; operating points that fail are shown so no setting is implied to be safe when it is not.

Empty filter (species acceptance disabled):

| P(empty) >= | Filtered | True empty filtered | Animal events filtered (95% CI) | Review | Pass |
|---|---|---|---|---|---|
| 0.5 | 79 / 939 | 43 / 86 | 36 (4.22% [3.1, 5.8]) | 91.6% | fail |
| 0.6 | 50 / 939 | 37 / 86 | 13 (1.52% [0.9, 2.6]) | 94.7% | fail |
| 0.65 | 38 / 939 | 29 / 86 | 9 (1.06% [0.6, 2.0]) | 96.0% | **pass** |
| 0.7 | 33 / 939 | 26 / 86 | 7 (0.82% [0.4, 1.7]) | 96.5% | **pass** |
| 0.8 | 20 / 939 | 18 / 86 | 2 (0.23% [0.1, 0.9]) | 97.9% | **pass** |
| 0.85 | 14 / 939 | 13 / 86 | 1 (0.12% [0.0, 0.7]) | 98.5% | **pass** |
| 0.9 | 9 / 939 | 9 / 86 | 0 (0.00% [0.0, 0.4]) | 99.0% | **pass** |
| 0.95 | 5 / 939 | 5 / 86 | 0 (0.00% [0.0, 0.4]) | 99.5% | **pass** |
| 0.97 | 0 / 939 | 0 / 86 | 0 (0.00% [0.0, 0.4]) | 100.0% | **pass** |
| 0.98 | 0 / 939 | 0 / 86 | 0 (0.00% [0.0, 0.4]) | 100.0% | **pass** |
| 0.99 | 0 / 939 | 0 / 86 | 0 (0.00% [0.0, 0.4]) | 100.0% | **pass** |
| 0.995 | 0 / 939 | 0 / 86 | 0 (0.00% [0.0, 0.4]) | 100.0% | **pass** |
| 0.999 | 0 / 939 | 0 / 86 | 0 (0.00% [0.0, 0.4]) | 100.0% | **pass** |

Species acceptance (empty filter 0.65):

| Species >= | Accepted | Precision (95% CI) | Unsupported accepted as known | Review | Pass |
|---|---|---|---|---|---|
| 0.5 | 219 / 939 | 68.0% [61.6, 73.9] | 21 | 72.6% | fail |
| 0.6 | 150 / 939 | 74.7% [67.2, 81.0] | 10 | 80.0% | fail |
| 0.7 | 85 / 939 | 80.0% [70.3, 87.1] | 4 | 86.9% | fail |
| 0.8 | 35 / 939 | 80.0% [64.1, 90.0] | 1 | 92.2% | fail |
| 0.85 | 20 / 939 | 90.0% [69.9, 97.2] | 0 | 93.8% | fail |
| 0.9 | 7 / 939 | 100.0% [64.6, 100.0] | 0 | 95.2% | fail |
| 0.95 | 2 / 939 | 100.0% [34.2, 100.0] | 0 | 95.7% | fail |
| 0.97 | 1 / 939 | 100.0% [20.7, 100.0] | 0 | 95.8% | fail |
| 0.98 | 0 / 939 | n/a | 0 | 96.0% | fail |
| 0.99 | 0 / 939 | n/a | 0 | 96.0% | fail |
| 0.995 | 0 / 939 | n/a | 0 | 96.0% | fail |
| 0.999 | 0 / 939 | n/a | 0 | 96.0% | fail |

## Same thresholds on the calibration cameras (context only)

| Measure | Value |
|---|---|
| Animal events filtered as empty | 40 / 724 |
| Labels accepted (precision) | 0 (n/a) |
| Events left for review | 87.0% |

## Limitations

- Thresholds rest on 2 policy-validation cameras (90, 125); calibration on 2 others (51, 108). New cameras may behave differently; start them conservatively and audit.
- Uncalibrated sweeps on these development cameras were published in Stage 7 before this rule was written.
- Choosing the lowest passing threshold on the same data is mildly optimistic; the final test measures the frozen operating point once.
- The unfamiliar-input score is not part of this policy yet. When it is added, the operating point is re-chosen with this same rule.

## Reproduce

```bash
uv run wildinbox calibrate --rule configs/experiments/operating_point.yaml
```

Code `14f02bbc658d28cbc1565fb84f545dcaee00357d`; split `cct20-splits-v1-7de740aba75f`.
