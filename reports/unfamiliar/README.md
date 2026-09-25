# Unfamiliar-input score: `finetune-e3-deep-balanced`

Rule: [`configs/experiments/unfamiliar.yaml`](../../configs/experiments/unfamiliar.yaml), committed before any score was computed. Development partitions only; **the locked final test was not opened.**

## Decision

**The distance score is not adopted.** On the calibration cameras, at a false-flag rate of at most 10% of supported-species animal images, it flags -1.5 percentage points of tuning-species images compared with ordinary confidence (adoption needs >= +5). The policy uses confidence only; the score is reported for reviewers' context.

## Results

Known = supported-species animal images; unknown = tuning species (squirrel, rodent), which the classifier was never fit on. Each threshold is fit on calibration and applied unchanged to policy_validation. AUROC = chance an unknown image scores higher than a known one.

| Partition | Score | Threshold | Known flagged (false flags) | Unknown flagged (detection) | AUROC |
|---|---|---|---|---|---|
| calibration (fit) | Distance to training images (kNN cosine) | 0.5143 | 155 / 1557 (10.0% [8.6, 11.5]) | 2 / 453 (0.4% [0.1, 1.6]) | 0.422 |
| calibration (fit) | Confidence (1 - max calibrated probability) | 0.7646 | 155 / 1557 (10.0% [8.6, 11.5]) | 9 / 453 (2.0% [1.0, 3.7]) | 0.424 |
| policy_validation (check) | Distance to training images (kNN cosine) | 0.5143 | 126 / 2127 (5.9% [5.0, 7.0]) | 0 / 297 (0.0% [0.0, 1.3]) | 0.402 |
| policy_validation (check) | Confidence (1 - max calibrated probability) | 0.7646 | 97 / 2127 (4.6% [3.8, 5.5]) | 1 / 297 (0.3% [0.1, 1.9]) | 0.486 |

Detection by tuning species:

| Partition | Score | squirrel | rodent |
|---|---|---|---|
| calibration | Distance to training images (kNN cosine) | 0 / 402 (0.0% [0.0, 0.9]) | 2 / 51 (3.9% [1.1, 13.2]) |
| calibration | Confidence (1 - max calibrated probability) | 5 / 402 (1.2% [0.5, 2.9]) | 4 / 51 (7.8% [3.1, 18.5]) |
| policy_validation | Distance to training images (kNN cosine) | 0 / 297 (0.0% [0.0, 1.3]) | n/a |
| policy_validation | Confidence (1 - max calibrated probability) | 1 / 297 (0.3% [0.1, 1.9]) | n/a |

## Species and pretraining exposure

| Species | Role | Unseen in fine-tuning | In ImageNet-1k pretraining classes |
|---|---|---|---|
| squirrel | tuning | yes | yes (fox squirrel) |
| rodent | tuning | yes | related classes (hamster, marmot, beaver, porcupine) |
| skunk | held out | yes | yes |
| bird | held out | yes | yes (many bird classes) |
| badger | held out | yes | yes |
| fox | held out | yes | yes (red, kit, Arctic, grey fox) |
| deer | held out | yes | no (no development or final-test events) |

Every unfamiliar species that can be evaluated was absent from fine-tuning but present in the pretraining class list, so these results say nothing about species the backbone has never seen. Held-out species are excluded from every number here and are evaluated once, on the final test.

## Artifact

`knn_cosine`, k = 10, threshold 0.5143, reference = 20287 training images (ids digest `feb80b9a059fce5f`), weights `3ab6fec236ce`; version `66c288910007`.

## Limitations

- Tuning species on the policy_validation cameras are almost all squirrels; rodents appear only on the calibration cameras.
- A 10% false-flag budget sends that share of real animals to review as possibly unknown; the operating-point rule measures the combined effect.

## Reproduce

```bash
uv run wildinbox unfamiliar --rule configs/experiments/unfamiliar.yaml
```

Code `88efc6bf2821880eaa1fe0cf459e67618097758f`; split `cct20-splits-v1-7de740aba75f`.
