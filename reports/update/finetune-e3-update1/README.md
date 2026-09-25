# Update candidate: `finetune-e3-update1`

Protocol [`configs/experiments/update_cycle.yaml`](../../../configs/experiments/update_cycle.yaml), committed before any review was collected or candidate trained. Snapshot `0837a1422904`: 1066 images from 371 reviewed events. **Reviews were simulated from the dataset's ground truth** (reviewer `simulated-ground-truth`).

## Decision

**Promote** `finetune-e3-update1@449138a9c367` over `finetune-e3-deep-balanced@518a8da39ee0`.

| Check | Result | Requirement | Pass |
|---|---|---|---|
| Holdout event macro-F1 gain | +0.062 | >= +0.02 | yes |
| Macro-F1 change, calibration cameras | +0.038 | >= -0.02 | yes |
| Macro-F1 change, seen camera diagnostic | +0.005 | >= -0.02 | yes |
| Holdout animal events suggested as empty | 41 | <= 55.0 | yes |

## Holdout: later reviewed events on the updated cameras

|  | Deployed | Candidate |
|---|---|---|
| Events scored (supported label) | 423 | 423 |
| Event macro-F1 | 0.486 | 0.548 |
| Event macro-F1, camera cct-125 | 0.479 | 0.500 |
| Event macro-F1, camera cct-90 | 0.437 | 0.491 |
| Animal events suggested as empty | 50 / 435 | 41 / 435 |
| Temperature | 2.008 | 2.061 |

| Class | Deployed recall | Candidate recall | Holdout events |
|---|---|---|---|
| empty | 0.765 | 0.706 | 34 |
| bobcat | 0.595 | 0.762 | 84 |
| cat | 0.195 | 0.549 | 82 |
| coyote | 0.438 | 0.562 | 16 |
| dog | 0.656 | 0.500 | 32 |
| opossum | 0.652 | 0.621 | 132 |
| rabbit | 0.500 | 0.000 | 4 |
| raccoon | 0.795 | 0.718 | 39 |

## Regression checks: cameras neither model trained on

| Data | Deployed macro-F1 | Candidate macro-F1 |
|---|---|---|
| calibration cameras | 0.452 | 0.491 |
| seen camera diagnostic | 0.747 | 0.752 |

## What this shows

Holdout and snapshot share cameras and backgrounds by design: the gain is what reviewing a camera buys that camera's later photos, not evidence of generalization to new cameras; the regression checks cover cameras outside the update. The candidate keeps the deployed policy settings (automation off); only its calibration was refit.

Code `37305d7f275a0d46eb221e08fedf6c341c4a2df4`.
