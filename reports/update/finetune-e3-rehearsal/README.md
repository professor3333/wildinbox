# Update candidate: `finetune-e3-rehearsal`

Protocol [`configs/experiments/update_rehearsal.yaml`](../../../configs/experiments/update_rehearsal.yaml) (sha256 `ee7609a52cad`): committed after the snapshot's reviews were made; not shown to predate training ([evidence](#protocol-provenance)). Snapshot `de9679c534af`: 1032 images from 359 reviewed events. **Reviews were simulated from the dataset's ground truth** (reviewer `simulated-ground-truth`).

## Decision

**Promote** `finetune-e3-rehearsal@5abc63b2fe2b` over `finetune-e3-deep-balanced@7a25aea97c76`.

| Check | Result | Requirement | Pass |
|---|---|---|---|
| Holdout event macro-F1 gain | +0.087 | >= +0.02 | yes |
| Macro-F1 change, calibration cameras | -0.008 | >= -0.02 | yes |
| Macro-F1 change, seen camera diagnostic | +0.023 | >= -0.02 | yes |
| Holdout animal events suggested as empty | 32 | <= 52.8 | yes |

## Holdout: later reviewed events on the updated cameras

|  | Deployed | Candidate |
|---|---|---|
| Events scored (supported label) | 413 | 413 |
| Event macro-F1 | 0.491 | 0.578 |
| Event macro-F1, camera cct-125 | 0.485 | 0.514 |
| Event macro-F1, camera cct-90 | 0.435 | 0.542 |
| Animal events suggested as empty | 48 / 422 | 32 / 422 |
| Temperature | 2.008 | 2.239 |

| Class | Deployed recall | Candidate recall | Holdout events |
|---|---|---|---|
| empty | 0.765 | 0.765 | 34 |
| bobcat | 0.593 | 0.333 | 81 |
| cat | 0.200 | 0.675 | 80 |
| coyote | 0.438 | 0.750 | 16 |
| dog | 0.656 | 0.688 | 32 |
| opossum | 0.656 | 0.648 | 128 |
| rabbit | 0.500 | 0.500 | 4 |
| raccoon | 0.816 | 0.789 | 38 |

## Regression checks: cameras neither model trained on

| Data | Deployed macro-F1 | Candidate macro-F1 |
|---|---|---|
| calibration cameras | 0.452 | 0.444 |
| seen camera diagnostic | 0.747 | 0.770 |

## Protocol provenance

| Evidence | Value |
|---|---|
| Protocol content | sha256 `ee7609a52cad` |
| First committed with this content | `6b410b2` at 2026-09-27T15:31:52+05:45 |
| Earliest review in the snapshot | 2026-09-25T10:19:19.806264+00:00 |
| Candidate trained at | `1c5de83`, with uncommitted changes |
| Committed before the reviews | **no** |
| Committed before training | not established |

Read from git history, the snapshot's review times, and the candidate's training record; nothing is claimed that they do not show. This is about when the rules were fixed, not a check of the metrics above.

## Development data only

Evaluated on the snapshot holdout, the calibration cameras, and the seen-camera diagnostic; the final test was not read. The snapshot holds no protected evaluation frame (checked before scoring). This was comparison #1 on this holdout; every comparison is logged in `../comparisons.jsonl`.

## What this shows

Holdout and snapshot share cameras and backgrounds by design: the gain is what reviewing a camera buys that camera's later photos, not evidence of generalization to new cameras; the regression checks cover cameras outside the update. The candidate keeps the deployed policy settings (automation off); only its calibration was refit.

Code `6b410b2084e0329280f13806807738f568e1b2b9`.
