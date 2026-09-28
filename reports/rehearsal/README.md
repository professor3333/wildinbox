# Local rehearsal: deployment to retraining, release, and rollback

On 2026-09-27 the whole operational loop was run once more, end to end, on the
code that became v1.5.5. It ran on the local Docker Compose stack on an Apple M1
laptop (8 GB; Docker VM with 8 vCPUs and 3.9 GB), not on the AWS staging VM,
which stayed stopped. Every step used the documented commands
([deployment](../../docs/deployment.md), [retraining](../../docs/retraining.md)).

**Reviews were simulated** from Caltech Camera Traps ground truth: this reuses
update cycle 1's 913 reviews of cameras 90 and 125 (reviewer
`simulated-ground-truth`, [cycle 1](../update/README.md)). No person labeled these
photos.

| Step | Command | Result | Record |
|---|---|---|---|
| Backup before touching the stack | `pg_dump` | complete dump, read back (15 tables) | kept outside the repository |
| Deploy code `1c5de83` | `docker compose build && docker compose up -d` | migration `3c1f7a9d2e60` applied; existing data intact (1,121 study trials, 916 reviews) | [deploy.log](deploy.log) |
| Deploy check | `GET /ready`, `GET /version` | all checks ok; release `finetune-e3-deep-balanced@7a25aea97c76`, weights `3ab6fec2…`, preprocessing `6d9a950a6543`, calibration `55cdb7daad08`, policy `conservative/v2+378312635a29`, as the deployment doc expects | [deploy.log](deploy.log) |
| 1,000 new photos | `scripts/loadtest.py --scenarios thousand` | 348 events, processed in 167 s; every image scored once, every event exactly one decision | [loadtest.json](loadtest.json) |
| Worker restart mid-batch | `scripts/loadtest.py --scenarios restart` | restarted at 400/1,000 scored; job recovered on attempt 2; nothing lost or duplicated | [loadtest.json](loadtest.json) |
| Review, correct, export | API | a wrong suggestion (dog, 17%) corrected to opossum; the 348-row export shows the correction, reviewer, release, and policy | [review-export.log](review-export.log) |
| Snapshot | `wildinbox snapshot build` | `de9679c534af`: 1,032 training frames, 458 holdout events; the same version as the Stage 11 rebuild, and byte-identical when rebuilt under the rehearsal protocol | [snapshot.log](snapshot.log) |
| Candidate | `wildinbox finetune train --config configs/experiments/finetune-e3-rehearsal.yaml` | 52 minutes on MPS | [train.log](train.log) |
| Gate, on CPU | `wildinbox update gate --protocol configs/experiments/update_rehearsal.yaml --device cpu` | **promote** `finetune-e3-rehearsal@5abc63b2fe2b`; 27 minutes | [gate report](../update/finetune-e3-rehearsal/README.md) |
| Release, roll back | `scripts/rollback_restores.py` | candidate registered and activated, then rolled back; rollback restores 62/62 frame labels and 22/22 decisions exactly (max probability difference 0) | [rollback-restore.json](rollback-restore.json) |
| Final check | `GET /ready`, `GET /version`, `GET /releases` | E3 active again; the activation history records the candidate and the rollback | [final-check.log](final-check.log) |

## Gate

| Check | Deployed E3 | Candidate | Requirement |
|---|---|---|---|
| Holdout event macro-F1 (later events, cameras 90 and 125) | 0.491 | 0.578 (+0.087) | gain ≥ +0.02 |
| Calibration cameras 51 and 108 | 0.452 | 0.444 | drop ≤ 0.02 |
| Seen-camera diagnostic | 0.747 | 0.770 | drop ≤ 0.02 |
| Holdout animal events suggested as empty | 48 | 32 | ≤ 52.8 |

Every check passed, but not every species improved: on the holdout, bobcat
recall fell from 0.593 to 0.333 while cat rose from 0.200 to 0.675 and coyote
from 0.438 to 0.750. The pre-registered checks are macro-F1 and false empties, so a
trade between two species passes them. The candidate was not left in service: as
in cycle 1, the rehearsal ends with the rollback.

Since then, the gate also applies per-species recall limits and minimum
evidence, with a third outcome, inconclusive
([policy](../../docs/retraining.md#promotion-policy-critical-species-and-minimum-evidence)).
Applied to these recorded results, it **rejects** this candidate: bobcat's drop of
0.259 over 81 events exceeds the 0.10 limit. Coyote and rabbit, with 16 and 4
events, are too sparse to judge. This was applied after the fact, to a result
already seen, so it shows what the policy catches, not an independent test of it.
The recorded gate report above is unchanged.

## What differs from cycle 1, and why

- **Protocol.** [`update_rehearsal.yaml`](../../configs/experiments/update_rehearsal.yaml)
  is cycle 1's protocol with one change: it names the release this deployment
  serves (E3 with the v2 policy artifact) instead of E3 with v1. It was written
  after the reviews existed (they are cycle 1's) and before the candidate was
  gated.
- **Candidate.** The same recipe and snapshot data as cycle 1's candidate, but
  trained on current code, which crops at the corrected aspect ratio (PR #47).
  It is a new experiment, not a reproduction of `finetune-e3-update1`.
- **Training record.** `models/finetune-e3-rehearsal/meta.json` says
  `dirty: true`: the rehearsal's config and a test edit were not yet committed when
  training started. The source code was `1c5de83` unchanged; both files are in
  the next commit (`6b410b2`), on which the gate ran clean.

## Findings

- **`GET /events` (100 per page) missed its latency target on this laptop:** p95
  1,156 ms against < 500 ms (it was 360 ms on the 2-vCPU AWS VM). The laptop ran
  the load test and the API on a 3.9 GB Docker VM; this is not the declared
  hardware, but the number is reported as measured.
- **The gate ran on CPU** for the first time (both models, 27 minutes), after
  PR #50 removed the hard-coded MPS device.
- **Snapshot building is reproducible** across every change since Stage 11: the
  same reviews give the same version, `de9679c534af`.
