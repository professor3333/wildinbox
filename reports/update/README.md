# Update cycle 1: corrections -> snapshot -> candidate -> gate -> release -> rollback

One complete model-update cycle on the running deployment, under the protocol
[`configs/experiments/update_cycle.yaml`](../../configs/experiments/update_cycle.yaml),
committed before any review was collected or candidate trained.

**Reviews were simulated** from Caltech Camera Traps ground truth (reviewer
`simulated-ground-truth`, with a note on every review): no person labeled
these photos. Everything else (uploads, processing, reviews through the API,
snapshot, training, gate, release, rollback) is the real system.

## 1. Collect corrections

`scripts/simulate_deployment.py` uploaded every image from cameras 90 and 125
(2,732 photos, one batch per camera) through the API; E3 processed them with
automation off. Each of the 939 events was then reviewed through the API:
**454 confirmed, 463 corrected, 22 unresolved** (mixed-species sequences).

## 2. Versioned training snapshot

`wildinbox snapshot build` read the reviewed events through the API and split
each camera at its median reviewed event time (camera 90: 2011-11-16, camera
125: 2012-01-10). Snapshot `0837a1422904`
([summary](snapshot-0837a1422904.json)):

- **Training additions:** 1,066 frames from 371 earlier events with a supported
  label (bobcat 493, opossum 162, cat 117, empty 111, coyote 69, raccoon 69,
  rabbit 30, dog 15). Unsupported and unresolved events are not fit.
- **Holdout:** 471 later events (423 with a supported label), never trained on.
- Originals are fetched through the API and checked against their SHA-256.

## 3. Candidate

`finetune-e3-update1`: the deployed E3 recipe unchanged (a test enforces it)
on the training partition plus the snapshot, 21,353 images; 3 epochs, 28.5
minutes on the M1. Temperature refit on cameras 51 and 108 (never trained on):
2.061. Deployed policy settings kept (automation off).

## 4. Gate

[Full report](finetune-e3-update1/README.md). **Promote**: every pre-registered
check passed.

| Check | Deployed (E3) | Candidate | Requirement |
|---|---|---|---|
| Event macro-F1, later events on cameras 90 and 125 | 0.486 | **0.548** (+0.062) | gain >= +0.02 |
| Macro-F1, cameras 51 and 108 (neither trained there) | 0.452 | 0.491 | drop <= 0.02 |
| Macro-F1, held-out sequences of training cameras | 0.747 | 0.752 | drop <= 0.02 |
| Later animal events suggested as empty | 50 / 435 | 41 / 435 | <= 55 |

Biggest changes on the holdout: cat recall 0.20 -> 0.55, bobcat 0.60 -> 0.76,
coyote 0.44 -> 0.56; dog 0.66 -> 0.50, raccoon 0.80 -> 0.72, empty 0.77 ->
0.71 (rabbit: 4 events, too few to read).

The holdout shares cameras and backgrounds with the new training data by
design: the gain is what reviewing a camera buys that camera's later photos,
not evidence of generalization to new cameras. The final test is spent and
was not used.

## 5. Release and rollback

`scripts/release_rollback.py` on the live deployment ([log](release-log.json)):

| Step | Result |
|---|---|
| Register and activate `finetune-e3-update1@449138a9c367` | weights sha256 `edd6c897…`, preprocessing `6d9a950a6543`, policy `conservative/v1+fe20c7586555` |
| Batch A (29 new photos from cameras 51 and 108) | processed by the candidate |
| Roll back: re-activate `finetune-e3-deep-balanced@518a8da39ee0` | `GET /version` reports E3 |
| Batch B (33 new photos) | processed by E3 |
| Provenance | batch A's 11 decisions still name the candidate |

Activation history (`GET /releases`, append-only): E3 (initial) -> candidate
("promoted: passed the update gate") -> E3 ("rollback demonstration").
**The deployment is currently on E3** because the cycle ends with the rollback;
re-activating the candidate is one command:
`docker compose exec api wildinbox release activate finetune-e3-update1@449138a9c367`.

## Stage 11: the learning loop under the current controls

The cycle's tooling now enforces approval, protection, provenance,
development-only comparison, and a checked rollback
([workflow](../../docs/retraining.md)). Each was run against this cycle.

**Snapshot rebuilt with the current command.** The deployment database had
been reset since cycle 1, so `scripts/simulate_deployment.py` uploaded and
reviewed cameras 90 and 125 again (913 events: 444 confirmed, 448 corrected,
21 unresolved). `wildinbox snapshot build` then produced
[`stage11/snapshot-rebuild.json`](stage11/snapshot-rebuild.json):

- **Approved reviewers only**: `simulated-ground-truth`. One reviewed event on
  those cameras (a `demo-reviewer` correction from the demo script) was
  skipped as not approved.
- **Protected records**: 30,229 frames of the final test (23,275), calibration
  (2,641), and seen-camera diagnostic (4,313) were checked by SHA-256 and file
  name. None were among these cameras' events, as the split design requires,
  so none were excluded.
- **Provenance**: `labels.jsonl` records the label, review, reviewer,
  suggestion, suggesting release, review chain, and use of all 913 events.
- **Same data as cycle 1.** All 1,032 training frames and 458 holdout events
  match snapshot `0837a1422904` exactly (frames and labels). The rebuild has
  34 fewer training frames and 13 fewer holdout events. 67 of the photos
  involved were already in the deployment from earlier batches that day, and
  duplicate detection skipped them. Two further unsupported-species (skunk)
  events moved before camera 125's median cutoff, which shifted as a result,
  and unsupported labels are not fit. On a fresh deployment the rebuild
  would match. The candidate stays the one trained on `0837a1422904`.

**Gate rerun with the new controls.** `wildinbox update gate` reproduced
every number (holdout gain +0.062, both regression checks, 41 false-empty
suggestions; candidate release `finetune-e3-update1@449138a9c367`). It now
also:
- verifies before scoring that the snapshot holds no protected frame;
- records that only development data was read;
- logs itself in [`comparisons.jsonl`](comparisons.jsonl) as comparison #2 on
  holdout `0837a1422904`. The original run is entry #1, recorded
  retroactively because it predates the log.

**Rollback restores predictions** (`scripts/rollback_restores.py`,
[report](rollback-restore.json)). The same 62 photos (cameras 51 and 108) were
processed three times:
1. on E3;
2. on the update candidate;
3. on E3 again after rolling back.

| | Batch 3 (after rollback) vs batch 1 | Batch 2 (candidate) vs batch 1 |
|---|---|---|
| Max calibrated-probability difference | **0** | 0.479 |
| Frame labels | **62 / 62 identical** | 17 / 62 changed |
| Event decisions | **22 / 22 identical** | 22 / 22 name the candidate |

The stored results of batches 1 and 2 were unchanged after both switches.
The activation history records the candidate and the rollback as new rows.

## Reproduce

```bash
uv run python scripts/make_sample_batch.py --partition policy_validation --images 100000 \
  --out data/samples/deploy-cams-90-125
uv run python scripts/simulate_deployment.py --batch-dir data/samples/deploy-cams-90-125
uv run wildinbox snapshot build
uv run wildinbox finetune train --config configs/experiments/finetune-e3-update1.yaml
uv run wildinbox update gate
uv run python scripts/make_sample_batch.py --partition calibration --images 60 \
  --seed release-demo --out data/samples/release-demo
uv run python scripts/release_rollback.py
```

The snapshot version depends on the reviews; a fresh run gets a new version,
and `configs/experiments/finetune-e3-update1.yaml` must point at it.
