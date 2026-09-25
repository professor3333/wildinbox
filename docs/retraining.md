# Retraining, candidate comparison, and rollback

The learning loop turns reviewers' corrections into a candidate model without
silently degrading the system. Every step leaves a record, and nothing is
released unless it passes a gate set up in advance.

```
reviews (API/UI) -> snapshot build -> train candidate -> update gate -> register + activate -> monitor
                     approved only     same recipe        development          GET /version      audits,
                     protected out     + snapshot         data only            check             behavior
                     provenance kept                      comparisons logged                     -> rollback
```

## 1. Protocol first

Write the cycle's protocol before collecting reviews or training, as
[`configs/experiments/update_cycle.yaml`](../configs/experiments/update_cycle.yaml)
was for cycle 1. It fixes:
- the deployed release;
- the training recipe (unchanged);
- the cameras;
- the split of reviewed events into training and holdout;
- the gate thresholds.

Optional fields for later cycles:

```yaml
name: update2                          # snapshot directory prefix
approval:
  reviewers: [ranger-a, ranger-b]      # whose reviews count; default: deployment.reviewer
protected:                             # default shown; always applied
  splits: data/splits/cct20-splits-v1/images.jsonl.gz
  partitions: [final_test, calibration, seen_camera_diagnostic]
  snapshot_holdouts:                   # earlier cycles' holdouts stay evaluation data
    - data/snapshots/update1-0837a1422904/holdout.jsonl
gate:
  max_comparisons_per_holdout: 5       # after this, a fresh holdout is needed
```

## 2. Snapshot from approved labels: `wildinbox snapshot build`

```bash
uv run wildinbox snapshot build --protocol configs/experiments/update_cycle.yaml
```

It reads through the API, as an offline training machine would.

- **Approved labels only.** An event counts only when its *latest* review is by
  an approved reviewer. Automatic labels, unreviewed suggestions, and reviews by
  anyone else never do (the count of skipped events is recorded).
- **Protected evaluation records are excluded, including corrections made to
  them.** An event is dropped, with every review on it, when any frame belongs
  to a protected partition or the event was in an earlier snapshot's holdout.
  Frames are matched by SHA-256, or by source file name for re-encoded
  copies. Protected partitions are the final test, the calibration cameras
  (temperature fit and regression check), and the seen-camera diagnostic
  (regression check). Exclusion happens before the time split, so a protected
  event cannot even move a cutoff. `snapshot.json` lists every excluded event
  with its reason and review ids.
- **Split by time per camera**: events before the camera's median reviewed
  start time go to `train.jsonl` (only supported labels are fit); later events
  go to `holdout.jsonl`, which is never trained on.
- **Provenance**: `labels.jsonl` has one row per considered event:
  - the label, review id, reviewer, time, and outcome;
  - the suggestion, and the release and policy that made it;
  - whether it was an audit sample;
  - the full review chain and the frame hashes;
  - its use: `train`, `holdout`, `not_fit`, or `excluded:<reason>`.

  `snapshot.json` records the file's SHA-256.
- The **version** is a hash of the train and holdout manifests: the same
  approved reviews give the same version.

## 3. Train the candidate

The deployed recipe is kept unchanged (a test enforces it), with the snapshot
added:

```bash
uv run wildinbox finetune train --config configs/experiments/finetune-e3-update1.yaml
```

Training runs offline on a separate machine, never in CI or on the serving VM.

## 4. Compare with the deployed model: `wildinbox update gate`

```bash
uv run wildinbox update gate --candidate models/finetune-e3-update1
```

The candidate's temperature is refit on the calibration cameras. The deployed
and candidate models are then compared on **development data only**:

| Check | Data | Rule (cycle 1) |
|---|---|---|
| Improvement | snapshot holdout (later reviewed events) | event macro-F1 gain ≥ +0.02 |
| No regression | calibration cameras (neither model trained there) | macro-F1 drop ≤ 0.02 |
| No regression | seen-camera diagnostic | macro-F1 drop ≤ 0.02 |
| No lost animals | holdout animal events suggested as empty | ≤ 110% of the deployed model's |

- **The final test is never read.** It was opened once, for the release
  decision, and is spent. Repeated candidate comparisons use the development
  checks above.
- **Leakage guard**: the gate refuses to run if the snapshot's training or
  holdout frames include any final-test, calibration, or diagnostic frame,
  because the regression checks would then be meaningless.
- **Comparisons are counted.** Every gate run is appended to
  [`reports/update/comparisons.jsonl`](../reports/update/comparisons.jsonl),
  with the candidate, weights, holdout version, gain, and verdict. Trying many
  candidates against one holdout slowly fits it. The count is reported, and a
  protocol's `max_comparisons_per_holdout` fails the gate once it is exceeded.
  After that, collect a fresh holdout (the next cycle's later reviews).

The gate writes `reports/update/<candidate>/README.md` and the candidate's
policy artifact.

## 5. Release

```bash
docker compose exec api wildinbox release register --model-dir models/<candidate> \
  --policy models/<candidate>/policy.json --activate --note "promoted: passed the update gate"
curl localhost:8000/version     # expected release, weights SHA-256, preprocessing, calibration, policy
```

Releases are immutable, and activation is append-only (`GET /releases`).
Running jobs keep the release they started with.

## 6. Monitor, then keep or roll back

After a release, watch the **Model behavior** view (see
[monitoring.md](monitoring.md)):
- the latest batch per camera against earlier batches (flagged "release also
  changed");
- the audit queue: audited false empties and species errors are the only
  unbiased measure of automatic decisions.

Roll back with one command:

```bash
docker compose exec api wildinbox release activate <previous-release-id> --note "rollback: <reason>"
```

New batches use the previous release immediately. Stored results are never
rewritten: each decision names the release that made it. Reverting restores
the previous release's predictions exactly, as
`scripts/rollback_restores.py` checks: the same photos processed before a
release switch and after a rollback get identical probabilities, labels, and
decisions ([report](../reports/update/rollback-restore.json)).

## Evidence

- Cycle 1, from corrections to candidate, gate, release, and rollback:
  [reports/update/README.md](../reports/update/README.md).
- Rebuilding cycle 1's snapshot with the current command (approval,
  protection, provenance), the gate rerun with the new controls, and the
  rollback check: the same report, section "Stage 11".
