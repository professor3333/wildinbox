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
export WILDINBOX_API_URL=https://wildinbox.example.org   # default http://localhost:8000
export WILDINBOX_TOKEN=...                               # a token of that deployment
uv run wildinbox snapshot build --protocol configs/experiments/update_cycle.yaml
```

It reads through the API, as an offline training machine would. Like the UI and
the scripts, it sends `Authorization: Bearer $WILDINBOX_TOKEN` when the variable
is set. If the API refuses a request, the command stops with exit status 1 and
names the request and status. For 401 or 403 it also says whether a token was
sent. It never reads an error response as data.

- **Approved labels only.** An event counts only when its *latest* review is by
  an approved reviewer. Automatic labels, unreviewed suggestions, and reviews by
  anyone else never do (the count of skipped events is recorded).
  Approval trusts the reviewer name, so it must be authenticated: with token
  access the API records every review under the caller's own principal, and only
  principals in `WILDINBOX_REVIEW_DELEGATES` may record one for someone else (both
  names are kept, `reviewer` and `recorded_by`). A deployment without
  authentication cannot prove who reviewed; build snapshots from one that has it.
- **Protected evaluation records are excluded, including corrections made to
  them.** An event is dropped, with every review on it, when any frame belongs
  to a protected partition, or the event or any of its frames was in an
  earlier snapshot's holdout. Frames are matched by SHA-256, and dataset frames
  also by source file name (to catch re-encoded copies). Earlier holdouts are
  matched by frame content as well as event id, so a re-upload that gets a new
  event id is still caught. Protected partitions are the final test, the calibration cameras
  (temperature fit and regression check), and the seen-camera diagnostic
  (regression check). Exclusion happens before the time split, so a protected
  event cannot even move a cutoff. `snapshot.json` lists every excluded event
  with its reason and review ids.
- **Split by time per camera**: events before the camera's median reviewed
  start time go to `train.jsonl` (only supported labels are fit); later events
  go to `holdout.jsonl`, which is never trained on.
- **Content separation after the split**, across all cameras, whole events at
  a time. The same bytes can be uploaded again under another name, camera, or
  time, and get a new event. Two rules apply:
  - A holdout event with a frame from the dataset's training partition is
    excluded (`holdout_frame_in_training_partition`), because both models have
    already seen it.
  - A training event that shares any frame with a holdout event is excluded
    (`train_frame_in_holdout`), together with its other frames. The holdout
    keeps its copy.

  This matches exact bytes only. A copy that has been re-encoded, or whose EXIF
  was edited, has a different SHA-256. It is not caught, except for dataset
  frames, which are also matched by source file name.
- **Provenance**: `labels.jsonl` has one row per considered event:
  - the label, review id, reviewer, time, and outcome;
  - the suggestion, and the release and policy that made it;
  - whether it was an audit sample;
  - the full review chain and the frame hashes;
  - its use: `train`, `holdout`, `not_fit`, or `excluded:<reason>`.

  `snapshot.json` records the file's SHA-256.
- The **version** is a hash of the train and holdout manifests: the same
  approved reviews give the same version.
- **Schema**: `snapshot.json` carries `schema: snapshot/v2` (fields include
  `approved_reviewers` and `provenance`). Snapshots built before the field
  existed are read as `snapshot/v1`, whose single `reviewer` becomes
  `approved_reviewers` and which has no provenance file.

## 3. Train the candidate

The deployed recipe is kept unchanged (a test enforces it), with the snapshot
added:

```bash
uv run wildinbox finetune train --config configs/experiments/finetune-e3-update1.yaml
```

Before training starts, it validates the snapshot. It refuses an unknown
schema, missing approved reviewers or train counts, a missing `train.jsonl`,
or a `labels.jsonl` whose SHA-256 differs from the summary. The model's
`meta.json` records `trained_on.snapshot` with the version, schema, approved
reviewers, and provenance SHA-256. The gate checks that the snapshot directory
still holds that version.

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
- **Leakage guard**: the gate re-derives content separation from the snapshot
  files, without relying on the builder's exclusions. It refuses to run if any
  of these holds:
  - a training or holdout frame is a final-test, calibration, or diagnostic
    frame, or is in an earlier protected holdout;
  - a training frame's bytes are also in the holdout;
  - an event is on both sides;
  - a holdout frame is in the dataset's training partition.
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
