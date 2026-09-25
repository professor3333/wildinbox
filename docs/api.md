# WildInbox API and processing

The API accepts uploads and serves results; image processing never happens
inside an HTTP request. Workers pick up jobs from a Redis queue, but
PostgreSQL is the source of truth for every job, image, prediction, event,
decision, and review. Interactive schema: `http://localhost:8000/docs`.

Errors are JSON: `{"error": "<code>", "detail": "<message>"}`.

## Endpoints

| Method and path | Behavior |
|---|---|
| `POST /batches` | Create a batch from multipart `files` (+ optional `metadata`). Returns `202` with the batch summary and its job, immediately. |
| `GET /batches/{id}` | Status, counts, **progress**, every **failed file with its error**, the job (attempts, lease, next retry), and the pinned release. |
| `GET /batches/{id}/images` | Every uploaded file with validation and processing status. |
| `GET /batches/{id}/export` | Current observations with provenance, one row per capture event. `?format=csv` (default) or `json`. |
| `GET /events` | Paginated, filterable events: `batch_id`, `camera_id`, `disposition`, `label`, `reason`, `reviewed`, `limit` (≤ 500), `offset`. Returns `total` and `next_offset`. |
| `GET /events/{id}` | Frames with status, raw and calibrated predictions, the decision, and every review. |
| `POST /events/{id}/reviews` | Append a human review (`reviewer`, `outcome`: confirmed / corrected / unresolved, `confirmed_label`, `note`). Reviews are never overwritten; each links to the one it supersedes. |
| `GET /jobs/{id}` | Job lifecycle only. |
| `GET /version` | The active release's id, weights SHA-256, preprocessing, calibration, and policy versions. The deploy check. |
| `GET /releases` | Every registered release and which one is active. |
| `GET /monitoring` | Operations, label-free signals per camera, review-based accuracy, and alerts (thresholds: `configs/monitoring/monitoring.yaml`). |
| `GET /metrics` | The same in Prometheus text format, for scraping and alerting. |
| `GET /batches` | Recent batches. |
| `GET /images/{id}/thumbnail` | A JPEG thumbnail (`size` 64-1024), generated once and kept in object storage. |
| `GET /health` | Database reachable. |

### Uploading

```bash
curl -F files=@IMG_0001.JPG -F files=@IMG_0002.JPG \
     -F 'metadata={"camera_id": "north-trail"}' \
     -H 'Idempotency-Key: card-2024-05-02' \
     http://localhost:8000/batches
```

`metadata` may also give per-file values:
`{"files": {"IMG_0001.JPG": {"camera_id": "...", "captured_at": "...", "sequence_id": "..."}}}`.
Repeating a request (same `Idempotency-Key`, or identical content without
one) returns the existing batch with `"duplicate_request": true`; reusing a key
for different content is `409`. Files are checked by content, not extension:
unsupported, empty, oversized, and undecodable files get an explicit error
and never reach the model; the rest of the batch continues.

### Batch summary (abridged)

```json
{
  "id": "…", "status": "processing",
  "counts": {"images": 1001, "pending": 0, "valid": 610, "invalid": 1, "duplicate": 0,
             "events": 0, "processing_failed": 0},
  "progress": {"images_to_score": 1000, "images_scored": 610, "images_failed": 0,
               "finished": false},
  "failures": [{"filename": "corrupt.jpg", "stage": "validation",
                "error": "unreadable image: UnidentifiedImageError: …"}],
  "job": {"status": "running", "attempts": 1, "max_attempts": 3, "worker_id": "…",
          "heartbeat_at": "…", "next_attempt_at": null, "error": null},
  "release": {"id": "finetune-e3-deep-balanced@518a8da39ee0", "weights_sha256": "…",
              "preprocessing_version": "6d9a950a6543", "calibration_version": "55cdb7daad08",
              "policy_version": "conservative/v1+fe20c7586555"}
}
```

`events` stays `0` until the batch completes: events and their decisions are
written together at the end, never from a partial batch.

### Export columns

`event_id, batch_id, camera_id, start_at, end_at, frames, filenames,
observation, label_source, review_outcome, reviewer, reviewed_at, review_id,
disposition, suggested_label, confidence, reasons, model_release_id,
weights_sha256, preprocessing_version, calibration_version, policy_version,
exported_at`.

`observation` is the latest review's label; with no review, the automatic
label only if automation decided the event (`label_source = automatic`),
otherwise empty with `label_source = pending_review` (or `unresolved`). Rows
are capture events, not individual animals or population counts.

## Releases

A release is immutable and bundles weights (stored by SHA-256), class names and
their fingerprint, preprocessing, calibration, and the decision policy with its
thresholds. A database trigger rejects any change to a release row.

```bash
# Inside the deployment (weights and policy from the training machine):
docker compose run --rm \
  -v "$PWD/models/finetune-e3-deep-balanced:/app/models/finetune-e3-deep-balanced:ro" \
  -v "$PWD/reports/policy:/app/reports/policy:ro" \
  worker wildinbox release register --activate --note "E3, automation off" \
    --policy reports/policy/finetune-e3-deep-balanced-v2/policy.json
docker compose exec api wildinbox release list
docker compose exec api wildinbox release activate <release-id>   # also how to roll back
curl http://localhost:8000/version
```

Registering checks that the policy artifact belongs to these weights (its
calibration records the weights digest), that classes and preprocessing match
their recorded versions, and refuses a different release under an existing id.
Activation only affects **new** batches: each job is pinned to the release it
was created with, and retries use it even after another release becomes active.

## Processing and recovery

```
queued ──claim──> running ──all images scored, events decided──> succeeded
   ^                 │
   │   error, attempts left (backoff: 30 s, 60 s, … ≤ 15 min)
   └─────────────────┤
                     └── error on the last attempt, or lease expired on the last attempt ──> failed
running ──worker stops renewing its lease (lease_seconds)──> queued (recovered)
```

- **Lease.** A worker claims a job with a token and a heartbeat. Images are
  scored in chunks of `inference_chunk`; every chunk commit renews the
  heartbeat in the same transaction and only succeeds while the worker still
  holds the lease, so a worker presumed dead cannot write.
- **Resume.** Committed chunks are kept. A resumed job skips images that
  already have a prediction for its release; predictions are unique per
  image and release, events per batch and group, decisions per event,
  release, and policy version.
- **Recovery.** Each worker runs a sweep on start and every
  `recovery_interval_seconds`: jobs whose heartbeat is older than
  `lease_seconds` go back to `queued` (or `failed` if out of attempts), and
  every due queued job is dispatched. `wildinbox jobs recover` runs it once.
- **Timeouts.** A job attempt is stopped after `job_timeout_seconds` and counts
  as a failed attempt.
- **Per-file failures.** A file that cannot be decoded is `invalid` with its
  error. A valid image whose inference fails is a **failed frame**: its event
  goes to review with `processing_failure` and is never filtered.
- **Models load once.** Workers run jobs without forking, so each release's
  weights are loaded (and their SHA-256 verified) once per process.

Settings (environment, prefix `WILDINBOX_`): `INFERENCE_DEVICE` (cpu),
`INFERENCE_CHUNK` (16), `LEASE_SECONDS` (120), `JOB_TIMEOUT_SECONDS` (3600),
`RETRY_BACKOFF_SECONDS` (30), `RETRY_BACKOFF_MAX_SECONDS` (900),
`RECOVERY_INTERVAL_SECONDS` (30). Jobs get `max_attempts = 3`.

Crash demonstration against the running deployment:
`uv run python scripts/crash_demo.py` (see `scripts/crash_demo.py`).
