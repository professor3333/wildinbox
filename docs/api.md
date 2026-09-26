# WildInbox API and processing

The API accepts uploads and serves results; image processing never happens
inside an HTTP request. Workers pick up jobs from a Redis queue, but
PostgreSQL is the source of truth for every job, image, prediction, event,
decision, and review. Interactive schema: `http://localhost:8000/docs`.

Errors are JSON: `{"error": "<code>", "detail": "<message>"}`.

## Access

With `WILDINBOX_AUTH=tokens` (the default, and always on staging), every
endpoint except `GET /health`, `GET /ready`, `/docs`, `/openapi.json`, and the
static upload page at `/` needs `Authorization: Bearer <token>`; without one
the answer is `401 {"error": "unauthorized"}` with `WWW-Authenticate: Bearer`.
Tokens are created with `wildinbox token new NAME`; the deployment stores only
their SHA-256 in `WILDINBOX_API_TOKENS` (a JSON object of name to hash), and
the name is recorded as `principal` in the request logs. There is one
workspace: every token holder sees every batch. `WILDINBOX_AUTH=disabled` is
for local development only (the local Compose stack sets it).

Every response carries `X-Request-ID` (the client's, or a generated one), the
key to the request's JSON log line.

## Endpoints

| Method and path | Behavior |
|---|---|
| `POST /batches` | Create a batch from multipart `files` (+ optional `metadata`). Returns `202` with the batch summary and its job, immediately. |
| `GET /batches/{id}` | Status, counts, **progress**, every **failed file with its error**, the job (attempts, lease, next retry), and the pinned release. |
| `GET /batches/{id}/images` | Every uploaded file with validation and processing status. |
| `GET /batches/{id}/export` | Current observations with provenance, one row per capture event. `?format=csv` (default) or `json`. |
| `GET /events` | Paginated, filterable events: `batch_id`, `camera_id`, `disposition`, `label`, `reason`, `reviewed`, `audit`, `start_after`, `start_before`, `limit` (≤ 500), `offset`. Returns `total` and `next_offset`. |
| `GET /events/{id}` | Frames with status (`completed`, `invalid`, `duplicate`, `failed`, `pending`), raw and calibrated predictions, the decision, and every review. An event keeps every file of its sequence; any frame that could not be scored sends it to review. |
| `POST /events/{id}/reviews` | Append a human review (`outcome`: confirmed / corrected / unresolved, `confirmed_label`, `note`). Reviews are never overwritten; each links to the one it supersedes. With token access the reviewer **is** the authenticated principal: `reviewer` may be omitted or repeat it; naming someone else is `403 reviewer_mismatch` unless the caller is in `WILDINBOX_REVIEW_DELEGATES` (an annotation importer, say), and then both are stored (`reviewer`, `recorded_by`). Without authentication (local development) `reviewer` is required and taken as given. Each event has exactly one review chain. If two reviews are submitted against the same history at the same moment, one is recorded and the other gets `409 review_conflict` (reload and retry), including the first review of an event. |
| `GET /jobs/{id}` | Job lifecycle only. |
| `GET /version` | The active release's id, weights SHA-256, preprocessing, calibration, and policy versions. The deploy check. |
| `GET /releases` | Every registered release and which one is active. |
| `GET /monitoring` | `operations` (queue, failures, job latency, `workers`, `api`, `storage`, `batches`), `behavior` (global, per camera, latest batch per camera, `periods`), `signals`, `audits`, `accuracy`, and `alerts`. Rules and fields: [monitoring.md](monitoring.md). |
| `GET /metrics` | The same in Prometheus text format, for scraping and alerting. |
| `GET /batches` | Recent batches. |
| `GET /images/{id}/thumbnail` | A JPEG thumbnail (`size` 64-1024), generated once and kept in object storage. |
| `GET /health` | Liveness: database reachable. Public. |
| `GET /ready` | Readiness: database, queue, object store, and the expected release (`WILDINBOX_EXPECTED_RELEASE`) active with its weights loaded and SHA-256-verified; `503` with the failing check otherwise. Public. |
| `GET /whoami` | The principal behind the token. |

### Uploading

```bash
curl -F files=@IMG_0001.JPG -F files=@IMG_0002.JPG \
     -F 'metadata={"camera_id": "north-trail"}' \
     -H 'Idempotency-Key: card-2024-05-02' \
     http://localhost:8000/batches
```

The `202` response carries `Server-Timing: receive;dur=…, validate;dur=…,
store;dur=…, db;dur=…` (milliseconds): body received and parsed, files
checked, originals written to object storage, rows committed. The rest of the
client's wait is network transfer.

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

Every column is defined in [`docs/export_format.md`](export_format.md).

## Releases

A release is immutable and bundles weights (stored by SHA-256), class names and
their fingerprint, preprocessing, calibration, and the decision policy with its
thresholds. A database trigger rejects any change to a release row.

```bash
# The published bundle (or --model-dir/--policy from a training machine):
deploy/fetch_release.sh
docker compose run --rm -v "$PWD/dist/release:/release:ro" worker \
  wildinbox release register --activate --note "E3, automation off" \
    --model-dir /release/model --policy /release/policy.json
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
