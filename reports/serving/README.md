# Serving: reliability and processing cost

Release `finetune-e3-deep-balanced@518a8da39ee0` (E3, calibrated, policy
`conservative/v1+fe20c7586555`, automation off) on the Docker Compose
deployment. Batch: 1,000 images from 348 whole sequences on the development
cameras (`scripts/make_sample_batch.py`; the final test is never read) plus one
corrupt file.

**Declared hardware:** Apple M1, 8 GB RAM; Docker VM with 8 vCPUs and 3.9 GB;
CPU inference, one worker, chunks of 16 images.

## Worker killed halfway (acceptance gate)

`scripts/crash_demo.py` sends the worker container `SIGKILL` once about half the
images are scored, restarts it, and checks the outcome through the API and
directly in PostgreSQL. Log: [`crash-demo.log`](crash-demo.log); numbers:
[`crash-demo.json`](crash-demo.json).

| Check | Result |
|---|---|
| Killed at | 512 of 1,000 images scored (93 s after upload) |
| While the worker was dead | scoring stayed at 512; job still `running` (lease not yet expired); **0 events finalized** |
| Recovery | the restarted worker's sweep found the expired lease and requeued the job; attempt 2 resumed at image 513 |
| Lost inputs | none: 1,000 of 1,000 scored |
| Duplicate predictions | none: 1,000 rows for 1,000 distinct images |
| Events | 348 (one per sequence), exactly one decision each |
| Corrupt file | explicit error `unreadable image: OSError: Truncated File Read`; every other event completed |
| Job | `succeeded` on attempt 2, no error |

Total wall time was 262 s, of which about two minutes was the deliberate wait
for the dead worker's 120 s lease to expire.

**Found by this demo and fixed:** a restarted container kept its hostname and
PID 1, so the new worker asked RQ for the same name the killed one still held in
Redis and refused to start. Worker names now include a random part per process
start, and the worker service restarts automatically (`restart: unless-stopped`).

## Processing cost (1,000 images)

Clean run, no kill ([`timing-1000.json`](timing-1000.json)):

| Measure | Value | Target |
|---|---|---|
| Upload (1,001 files, 110 MB) | 5.6 s | |
| Scoring, grouping, and decisions | **78.5 s** (12.7 images/s) | 1,000 images within 10 minutes |
| End to end | 84 s | |

## Metadata API latency

200 requests each after warm-up, same machine, on the 1,000-image batch
([`api-latency.json`](api-latency.json)):

| Endpoint | p50 | p95 | Target |
|---|---|---|---|
| `GET /version` | 2.0 ms | 3.5 ms | p95 < 500 ms |
| `GET /events/{id}` | 7.2 ms | 11.2 ms | |
| `GET /batches/{id}` | 18.2 ms | 103.6 ms | |
| `GET /events` (100 per page) | 95.7 ms | 138.6 ms | |

## Reproduce

```bash
docker compose down -v && docker compose up -d --build --wait
docker compose run --rm \
  -v "$PWD/models/finetune-e3-deep-balanced:/app/models/finetune-e3-deep-balanced:ro" \
  -v "$PWD/reports/calibration:/app/reports/calibration:ro" \
  worker wildinbox release register --activate
uv run python scripts/make_sample_batch.py --images 1000
uv run python scripts/crash_demo.py --report reports/serving/crash-demo.json
```

The same batch can only be uploaded once per deployment (the API returns the
existing batch), so reset the stack between runs.

## Limitations

- One machine, one worker; throughput with several workers is not measured.
- The event list builds each row with its own queries; fine at this size,
  worth batching before much larger pages.
- The unfamiliar-input score is not part of the released policy (not adopted
  in Stage 8), so workers do not compute features.
