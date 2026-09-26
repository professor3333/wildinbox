# Staging: deployment, load test, and failure scenarios

Stage 12. The release was deployed with [docs/deployment.md](../../docs/deployment.md) to one
AWS VM and measured there. Every number below comes from a JSON file in this directory.

## Summary against the targets

| Target | Measured on the declared hardware | Met |
|---|---|---|
| 1,000 resized images processed within 10 minutes | **111.4 s** (9.0 images/s), one worker | yes |
| Metadata API p95 < 500 ms | **6.9–359.9 ms** across five endpoints (client on the VM); worst is `GET /events`, 100 per page | yes |
| Readiness requires the expected model to load | wrong expected release → `/ready` 503, deploy wait fails, workers refuse to start | yes |
| Survives a worker restart mid-batch | 0 lost, 0 duplicate predictions, one decision per event; 138 s slower | yes |
| Survives invalid files | 6 of 6 bad files rejected with a reason; the 20 good ones processed | yes |
| Two concurrent batches | both complete; 2 workers take 162 s instead of 257 s, with no duplicated work | yes |
| Backup and restore | restore brings back exactly the backed-up state; later writes gone; stack processes again | yes |
| Another person can deploy the pinned release | a fresh stack deployed `v1.4.0` from the guide in 3 min 33 s and ran the demo ([rehearsal](rehearsal/README.md)) | yes |

Measured from a laptop in Nepal, metadata p95 is 0.7–1.0 s. The server's own time is
unchanged; the extra ~560 ms per request is the network round trip through the SSH
tunnel. See [Latency](#metadata-latency).

## Declared hardware and software

| | |
|---|---|
| Instance | AWS `m7i-flex.large`, us-east-1: 2 vCPU (Intel Xeon Platinum 8488C), 7.6 GiB RAM, 40 GB encrypted gp3 |
| Software | Ubuntu 24.04.5 LTS, kernel 7.0.0-1013-aws, Docker 29.1.3, Compose 2.40.3 |
| Stack | [`deploy/staging/compose.yml`](../../deploy/staging/compose.yml): API, 1 or 2 workers, UI, PostgreSQL 16, Redis 7; originals and weights in S3 |
| Application | image `wildinbox-staging` (0.97 GB) built from commit `298b087`; CPU inference, chunks of 16 images |
| Model release | `finetune-e3-deep-balanced@7a25aea97c76`: weights SHA-256 `3ab6fec2…9b360`, preprocessing `6d9a950a6543`, calibration `55cdb7daad08`, policy `conservative/v2+378312635a29` |
| Images | 1,000 JPEGs from `data/samples/dev-1000` (CCT20 development cameras, resized by LILA), 109.8 MB total, 110 KB mean |

The instance type was chosen because the AWS account only allows free-tier-eligible types;
`c7i.xlarge` (4 vCPU) was refused. The template's `InstanceType` parameter takes any type.

## How it was measured

[`scripts/loadtest.py`](../../scripts/loadtest.py) runs each workload through the API with a
real token. Each upload contains files no earlier run has seen, so duplicate detection never
skips work. A JPEG comment segment is added: new SHA-256, identical pixels.

- **Upload** is `POST /batches` timed by the client. The API's `Server-Timing` header splits
  its side into *receive* (body arrived and parsed), *validate*, *store* (originals to S3),
  and *db*. Transfer is the client time minus the server's work after receiving.
- **Processing** is job created → job finished, from the job's server timestamps. *Queued* is
  job created → started.
- **Integrity** is counted in PostgreSQL, not taken from the API. Checks: predictions per
  image (duplicates), valid images without a prediction (lost), and exactly one decision
  per event.
- **Memory** is the peak `docker stats` usage per container, sampled every ~2 s.

The workloads ran with the client on the VM itself ([`workers-1.json`](workers-1.json),
[`workers-2.json`](workers-2.json)). That way the SSH link from the laptop does not affect
processing or recovery. Internet upload and client latency were measured separately from the
laptop ([`internet-thousand.json`](internet-thousand.json)).

## Workloads (one worker)

[`workers-1.json`](workers-1.json), log [`workers-1.log`](workers-1.log).

| Workload | Files | Upload (server part) | Queued | Processing | Result | Integrity |
|---|---|---|---|---|---|---|
| Small clean batch | 24 | 0.36 s (0.35 s) | 0.1 s | 2.5 s | completed, 23 events | ok |
| 1,000-image batch | 1,000 | 5.6 s (5.3 s: store 4.7 s) | 0.5 s | **111.4 s** | completed, 348 events | ok |
| Invalid files | 20 + 6 bad | 0.35 s | 0.1 s | 2.0 s | completed_with_errors | ok |
| Two concurrent batches | 2 × 1,000 | 8.2 s and 8.9 s | 0.5 s / 113.2 s | 114.2 s / 256.1 s | both completed | ok, ok |
| Worker restart at 400/1,000 | 1,000 | 5.5 s | — | 249.1 s (attempt 2) | completed | ok |

**Invalid files.** Each one got its own reason; the good files were processed:

| File | Recorded error |
|---|---|
| `corrupt.jpg` (JPEG header, then text) | `unreadable image: OSError: Truncated File Read` |
| `truncated.jpg` (a third of a real photo) | `unreadable image: OSError: image file is truncated` |
| `empty.jpg` | `empty_file: file is empty (0 bytes)` |
| `notes.txt` | `unsupported_type: ... only JPEG and PNG are accepted` |
| `animation.gif` | `unsupported_type: unsupported file type (GIF) ...` |
| `huge.jpg` (20 MiB + 1 KiB) | `file_too_large: file is larger than the 20 MiB per-file limit` |

`corrupt.jpg` and `truncated.jpg` passed the upload checks and failed when decoded during
processing. The other four were refused at upload.

**Worker restart.** `restart worker` was issued with 400 of 1,000 images scored. The events
are from the JSON logs.

1. Docker sent SIGTERM. The worker kept scoring through the 30 s stop grace period (to
   688 of 1,000) and was then killed mid-batch.
2. The new worker loaded the release in 0.5 s. The job stayed `running` under the dead
   worker's lease, and no events were finalized.
3. When the 120 s lease expired, the recovery sweep logged
   `worker ... stopped responding; recovering`.
4. Attempt 2 kept the 688 saved predictions, scored the remaining 312, grouped, decided, and
   finished in 49.9 s.

The result was 1,000 of 1,000 scored, 0 duplicate predictions, 348 events, one decision
each. The restart cost 138 s: the grace period plus the lease.

## Upload transfer, separately

| Client | Files | Total | Network transfer | Server: receive / validate / store / db |
|---|---|---|---|---|
| On the VM (localhost) | 1,000 (109.9 MB) | 5.6 s | 0.3 s | 0.26 / 0.10 / 4.72 / 0.45 s |
| Laptop in Nepal, over an SSH tunnel | 1,000 (109.9 MB) | 40.0 s | 34.4 s (25.5 Mbit/s) | 33.3 / 0.14 / 4.99 / 0.43 s |

From a home connection, upload time is set by the link. Processing does not start until the
upload has been stored. The server's own share is about 5.5 s per 1,000 files, mostly S3
writes.

## Metadata latency

200 requests per endpoint after warm-up, on the 1,000-image batch, with the client on the VM.
Server-side figures are the API's own request timings for the whole run
([`workers-1.json`](workers-1.json) `latency_server`).

| Endpoint | p50 | p95 | max | Server p95 | Target |
|---|---|---|---|---|---|
| `GET /version` | 3.3 ms | 6.9 ms | 40.1 ms | 4.7 ms | p95 < 500 ms |
| `GET /jobs/{id}` | 10.1 ms | 19.5 ms | 26.6 ms | 15.9 ms | |
| `GET /events/{id}` | 14.1 ms | 29.0 ms | 219.9 ms | 22.2 ms | |
| `GET /batches/{id}` | 41.5 ms | 255.3 ms | 295.8 ms | 283.5 ms (1,117 requests, during processing) | |
| `GET /events` (100 per page) | 244.4 ms | **359.9 ms** | 570.0 ms | 358.1 ms | |

From the laptop (100 requests each), p95 was 711.6 ms (`/version`) to 1,024 ms
(`/events`). `/version` takes 3 ms on the server and ~566 ms at the client's median, so the
difference is round-trip time over this link. The 500 ms target is met by the API on the
declared hardware, not by this network path.

`GET /events` builds each row with its own queries. It is the endpoint closest to the target
and the first to optimise (the `max` exceeds 500 ms).

## Adding a second worker

| Configuration | One 1,000-image batch | Two concurrent 1,000-image batches | CPU while processing |
|---|---|---|---|
| 1 worker | 111.4 s | 257.1 s (7.8 images/s) | 50% idle |
| 2 workers | 116.8 s | **162.3 s (12.3 images/s)** | — |
| 2 workers, 1 PyTorch thread each | — | 168.3 s (11.9 images/s) | 20% idle |

- **A single batch does not get faster.** A batch is one job, and a job is claimed under a
  lease by exactly one worker. This is what keeps retries idempotent. It also means a second
  worker cannot split one batch.
- **Concurrent batches do: 1.6× throughput.** Each worker took one batch. The PostgreSQL
  checks found 0 duplicate predictions and one decision per event, so no work was duplicated.
- **Why it helps on 2 vCPUs:** one worker leaves about half of the CPU idle (`vmstat`: 39%
  user, 11% system, 50% idle, 0% I/O wait; [`vmstat-cpu-workers-1.log`](vmstat-cpu-workers-1.log)).
  Scoring is serial per worker: fetch from S3, decode and preprocess, run the model on a
  chunk of 16, write to the database. The model's parallel section does not fill both cores.
  With two workers the CPU is 80% busy
  ([`vmstat-workers-2-threads-1.log`](vmstat-workers-2-threads-1.log)).
- **Thread oversubscription is not the limit:** one PyTorch thread per worker was no faster
  than the default.
- **The next bottleneck is CPU.** At 80% busy, a third worker on this machine would have
  little left to use. More throughput needs more cores or a larger instance.

Staging now runs two workers (`WILDINBOX_WORKERS=2`).

## Memory

Peak per container, over all workloads (MiB):

| API | Worker (each) | PostgreSQL | Redis | UI |
|---|---|---|---|---|
| 778 | 545 | 88 | 7 | 94 |

The API holds the model because readiness loads it. Two workers plus everything else stay
under 2.1 GiB of the 7.6 GiB.

## Readiness

[`readiness-drill.log`](readiness-drill.log): `WILDINBOX_EXPECTED_RELEASE` was set to a
release that is not active.

- `up --wait` exited 1: `dependency failed to start: container ... api-1 is unhealthy`.
- `/ready` returned HTTP 503: `active release is 'finetune-e3-deep-balanced@7a25aea97c76',
  deployment expects 'finetune-e3-deep-balanced@000000000000'`.
- Both workers exited and restarted in a loop: `release ... is not registered`.
- Setting it back: `up --wait` exited 0 and the API was ready.

A corrupted weights file behaves the same way: SHA-256 check failure, 503, and workers that
refuse to start (`tests/test_staging.py`).

## Backup and restore

[`backup-restore.log`](backup-restore.log), run with the committed
[`deploy/staging/restore_drill.sh`](../../deploy/staging/restore_drill.sh):

| Step | Result |
|---|---|
| Backup (`backup.sh`) | 9.2 MB custom-format dump, 15 tables, to `s3://…/backups/postgres/`, in 2.6 s |
| Before backup | 27 batches, 18,201 images, 6,437 events, 18,177 predictions, 6,437 decisions, 3 reviews |
| Batch uploaded after the backup | present before the restore |
| Restore (`restore.sh`) | 35.8 s, including stopping and starting the app |
| After restore | counts identical to the backup; after-backup batch 404; the drill's review present; restored images load from S3; ready; a new upload processed |

The nightly backup also runs under cron's minimal environment (`PATH=/usr/bin:/bin`); see
[Found and fixed](#found-and-fixed-by-staging).

## Upgrade to v1.5.3

On 2026-09-26 staging was upgraded from `v1.4.1` to `v1.5.3` with the documented
[upgrade procedure](../../docs/deployment.md#upgrade-and-rollback): check out the tag,
set `WILDINBOX_GIT_REF`, back up, run `deploy.sh` (no bundle, since the model is
unchanged). Log: [`upgrade-v1.5.3.log`](upgrade-v1.5.3.log).

| Check | Result |
|---|---|
| Before the upgrade | 28 batches, 18,202 images, 6,438 events, 18,178 predictions, 3 reviews, 2 releases; no event with more than one first review |
| Backup | `backups/postgres/wildinbox-20260926T055748Z.dump`, 9.2 MB, 15 tables |
| Migrations | `3f1c2a9d7e40` → `4803c5c84b95` (review `recorded_by`) → `9b2d4e71c0a5` (one first review per event); index present |
| Readiness | `ready`; database, queue, object store, and model checks ok; model `finetune-e3-deep-balanced@7a25aea97c76` loaded |
| Active release | unchanged: `finetune-e3-deep-balanced@7a25aea97c76`, policy `conservative/v2+378312635a29` |
| After the upgrade | counts identical to before |
| Monitoring (new error-rate definition, 7-day window) | 18,178 frames attempted, 18,178 scored, 0 failed; rate 0.0 |
| Unauthenticated `/version` | 401 (token required, as configured) |

Not yet done after this upgrade: the token-authenticated part of the
[deploy check](../../docs/deployment.md#deploy-check). That means `/version` with a token,
and an upload → process → review smoke test (`scripts/smoke.py`). The VM holds only token
hashes, so these need the owner's token.

## Cost assumptions

On-demand list prices, us-east-1, from the AWS Price List API on 2026-09-25. Taxes, data
transfer out (thumbnails and exports; transfer in is free), and free-plan credits are
excluded.

| Item | Price | Per month |
|---|---|---|
| VM `m7i-flex.large`, always on | $0.09576/hour | $69.90 (730 h) |
| EBS gp3, 40 GB | $0.08/GB-month | $3.20 |
| S3 Standard, originals | $0.023/GB-month; ~110 KB per image | $0.0025 per 1,000 images stored |
| S3 PUT requests | $0.005 per 1,000 | $0.005 per 1,000 images uploaded |
| Database backups | 9.2 MB × 30 days retained | < $0.01 |

**Per 1,000 images processed**, the marginal VM time is 111.4 s × $0.09576/h ≈ **$0.003** with
one worker, or ≈ $0.002 at the two-worker concurrent rate. The VM costs the same whether it
is busy or not. At this scale the cost is the always-on instance (about $73/month in total).
Stopping it outside review sessions is the main saving; the EBS volume and S3 are still billed.

## Found and fixed by staging

The first load test ([`run-1-before-fixes/`](run-1-before-fixes/)) and the deployment itself
found problems that local Compose had hidden:

1. **Serial S3 writes inside the upload.** Each 1,000-file `POST /batches` spent ~60 s on
   2,000 sequential S3 requests: 78.5 s total, against 5.6 s after the fix. Originals are now
   written 16 at a time (`WILDINBOX_STORE_CONCURRENCY`).
2. **Queue time included storage.** `job.created_at` took the transaction start, before the
   S3 writes, so run 1 reported 61 s of queueing. The job is now stamped when it is created.
3. **S3 connection pool smaller than the writers.** The logs showed `Connection pool is full`
   (boto3 default 10, versus 16 writers). The pool is now sized to match.
4. **Nightly backups would have failed.** Under cron, `aws` (a snap) was not on `PATH`:
   `aws: command not found`. The scripts now set `PATH`.
5. **Deployment:** an image tag containing the branch name (`/` is not allowed in tags);
   four services building the same image at once (`already exists`); token JSON that bash
   could not `source`.
6. **Measurement:** the SSH tunnel from the laptop dropped under two simultaneous 110 MB
   uploads ([`tunnel-concurrent-failure.log`](run-1-before-fixes/tunnel-concurrent-failure.log)).
   The API logged no requests during the failure, so it was the link, not the service. The
   workloads were then run from the VM, and internet upload was measured on its own.

## Limitations

- **One workspace.** Access needs a token, and batches are private to the deployment's
  token holders, but every token holder sees every batch. There are no per-user batches.
- **A restart costs up to 2.5 minutes.** A job interrupted mid-batch waits out the 30 s grace
  period and the 120 s lease before another worker takes it over. Nothing is lost, but it is
  slow. A shorter `WILDINBOX_LEASE_SECONDS` trades that against false recoveries on long
  chunks.
- **No TLS.** Staging is reached over an SSH tunnel. A public endpoint needs a TLS reverse
  proxy first.
- **One machine, one size.** Only 2 vCPU were measured, and a single batch does not
  parallelise across workers.
- **Sample images are resized dataset images** (~110 KB). Full-size camera JPEGs (2–5 MB)
  would upload and decode more slowly; this was not measured.

## Reproduce

On the staging VM, after [deploying](../../docs/deployment.md) (a sample batch is built with
`scripts/make_sample_batch.py` and copied to the VM; the VM needs `python3-httpx`). Results
go outside the checkout, so a later `git checkout` of a new tag does not collide with them:

```bash
export WILDINBOX_TOKEN=<a token>
python3 scripts/loadtest.py --label workers-1 --api http://localhost:8000 \
  --stack "cd /opt/wildinbox && deploy/staging/wi" \
  --scenarios small,thousand,invalid,concurrent,restart --out ~/loadtest-results/workers-1.json
# set WILDINBOX_WORKERS=2 in /etc/wildinbox/secrets.env, then: deploy/staging/wi up -d --wait
python3 scripts/loadtest.py --label workers-2 ... --scenarios thousand,concurrent \
  --out ~/loadtest-results/workers-2.json
deploy/staging/restore_drill.sh
```

From your own machine, through the tunnel (adds network transfer and round-trip time):

```bash
uv run python scripts/loadtest.py --label internet-thousand --api http://localhost:8000 \
  --ssh "ssh wildinbox-staging" --stack "cd /opt/wildinbox && deploy/staging/wi" \
  --scenarios thousand --latency-requests 100 --out reports/staging/internet-thousand.json
```
