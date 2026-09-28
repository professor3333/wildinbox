# Load test with a year of history

Does the deployment keep its latency and memory targets once the database
holds a large, old history, and while batches are being processed? This run
measured that after the read path was changed to keep its cost bounded:
event pages, monitoring, and batch progress.

## Declared hardware

The laptop's local Docker Compose stack: an Apple M1 (8 GB) running macOS
26.6.2, and an OrbStack Docker VM with 8 vCPUs and 3.9 GiB. One worker, CPU
inference. The load generator runs on the same laptop. This is **not** the
2-vCPU AWS VM of the [staging runs](../staging/README.md), so the two sets of
numbers are not directly comparable. The machine record is in `loadtest.json`
(`machine`).

## Method

1. **Disposable stack.** `docker compose -p wildinbox-history up -d --build`
   gives the stack its own volumes, and the E3 release (`finetune-e3-deep-balanced@7a25aea97c76`,
   policy `conservative/v2`) is registered as in the README's Quick start.
2. **History.** [`scripts/seed_history.py`](../../scripts/seed_history.py)
   (seed 0) inserted 250 synthetic memory cards of 1,000 images each, from 40
   cameras, spread over the past 365 days. That is 250,000 images, 93,804
   events, one prediction per image and one decision per event, and reviews on
   20% of events (10% of those reviewed twice). The database was 383 MiB.
   Synthetic rows have no image files and are marked in their manifests.
3. **Load.** `scripts/loadtest.py --probe` ran its five scenarios: 24 images,
   1,000 images, invalid files, two 1,000-image batches uploaded together, and
   1,000 images with the worker restarted at 40%.
   - **During each scenario** a probe made one request every 0.25 s, cycling
     through the requests the review interface makes: the sidebar's batch
     list; the review, filtered and audit queues (8 per page, all batches);
     the timeline's first and last pages (100 per page) and camera list; and
     one night of last night's visitors. It also called `GET /monitoring`
     every 30 s.
   - **Afterwards, on the idle stack,** it made 200 requests per route, plus
     `GET /events?animal=true` over all history, which the interface does
     not issue.

```bash
docker compose -p wildinbox-history exec -T api \
    python - --batches 250 --images-per-batch 1000 --days 365 < scripts/seed_history.py
uv run python scripts/loadtest.py --label laptop-history-250k \
    --stack "docker compose -p wildinbox-history" --probe --out reports/history/loadtest.json
```

## Results (`loadtest.json`)

**Processing.** Every batch finished with integrity intact: no image scored
twice, and exactly one decision per event.

| Scenario | Processing time |
|---|---|
| 1,000 images | 83.5 s |
| Two 1,000-image batches together | 258 s for both (one worker takes them one after the other) |
| 1,000 images, worker restarted at 400 | 262 s, including the restart |

All are within the 10-minute target for 1,000 images.

**Metadata latency, p95 in ms** (target < 500):

| Route | Idle | During 1,000 images | During 2 batches | During restart |
|---|---|---|---|---|
| Batch list (sidebar) | 40 | 221 | **630** | 339 |
| Review queue | 217 | 475 | **876** | **980** |
| Automatically filtered | 114 | 231 | 353 | 285 |
| Audit queue | 67 | 165 | 302 | 205 |
| Timeline, first page | 100 | 221 | **656** | **547** |
| Timeline, last page | 231 | 298 | 460 | 377 |
| Cameras | 31 | 104 | 170 | 154 |
| One night's visitors | 35 | 130 | 212 | 196 |
| One batch's events, summary, event detail | 45-66 | | | |
| `animal=true` over all history (not used by the UI) | 402 | | | |

The "During" columns come from 27-72 requests per route, so their p95 is
close to each route's maximum. `GET /monitoring` is not a metadata route. It
took 1.6 s p50 and 2.9 s p95 idle, and up to 13 s while two batches
processed.

**Memory**, peaks over the run: API 741 MiB, worker 656 MiB (limit 3,072),
Postgres 207 MiB, object store 532 MiB.

## What this shows

- **Idle, a year of history is handled within target**, and so is processing
  one batch. Before this work, a page of 100 events cost 402 queries and
  monitoring loaded every event as ORM objects.
- **Concurrent processing still breaks the target on this laptop.** The
  review queue, the batch list, and the timeline's first page went over
  500 ms. The review queue is the slowest query even idle: it counts every
  unreviewed needs-review event in the history, 64,266 of them here. Under
  load the worker's CPU inference, Postgres (whose 128 MB default buffer is
  smaller than the 425 MiB database), and monitoring's heavy reads all compete
  on one machine.
- **Runs vary a lot.** Across three full runs, two-batch throughput ranged
  from 7.7 to 11.7 images/s. The earlier runs are kept for that record:
  - `loadtest-first-probe.json`: the probe included the all-history `animal`
    query, which the interface never makes.
  - `loadtest-before-summary-fix.json`: before batch progress was counted in
    SQL. A progress poll of a 1,000-image batch took 298 ms p95 and now takes
    22 ms.

## Changes this run led to

- Postgres JIT is off for the application's connections. It took 294 of
  584 ms for a filtered count over history.
- `created_at` is indexed on images, predictions, and events. Monitoring's
  window counts had been scanning every row.
- Monitoring reads a 90-day history window through one projected query:
  22,827 of 93,804 events here, 42 MB instead of about 12 KB per event as
  ORM objects. It reads active and recent jobs only.
- Batch progress (`GET /batches/{id}`) counts in SQL instead of loading every
  image.

## Not yet done

- Meet the target during concurrent processing. Followed up in
  [concurrency/](concurrency/README.md): the causes found were fixed and
  monitoring under load fell from about 20 s to 0.1-2.2 s. Its "during"
  numbers above were also inflated around uploads, because the probe shared
  the uploading process. The target while batches process is still not met
  reliably on this laptop.
- Measure on declared server hardware, not the laptop.
