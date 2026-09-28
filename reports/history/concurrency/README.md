# Metadata latency while batches process

The [history load test](../README.md) met the 500 ms p95 target idle and during
one batch, but not while two batches were processed or a worker restarted:
review queue 876-980 ms, batch list 630 ms, `GET /monitoring` about 13 s. This
work looked for the causes, fixed each one it found, and measured again.

**Result, briefly.**
- Each cause found was confirmed with a controlled measurement, and each fix
  removed its effect.
- `GET /monitoring` under load went from 20-22 s to 0.1-2.2 s. The first call
  after a restart, before anything is cached, still takes 6-8 s (43 s
  before).
- Idle latency improved: the worst idle route went from 456 ms to 145-208 ms.
- **The p95 target while batches process is still not met reliably on this
  laptop.** Across the final runs it ranged 495-923 ms, and the laptop's own
  state moved results by 2x between runs of identical code.
- The operating envelope measured here is at the end of this page.

## Hardware and its limits

The same laptop stack as the history load test:
- Apple M1 with 8 GB of RAM, running an OrbStack Docker VM with 8 vCPUs and
  3.9 GiB.
- One worker, on a disposable history of 105,000-118,000 events.
- The load generator runs on the same laptop.

Throughout these runs the host was 13-16 GB into swap (browser and editor
included). The time to process 1,000 identical images varied from 85 s to
161 s between runs of the same code, and latency under load moved with it.
**These are laptop measurements, not a benchmark of the declared server
hardware** (the 2-vCPU AWS VM of the staging runs, which was not started).

## What was wrong, and how each cause was shown

1. **The load test timed its own client.** Its probe thread ran in the
   process that was uploading, and that process spent seconds building a
   1,000-file multipart body under its own interpreter lock. With the probe in
   a separate process, the stalls during the receive phase disappeared: 3-4 s
   became at most 350 ms. The probe now runs in its own process
   (`scripts/loadtest.py`), so earlier "during" numbers overstate latency
   around uploads.
2. **Uploads stalled every request in the API process.** Probing every 50 ms
   during an upload and aligning each request with the server's own phase
   timings:
   - the store phase (16 boto3 threads writing originals) held the lock. A
     7 ms `GET /version` took up to 874 ms, an event page up to 1.1 s;
   - with 2 threads instead of 16, the median stall fell (70 → 17 ms), but
     the store phase doubled.
   - **Fix:** originals are written from a spawned store process
     (`wildinbox/api/originals.py`). During the store phase `GET /version`
     then peaked at 75-107 ms.
3. **The upload's database step and garbage collection paused the process.**
   Sampling which thread held the lock during an upload (py-spy) showed:
   - building and flushing 1,000 ORM image objects held it for 2-3 s;
   - a full garbage collection, walking ~380,000 objects created by imports
     (PyTorch among them), took 2.0 s. Its timing matched a 2.35 s stall.
   - **Fix:** one multi-row INSERT, and `gc.freeze()` after startup.
4. **Inference oversubscribed the machine.** The worker's PyTorch used all 8
   vCPUs that the API and Postgres also need. Two 1,000-image batches were
   probed only while processing, alternating 8 and 4 threads over 8 runs
   (order A B A B, then B A A B, so drift cancels):

   | Threads | Pooled p95 per run (ms) | Median p95 | Median processing |
   |---|---|---|---|
   | 8 (all) | 1,418 · 613 · 851 · 654 | 752 ms | 298 s |
   | 4 (half) | 688 · 467 · 498 · 490 | 494 ms | 210 s |

   Four threads won every adjacent pair, and inference itself got faster.
   **Fix:** unset `WILDINBOX_TORCH_THREADS` now means half the CPUs. Raw data:
   [`threads-ab.jsonl`](threads-ab.jsonl).
5. **Every review-queue page counted the whole history.** The page query took
   9 ms; its `total` counted all ~81,000 unreviewed events (warm about 100 ms,
   cold or under load 0.5-1 s).
   - **Fix:** the queues and timeline request `exact_total=false`. The count
     stops 10,000 past the page (warm 24 ms), and the interface shows
     "10,000+". Totals stay exact by default, and `next_offset` is exact
     either way.

Also carried here from the earlier attempt (`perf/concurrent-latency`):
- an index on `events (start_at, id)`;
- Postgres memory sized to the database;
- a shared, cached monitoring summary.

The `main` baseline below had none of these.

## Full load tests, before and after

The same `scripts/loadtest.py --probe` (separate-process probe) for every
run. The **pooled p95** is every metadata request made during the scenario;
per route there are only 30-70 requests, so a route's p95 is about its
second-worst.

| Run | Code | 1,000 images | Two batches | Worker restart | Monitoring under load | Worst idle route |
|---|---|---|---|---|---|---|
| [before](before-main.json) | `main` `98c14ad` | 753 ms (161 s) | 686 ms | 401 ms | 20.3-21.9 s | 456 ms |
| [after 1](after-1-upload-fixes.json) | upload fixes, 8 threads | per-route only* | per-route only* | per-route only* | 0.03-1.7 s | 151 ms |
| [after 2](after-2-threads.json) | + 4 threads | 290 ms (85 s) | 505 ms | 382 ms | 0.13-0.54 s | 373 ms |
| [after 3](after-3-final.json) | + count floor (final) | 495 ms (109 s) | 714 ms | 923 ms | 0.34-2.2 s | 145 ms |
| [after 4](after-4-final.json) | final | 871 ms (161 s) | 538 ms | 739 ms | 0.14-0.37 s | 208 ms |

\* Run before the probe reported pooled figures.

"Monitoring under load" is the p95 of `GET /monitoring` during the 1,000-image,
two-batch, and restart scenarios. The first call after the stack starts
computes the summary cold: 5.9-8.4 s in the after-runs, 42.9 s before.

Brackets give the 1,000-image processing time, a measure of how loaded the
laptop was. Every batch in every run finished with integrity intact: no image
scored twice, one decision per event, and no probe errors. The runs were made
in the order after 1, after 2, before, after 3, after 4.

Reading the table:
- **Where the machine was equally loaded, the result was about the same.**
  After 4 processed 1,000 images in 161 s, like the baseline, and its p95
  (871 ms) is no better than the baseline's (753 ms). The controlled
  experiments above show each cause's effect. These full runs cannot show the
  end-to-end effect under this much noise, and do not show that the target is
  met.
- **Monitoring and idle latency improved in every run.**

## Measured operating envelope on this laptop

| Load | Metadata p95 |
|---|---|
| Idle, a year of history (~115,000 events) | **within target**: worst route 145-208 ms (final code) |
| `GET /monitoring` | idle 0.05-0.37 s, under load 0.03-2.2 s, first call after a restart 6-8 s. Not a metadata route; before this work 11-43 s |
| One 1,000-image batch processing | **not reliably within target**: 290-871 ms across runs |
| Two batches, or a worker restart | **not within target**: 505-923 ms |

## Not done

- **Measure on the declared server hardware**, or on this laptop with its
  memory freed (no swap), before claiming the target under processing load.
  The same commands reproduce every run here:

  ```bash
  uv run python scripts/loadtest.py --label <label> \
      --stack "docker compose -p wildinbox-history" --probe --out <file>.json
  ```

- **Remaining suspects under load, not tested yet:**
  - the batch list and filtered queue, the worst routes in some runs;
  - Postgres and the API competing with inference for memory in a 3.9 GiB
    VM.
