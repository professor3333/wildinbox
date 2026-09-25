# Monitoring and alert rules

`GET /monitoring` (JSON), `GET /metrics` (Prometheus text), and the review
interface's **Monitoring** page show the same data in two views. Everything is
computed from PostgreSQL on request, except API latency and error counts, which
the API process keeps in memory since it started. Thresholds live in
[`configs/monitoring/monitoring.yaml`](../configs/monitoring/monitoring.yaml).

| Operational health | Model behavior |
|---|---|
| Queue age and job failures | Fraction automatically filtered |
| Throughput and processing latency | Fraction requiring review (and auto-labeled) |
| API errors and response latency | Class and confidence distributions |
| Worker memory and restarts | Audited false-empty and species errors |
| Storage growth and batch cost | Differences between cameras and time periods |

## Three kinds of evidence, never mixed

1. **Operational health.** Is processing working?
2. **Behavior signals (no labels).** Did what the model decides, or the photos
   it sees, change? A shift is a *warning sign*: a moved camera, a new season,
   a dirty lens, or a new release. It does not say whether accuracy changed.
3. **Measured errors (labels).** Only reviews say whether suggestions were
   right. The **audit sample** is the only unbiased measure of automatic
   decisions, because its events are drawn at random (5% by default; see
   `WILDINBOX_AUDIT_RATE`). Other reviews were chosen, mostly because the
   event was uncertain, so their correction rate describes those events only.
   **A camera with no audit labels has unknown observed quality**, not zero
   errors, and the dashboard says so.

## Model behavior: cameras, batches, and time periods

- **Per camera**: shares filtered, auto-labeled, and sent to review, the mix of
  suggested labels, and confidence.
- **Latest batch against the camera's earlier batches**: a memory card is the
  natural unit of change. The comparison covers the label mix and confidence
  distribution (population stability index, PSI), changes in each share and in
  mean confidence, and photo quality (night share, sharpness).
- **Time periods, with the camera mix separated**: the last 7 days of decisions
  against the 7 before. A global distribution can shift only because a
  different mix of cameras sent photos, while no camera changed. The shift is
  split in two:
  - *from camera mix*: earlier period vs each camera's earlier behavior,
    reweighted to the recent camera mix;
  - *within cameras*: that reweighted expectation vs what recent events
    actually got.

  A large total shift with a small within-camera shift means that the cameras
  changed, not the model's behavior on them. Cameras with no earlier events
  are new and count toward the mix.
- **Over capture time, per camera**: a camera's most recent 100 events by
  capture time against its earlier ones (seasons, vegetation).

## Alert rules

Level meanings: **critical** means act now (work is stuck or automation is
losing animals); **warning** means investigate soon; **signal** (`info`) means
something changed and needs a look, not necessarily a fault.

| Area | Alert | Level | Rule (default) | What to do |
|---|---|---|---|---|
| operations | a worker stopped renewing its lease | critical | running jobs with a heartbeat older than the lease (120 s) > `max_stale_leases` (0) | Check workers; the job is recovered automatically when a worker runs. |
| operations | no worker is running while jobs wait | critical | queued or running jobs, and no worker reported within the lease | `docker compose up -d worker`; check the worker logs. |
| operations | jobs failed terminally | critical | failed jobs in the window (7 days) > `max_failed_jobs` (0) | Read the job error on the Batches page; re-upload if the cause is fixed. |
| operations | a worker died without shutting down | warning | worker processes that stopped reporting with no clean stop in the last 24 h > `max_worker_deaths` (0) | Look for out-of-memory kills or crashes; jobs were recovered by lease. |
| operations | workers restarted often | warning | worker starts in the last 24 h > `max_worker_starts` (6) | Crash loop; check logs and memory. |
| operations | a worker uses a lot of memory | warning | resident memory > `max_worker_rss_mb` (3,072 MB) | Lower `WILDINBOX_INFERENCE_CHUNK` or add memory. |
| operations | a job has waited too long to run | warning | oldest runnable queued job > `max_queue_age_seconds` (900) | Add workers: `docker compose up -d --scale worker=3`. |
| operations | frames failed inference | warning | failed frames / images scored > `max_processing_error_rate` (2%) | Check the frame errors on the event details. |
| operations | many uploaded files were unusable | warning | unusable files / files uploaded > `max_unreadable_rate` (5%) | The camera or card may be producing corrupt files. |
| operations | the API returned server errors | warning | 5xx / responses since the API started > `max_api_server_error_rate` (1%) | Check the API logs. |
| operations | metadata API is slow | warning | p95 of the slowest metadata GET route with ≥ 20 requests > `max_api_p95_ms` (500 ms) | Check database load; add indexes. |
| operations | stored originals are near the disk budget | warning | original photos stored > `max_storage_gb` (200 GB) | Archive old batches or add disk. |
| audit | audited false-empty rate above the limit | critical | lower 95% bound of false empties among audited filtered events > `max_false_empty_rate` (2%) | Turn automatic filtering off for the release; retrain or recalibrate. |
| audit | an audit found an animal in an automatically filtered event | warning | any audited filtered event reviewed as an animal | Look at the event; one is a sighting that would have been missed. |
| audit | audited species-error rate above the limit | warning | lower 95% bound of wrong species among audited auto-labeled events > `max_species_error_rate` (5%) | Stop auto-accepting that species. |
| audit | automation runs here with no audit labels yet | signal | a camera has automatic decisions and no reviewed audit samples | Review that camera's audit queue: its quality is unknown. |
| accuracy | reviewers correct most suggestions | warning | lower 95% bound of a camera's correction rate > `max_correction_rate` (50%), with ≥ 30 judged reviews | That camera needs more review or a model update. |
| behavior | latest batch differs from earlier batches | signal | a camera's latest batch (≥ 20 events) against its earlier batches (≥ 50 events): label or confidence PSI > 0.25; a share (filtered, auto-labeled, to review, unfamiliar) changing by > 15 points; mean confidence changing by > 0.10; night share changing by > 20 points; sharpness changing by > 50% | Look at the batch. Was the camera moved, or the lens dirty? Is it a new season? Audit before trusting automation there. |
| behavior | shifted globally only because the camera mix changed | signal | global PSI > 0.25 with within-camera PSI ≤ 0.25 | Usually nothing: different cameras sent photos. |
| behavior | shifted within cameras | signal | within-camera PSI > 0.25 | As for a changed batch, across cameras. |
| signal | label mix, confidence, uncertainty, night share, or sharpness changed over capture time | signal | a camera's recent 100 events against its earlier ones (thresholds under `drift`) | As above. When the release also changed between windows, the alert says so. |

Signals never trigger an automatic action. They make a person look, and they
are the reason to audit more before enabling automatic filtering on a changed
camera.

## Where the data comes from

- Workers write one `worker_processes` row per process start, refreshed every
  15 s with resident memory. A clean shutdown records `stopped_at`; a row that
  went stale without one is a worker that died (killed, crashed, or out of
  memory). Job leases stay the source of truth for recovering work.
- Job latency is `started - created` (waiting), `finished - started`
  (processing), and `finished - created` (upload to done), for jobs finished
  in the window. Batch cost is the processing time per 1,000 images of each
  recent batch.
- Storage counts original photos (thumbnails are small and not counted), with
  bytes uploaded per day.

## Prometheus

`GET /metrics` exposes the same numbers as gauges (`wildinbox_*`), including:
- `wildinbox_alerts{level=...}`;
- `wildinbox_workers_live`, `wildinbox_workers_died_window`, and
  `wildinbox_worker_rss_bytes`;
- `wildinbox_http_responses{class=...}`;
- `wildinbox_camera_filtered_share` and `wildinbox_camera_audit_labels`.

An external Prometheus can alert on `wildinbox_alerts{level="critical"} > 0`.
The rules themselves stay in one place, `monitoring.yaml`.

## Acceptance evidence

`scripts/monitoring_demo.py` stages a worker failure (SIGKILL mid-batch) and
uploads a control batch and a degraded batch from camera 90. The results are
in [reports/monitoring/README.md](../reports/monitoring/README.md).
