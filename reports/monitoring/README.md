# Monitoring

Rules and fields: [docs/monitoring.md](../../docs/monitoring.md). This
directory holds the Stage 11 acceptance evidence
([`acceptance.json`](acceptance.json), from `scripts/monitoring_demo.py`) and
an earlier dashboard snapshot.

## Acceptance: a staged worker failure triggers operational alerts

On the Docker Compose deployment (release
`finetune-e3-deep-balanced@7a25aea97c76`), a 1,000-photo drill batch (camera
`crash-drill`, so real cameras stay clean) was uploaded, and the worker
container was SIGKILLed after 208 photos were scored.

| Seconds after the kill | Alert |
|---|---|
| 110.7 | **critical**: no worker is running while jobs wait |
| 110.7 | warning: a worker died without shutting down (last 24 h) |
| 122.2 | **critical**: a worker stopped renewing its lease |

- The alerts fire once the dead worker's heartbeat and the job's lease pass
  the 120 s lease period, which is the detection delay by design. `/metrics`
  showed `wildinbox_alerts{level="critical"} 2`.
- After `docker compose start worker`, the job was recovered and the batch
  completed 61 s later. Both critical alerts cleared. The death stays on
  record as a warning for 24 hours, which is an incident record, not an
  outage.
- The script counts only a *new* death. Deaths from earlier drill runs are
  still inside the 24-hour window (three were on record before this run).

**An unplanned test of the same mechanism.** During one earlier drill run,
the Mac hosting Docker went into idle sleep for about 26 minutes, right after
the worker restart. On wake, the worker's own heartbeat was 26 minutes old.
Recovery treated the attempt as stale and queued a new one, and the frozen
attempt found its lease taken and stopped. The new attempt finished the batch
in 4 s. The result was 1,000 predictions and 1,000 decisions for 1,000
photos, with no duplicates. Later runs used `caffeinate`.

## Acceptance: substantially changed inputs change model-behavior indicators

Camera 90's baseline is its deployment batch (1,007 photos) plus earlier
batches. From its photos, 40 whole sequences (118 photos) were uploaded
twice:
- **control**: unchanged, re-encoded at JPEG quality 95, because exact
  duplicates are skipped;
- **degraded**: the same photos blurred, low in contrast, and darkened, as
  through a dirty or misted lens.

| Latest batch vs earlier batches (camera 90) | Control | Degraded | Alert limit |
|---|---|---|---|
| Suggested-label mix, PSI | 0.167 | **3.757** | 0.25 |
| Confidence distribution, PSI | 0.040 | 0.107 | 0.25 |
| Mean confidence change | -0.012 | +0.018 | ±0.10 |
| Share with low confidence, change | -0.076 | +0.131 | (shown, not alerted) |
| Sharpness change | +6% | **-100%** | ±50% |
| Behavior alerts | none | label mix, sharpness | |
| Top suggestions | coyote 16, empty 7, opossum 5 | **empty 16**, dog 10, bobcat 10 | |

What this shows:

- **Confidence alone would have missed it.** On degraded photos the model did
  not become less sure on average; it confidently called many events empty,
  dog, or bobcat. The label mix and photo sharpness flagged the batch.
- **Why automatic filtering needs audits and conservative starts.** With
  automatic filtering on, those confident "empty" suggestions would be
  filtered away. A behavior signal is the cue to audit before trusting
  automation on a changed camera.
- **Batch size limits these indicators.** The control's label PSI of 0.167
  comes from sampling 40 events. Batches under 20 events are not compared.
- **About the "(the release also changed)" note in the alerts.** A few of
  camera 90's earlier events were processed today under
  `finetune-e3-deep-balanced@518a8da39ee0`. That is the same weights under
  decision policy v1. The alert reports this so that a model change is not
  mistaken for a data change.
- **The baseline includes earlier demo runs.** Batches from earlier demo runs
  (including earlier degraded batches) are part of the baseline, which makes
  the degraded batch harder to flag, not easier.

## Earlier: the live deployment after update cycle 1

Snapshot of `GET /monitoring` ([`snapshot.json`](snapshot.json)) on the Docker
Compose deployment: cameras 90 and 125 (2,732 photos, every event reviewed by
the simulated reviewer) and two small batches from cameras 51 and 108.

## What it shows

| View | Finding |
|---|---|
| Operations | 4 jobs succeeded; no queue, stale leases, failed jobs, unusable files, or failed frames. E3: 45 s per 1,000 images; the update candidate's single 29-photo batch: 92 s per 1,000 (model loading dominates a tiny batch). |
| Accuracy (reviews) | **Camera 90: reviewers corrected 65% of suggestions [60-70%]**, raising a warning; camera 125: 42% [38-46%]. Cameras 51 and 108 have no reviews, so their accuracy is unknown, not zero. |
| Signals (no labels) | Suggested-label mix shifted between earlier and recent events on camera 90 (PSI 0.48) and camera 125 (0.36), and camera 90's frames got 57% less sharp; no release change between those windows, so the shift is in the data (seasons, weather, vegetation), not the model. Cameras 51 and 108 have too few events to compare. |

The signals flagged change before any label; only the reviews say that camera
90's suggestions are mostly wrong. That is the distinction the dashboard keeps.
