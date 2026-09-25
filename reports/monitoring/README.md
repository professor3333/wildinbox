# Monitoring: the live deployment after update cycle 1

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
