# v2 experiment: camera adaptation

## Why

The v1 release meets animal retention by reviewing everything: 0% review
reduction against a 50% target, and no species label accepted automatically
([final evaluation](../reports/final_evaluation/README.md)). Only 8% of
final-test events are empty, so filtering cannot get there; automatic species
labels would have to. Development data says the current model cannot provide
them on new cameras
([feasibility](../reports/adaptation/feasibility/README.md)):

- on unseen development cameras, no label passes the release rule (95% Wilson
  lower bound on precision), and even a perfect selector over the model's
  suggestions labels under half the events;
- on cameras the model was trained on, an optimistic in-sample bound reaches
  43%.

A reserve's cameras are not new forever. Once a person has reviewed a camera's
first events, the model can be adapted to that camera. Cycle 1 showed this
helps that camera's later photos (event macro-F1 0.486 to 0.548,
[update cycle](../reports/update/README.md)).

## Question

After a person reviews the first **N** events of a new camera, can an adapted
model label that camera's later events automatically, with accepted-species
precision >= 95% and animal-event retention >= 98%? What share of later events
does it cover, and how does that compare with the unadapted release?

## Data and order of steps

The locked CCT20 final test has been used twice and stays historical evidence:
it is not read again, and v1's result against the 50% target stays on record.

1. **Feasibility** (done, development only):
   [`reports/adaptation/feasibility`](../reports/adaptation/feasibility/README.md).
2. **Camera-selection rule, committed before reading any metadata:**
   [`configs/experiments/fresh_cameras.yaml`](../configs/experiments/fresh_cameras.yaml).
   Unused Caltech Camera Traps locations (outside CCT20), split by hash into
   adaptation-development cameras and locked fresh-test cameras.
   **Amended once, before any image was downloaded:** v1 selected 8 test and
   0 development cameras, 4 of them with one image per sequence id, so
   [v2](../configs/experiments/fresh_cameras_v2.yaml) requires multi-frame
   sequences and smaller cameras (the reasons, and what was looked at, are in
   the file). Result: 12 fresh-test and 8 adaptation-development cameras,
   20,793 images ([`manifests/fresh-cameras-v2.json`](../manifests/fresh-cameras-v2.json);
   v1's output is kept beside it). Cameras whose sequence ids are per image
   are left out, a limitation of the experiment.
   **Development cameras acquired** (fresh test not downloaded): 8,956
   images, all accepted by `wildinbox data ingest`
   ([`manifests/cct_fresh_dev.lock.json`](../manifests/cct_fresh_dev.lock.json)),
   grouped into 3,107 capture events by `wildinbox dataset fresh`
   ([`configs/splits/cct_fresh_dev.yaml`](../configs/splits/cct_fresh_dev.yaml),
   [`manifests/cct-fresh-dev-v1.lock.json`](../manifests/cct-fresh-dev-v1.lock.json)).
   No camera, file, or near-duplicate image is shared with CCT20.
3. **Development**, on the adaptation-development cameras (plus the four CCT20
   development cameras): choose the adaptation method, N, the per-camera
   threshold rule, and the v2 target.
4. **Protocol, committed before any fresh-test image is downloaded:** method,
   N, thresholds, metrics, confidence intervals, and the v2 target with its
   justification.
   **Frozen:** [`configs/experiments/fresh_test.yaml`](../configs/experiments/fresh_test.yaml).
   No candidate reached the rule on development cameras
   ([iterations 1-3](../reports/adaptation/development/README.md)), so the
   target is set prospectively: the other-animal head at N = 50, species
   labels only (empty filtering off), species threshold 0.82. It passes if at
   least 30 labels are accepted with pooled precision >= 95% and Wilson lower
   bound >= 90%. Development estimates 93.6% (88.9-96.4) and 5.3% review
   reduction, so a pass is not expected with confidence.
5. **Fresh test, once:** download, adapt per camera from its first N events
   (with their ground-truth labels standing in for reviews, as in the update
   cycle), score the later events, report against the protocol whatever the
   result.
   `uv run wildinbox adaptation fresh-test` is the only code path that reads
   the fresh-test partition. It checks the protocol is committed and unchanged,
   every pinned hash, and a clean `src/` and `configs/`. Before reading any
   fresh data it binds the run to the committed split and ingest locks
   (`manifests/cct-fresh-test-v1.lock.json`, `manifests/cct_fresh_test.lock.json`),
   whose totals must reconcile with the protocol's expected 4,233 events and
   11,837 images, where only files the ingest declared rejected may be missing,
   and it creates `reports/adaptation/fresh_test/opened.json` atomically with
   that input identity, so a failed inference still leaves the opening
   recorded. Once read, the split must reproduce the lock's content digest,
   and every locked event and image must be loaded or excluded with a recorded
   reason. A later run must use the same protocol and inputs, reproduce the
   recorded `metrics.json`, and never rewrites either file. (The recorded run
   used the earlier runner, which wrote `opened.json` after scoring and did not
   check the locks; its split reproduces the lock digest `7e06132ddd4c` and
   its counts match the protocol.) `--dev-check` runs the same
   procedure on the development cameras only and must reproduce the recorded
   development result (216 / 220 correct, 6.7%) and the rule's choice of 0.82;
   it passes, and never opens the test.

   **Result (opened 2026-09-30): fail.** 505 of 570 automatically accepted
   species labels correct (88.6%, Wilson 85.7-91.0) against 95%; 12.8% review
   reduction; one camera (75) gave 39 of the 65 wrong labels. Automatic
   species acceptance stays off
   ([report](../reports/adaptation/fresh_test/README.md)). The fresh test is
   now opened; a further method needs new, untouched cameras.

## What will not count as success

- Coverage measured on the adaptation events themselves.
- A threshold or N chosen after seeing fresh-test results.
- Review reduction that does not count the N adaptation reviews and the audit
  sample.
