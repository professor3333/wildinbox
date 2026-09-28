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
   6 adaptation-development cameras and 12 locked fresh-test cameras.
3. **Development**, on the adaptation-development cameras (plus the four CCT20
   development cameras): choose the adaptation method, N, the per-camera
   threshold rule, and the v2 target.
4. **Protocol, committed before any fresh-test image is downloaded:** method,
   N, thresholds, metrics, confidence intervals, and the v2 target with its
   justification.
5. **Fresh test, once:** download, adapt per camera from its first N events
   (with their ground-truth labels standing in for reviews, as in the update
   cycle), score the later events, report against the protocol whatever the
   result.

## What will not count as success

- Coverage measured on the adaptation events themselves.
- A threshold or N chosen after seeing fresh-test results.
- Review reduction that does not count the N adaptation reviews and the audit
  sample.
