# Adaptation development, iteration 1 (development cameras only)

`uv run wildinbox adaptation develop` on the 8 adaptation-development cameras
(3,107 events; [`metrics.json`](metrics.json)). Each camera's first N events,
in time order, stand in for reviews (ground truth); its later events are
scored. Thresholds follow the release rule (95% Wilson lower bound on accepted
precision; false-empty upper bound 2%) and are chosen leave-one-camera-out.
Review reduction counts every event, reviewed ones included, and charges a 5%
audit on automated events. The fresh test is not downloaded.

## Result: no method passes the rule for species labels

| Method | N | Review reduction | Species labels accepted | Animal retention |
|---|---|---|---|---|
| Release (no adaptation) | 50 | 3.7% | 0 | 97.8% |
| Prior shift (camera class mix) | 50 | 7.0% | 0 | 94.8% |
| Camera head (linear, frozen features) | 50 | 0.0% | 0 | 100.0% |

N = 25 and 100 give the same picture (metrics.json). All reduction comes from
the empty filter, and applied to an unseen camera it loses more than 2% of
animal events (camera 64 alone loses 111 of 175 under prior shift).

Precision against the species threshold, later events pooled, N = 50
(in-sample, so optimistic):

| Threshold | Release: accepted, precision | Prior shift | Camera head |
|---|---|---|---|
| 0.8 | 1.9%, 0.77 | 11.3%, 0.84 | 53.4%, 0.39 |
| 0.9 | 0.3%, 0.43 | 4.2%, 0.90 | 50.2%, 0.40 |
| 0.95 | 0.1%, 0.00 | 1.6%, 0.93 | 48.2%, 0.40 |

## What limits precision

- **Unsupported animals called a supported species.** Birds are 535 of 3,107
  events here (17%); confident errors are mostly birds called bobcat, rabbit,
  or coyote, and rodents called rabbit. The model has no "other animal"
  output, and the unfamiliar score was not adopted.
- **Empty frames called an animal** (dog, bobcat) and **raccoon called
  opossum**.
- **Prior shift** raises precision (0.84 to 0.93 at 1.6-11% coverage) but not
  to the rule's lower bound.
- **The camera head as configured is overconfident**: 40% precision even at
  0.99. It never sees unsupported animals, so birds get a confident supported
  label, and its regularisation and camera weight were not tuned.

## Corrected during this iteration

The first run pooled review reduction by adding each camera's rate (reporting
e.g. 25% for the release instead of 3.7%). Pooled rates are now computed from
pooled counts, with a test.

## Next candidates

1. Camera head with an **other-animal class** learned from the camera's
   reviewed unsupported events (a reviewer naming a bird is exactly the signal
   a deployed camera produces), with its regularisation chosen by nested
   leave-one-camera-out.
2. Per-camera fine-tuning with the update-cycle recipe.
3. If neither reaches the rule, a prospectively defined v2 target, stated with
   this evidence.
