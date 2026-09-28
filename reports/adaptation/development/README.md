# Adaptation development (development cameras only)

Iterations are kept in order; [`metrics.json`](metrics.json) holds the latest
run, which recomputes every method below.

## Iteration 2: a head with an other-animal class

**Method.** A linear head on the released model's frozen features with 9
outputs: the 8 classes plus `other_animal`. Its base frames come from the CCT20
calibration and policy-validation cameras, which the network never trained on
(5,305 frames; 976 unsupported-animal frames become `other_animal`). The
camera's reviewed frames are added with weight `share`, reviewed unsupported
animals as `other_animal`. An `other_animal` suggestion is never accepted.
Regularisation `C` and `share` are chosen by **nested** leave-one-camera-out:
for each held-out camera, configuration and thresholds come from the other 7.

**Result (N = 50 reviewed events per camera):**

| Method | Review reduction | Accepted labels correct | Animal retention, pooled | Worst camera |
|---|---|---|---|---|
| Release | 3.7% | none accepted | 97.8% | 93.1% |
| Prior shift | 7.0% | none accepted | 94.8% | 36.6% |
| Head, training-frame base (iteration 1) | 0.0% | none accepted | 100% | 100% |
| **Head with other-animal class** | **9.4%** | **158 / 168 (94.0%, CI 89.4-96.7)** | 98.3% | 82.8% |
| Same head without the camera's reviews (ablation) | 3.6% | none accepted | 97.7% | 86.0% |

- **Adaptation is what enables species labels**: the same head without the
  camera's reviews accepts none. The `other_animal` class sends 399 of 595
  unsupported-animal events to review.
- **Still short of the rule on unseen cameras**: 94.0% accepted precision,
  below 95%, and far from a 50% reduction. The gain is concentrated on
  cameras 112 and 58.
- **Empty filtering fails per camera for every method**: pooled retention
  near 98% hides one camera losing 17% of its animal events (camera 18,
  23 of 134, mostly to a low empty threshold chosen on the other cameras).
- **Unstable in N**: at N = 25 the head reaches 7.6% (43 of 47 correct), at
  N = 100 4.6% (3 of 6); with fewer later events per camera the rule rarely
  finds a threshold.
- **Grid**: the first grid (C 0.01-1, share 0.25-0.75) accepted no labels
  under nested selection (5.4%, 3.9%, 3.5% reduction at N = 25, 50, 100, all
  from empty filtering); in-sample curves showed precision rising as C fell,
  so the grid was extended to C 0.001-0.03, share 0.1-0.5. The chosen
  configurations sit at C = 0.001-0.003 and share 0.1.

**Reading:** frozen-feature adaptation buys roughly 5-10% of reviews at about
94% precision on development cameras, with empty filtering unsafe per camera.
The remaining candidate is per-camera fine-tuning; if it does not change the
picture, the v2 target has to be set prospectively from this evidence.

---

## Iteration 1

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
