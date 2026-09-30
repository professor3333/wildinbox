# v2 fresh test: result

The frozen operating point of [`configs/experiments/fresh_test.yaml`](../../../configs/experiments/fresh_test.yaml),
measured once on the 12 locked fresh-test cameras
([`metrics.json`](metrics.json), opened 2026-09-30 at commit `30387d5` on MPS,
[`opened.json`](opened.json)). Nothing was chosen on these cameras.

## Outcome: fail

**Claim tested:** after a person reviews a new camera's first 50 events,
species labels accepted automatically on its later events are correct at least
95% of the time.

| Pass condition | Required | Fresh test | Met |
|---|---|---|---|
| Accepted labels | >= 30 | 570 | yes |
| Pooled precision | >= 95% | **88.6%** (505 / 570) | **no** |
| Wilson 95% lower bound | >= 90% | **85.7%** (85.7-91.0) | **no** |

Automatic species acceptance stays off. The original 50% review-reduction
target was not met by v1 and is not addressed by this operating point.

## What the operating point did

4,233 capture events (11,837 images); the first 50 per camera stand in for
reviews, leaving 3,633 later events. Empty filtering off, so no animal event
leaves review (retention 100%).

| | Fresh test | Development estimate (leave-one-camera-out) |
|---|---|---|
| Accepted labels correct | 505 / 570 (88.6%, Wilson 85.7-91.0; camera bootstrap 73.2-96.9) | 161 / 172 (93.6%, 88.9-96.4) |
| Review reduction, reviews and 5% audit counted | 12.8% (camera bootstrap 6.1-20.0) | 5.3% |
| Later events labelled automatically | 15.7% | |

The adapted head accepted more labels than development predicted, at lower
precision. Review reduction is not a pass condition.

## By camera

| Camera | Events | Accepted correct | Review reduction |
|---|---|---|---|
| 93 | 271 | 93 / 93 | 32.6% |
| 98 | 350 | 111 / 117 | 31.8% |
| 103 | 677 | 153 / 161 | 22.6% |
| 75 | 478 | **56 / 95 (58.9%)** | 18.9% |
| 126 | 193 | 24 / 27 | 13.3% |
| 117 | 374 | 33 / 40 | 10.2% |
| 24 | 229 | 20 / 20 | 8.3% |
| 81 | 185 | 9 / 9 | 4.6% |
| 135 | 208 | 2 / 3 | 1.4% |
| 13 | 179 | 1 / 2 | 1.1% |
| 95 | 581 | 2 / 2 | 0.3% |
| 41 | 508 | 1 / 1 | 0.2% |

- Camera 75 contributes 39 of the 65 wrong labels. Without it the other 11
  cameras reach 449 / 475 (94.5%); this is a post-hoc description, not a
  result: the protocol pools all 12 cameras, and a camera like 75 is exactly
  what a deployment meets.
- Four cameras (103, 98, 75, 93) give 466 of the 570 accepted labels (82%);
  four get almost no automation (under 1.5%).

## Wrong labels, by what the event really was

| True event | Wrong labels |
|---|---|
| Supported species confused with another (dog called bobcat 19, bobcat called opossum 6, cat called opossum 5, dog called opossum 4, other pairs 9) | 43 |
| Unsupported animal called a supported species (coyote 8, rabbit 5, bobcat 3, opossum 1) | 17 |
| Empty event called bobcat or opossum | 5 |

Unsupported animals are 1,271 of 4,233 fresh-test events (30%), yet they
give 17 wrong labels; confusions between supported species give 43, most of
them dogs called bobcat.

## Comparisons (protocol-defined)

| Method | Threshold from development rule | At the frozen 0.82 (diagnostic) |
|---|---|---|
| Adapted head (this protocol) | 0.82: 505 / 570 (88.6%), 12.8% | same |
| Unadapted release | none passes: 0 accepted, 0% | 131 / 163 (80.4%), 3.7% |
| Head without the camera's reviews | none passes: 0 accepted, 0% | 269 / 338 (79.6%), 7.6% |

Under the release rule neither comparison accepts any label. At the same
threshold, the camera's 50 reviews raise precision from about 80% to 88.6%
and review reduction from 3.7% (release) and 7.6% (head without reviews) to
12.8%: adaptation helps, but not enough for the 95% target.

## Reading

Adapting to a new camera from its first 50 reviewed events makes automatic
species labels more accurate and more frequent than the unadapted model. It
is not reliable enough
to switch on: one camera in twelve produced 41% wrong accepted labels, and
nothing in development predicted which. This fresh test is now opened; any
further method needs new, untouched cameras.
