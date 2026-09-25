# Model card: WildInbox species classifier

**Status: selected candidate, calibrated, not yet released.** Automatic
filtering and automatic acceptance are both **off**: every event goes to review
with its suggested label. The locked final test has not been opened. Sections
marked *planned* describe the process a release will follow.

| | |
|---|---|
| Model | `finetune-e3-deep-balanced` (EfficientNet-B0, fine-tuned) |
| Selected by | pre-registered rule, [`configs/experiments/selection.yaml`](../configs/experiments/selection.yaml) |
| Evidence | [`reports/experiments/README.md`](../reports/experiments/README.md) |
| Fallback / reference | `baseline-frozen-effnetb0-logreg-v1` ([report](../reports/baseline/README.md)) |

## Intended use

Triaging trail-camera captures for human review: suggesting whether a capture
event is likely empty, shows a supported species, or needs review. Predictions
are suggestions for a person to check.

Not intended for identifying individual animals, estimating population counts
or abundance, or labeling species outside the supported set.

## Supported classes

`empty`, bobcat, cat, coyote, dog, opossum, rabbit, raccoon. Chosen from the
training partition only (rules: [`docs/dataset.md`](dataset.md)). Every other
species is kept as an **unsupported-input** evaluation case, never relabeled as
empty. One suggested species per event; mixed-species events stay in review.

## Provenance

- Architecture and weights: torchvision EfficientNet-B0,
  `EfficientNet_B0_Weights.IMAGENET1K_V1` (ImageNet-1k,
  `efficientnet_b0_rwightman-7f5810bc.pth`). Not wildlife-specific, so no
  pretraining overlap with the CCT20 test cameras.
- Fine-tuning: blocks 3-8 + classification head trainable, class-balanced
  sampling, horizontal flip and box-safe random crops (no crop removes more
  than 10% of an annotated animal), 3 epochs, lr 0.0003, batch 32, seed
  20260924.
- Lineage: code `7d4a388`, MLflow run `b57ee7aa105949519561a99c78bca741`,
  preprocessing `6d9a950a6543` (shared by training and inference).

## Training data and splits

- Caltech Camera Traps, CCT20 subset (LILA; Community Data License Agreement,
  permissive variant).
- Split `cct20-splits-v1-7de740aba75f`, pinned in
  [`manifests/cct20-splits-v1.lock.json`](../manifests/cct20-splits-v1.lock.json).
- Partitions are **camera-disjoint**: train (7 cameras, 20,287 fit images),
  calibration (cameras 51, 108), policy validation (90, 125), and a locked final
  test (9 cameras). Every capture sequence stays in one partition; exact and
  near-duplicate images are checked across partitions
  ([split report](../reports/splits/cct20/README.md)).

## Metrics on unseen cameras (development partitions)

Calibration + policy-validation cameras; reference thresholds P(empty) >= 0.9,
species >= 0.9 (not chosen operating points).

| Metric | E3 | Baseline |
|---|---|---|
| Macro-F1, supported classes (image level) | 0.413 | 0.258 |
| Lowest per-species recall | 0.163 (cat) | 0.100 |
| Animal events filtered as empty | 57 / 1577 (3.6% [2.8, 4.7]) | 164 / 1577 |
| Automatically accepted labels (precision) | 234 events (69.7% [63.5, 75.2]) | 728 (22.3%) |
| Unsupported-animal events accepted as a known species | 34 | 144 |
| Events left for review | 77.6% | 46.3% |
| Macro-F1, held-out sequences from training cameras | 0.747 | 0.724 |
| CPU p50 latency per image (Apple M1) | 132 ms | 131 ms |

Per-class recall: empty 0.69, bobcat 0.49, cat 0.16, coyote 0.42, dog 0.57,
opossum 0.44, rabbit 0.43, raccoon 0.39. Final-test results will be added once,
after thresholds are frozen.

## Known failure modes

- **New camera locations:** macro-F1 falls from 0.75 on training cameras to
  0.41 on unseen ones; camera 51 alone accounts for 43 of the 57 missed animal
  events.
- **Small animals:** misclassifications concentrate on animals whose box
  covers < 1% of the frame.
- **Nighttime illumination and blur:** night macro-F1 0.356 vs day 0.394;
  blurry infrared frames are often called empty.
- **Occlusion:** not measured separately (no occlusion labels); partly visible
  animals appear among the confused-species and false-empty galleries.
- **Unsupported species:** squirrels and rodents are labeled rabbit, coyote,
  or dog with confidence up to 1.00; confidence alone does not reject them.
- **Cats:** recall 0.16, confused with dog, raccoon, and coyote.

## Calibration and operating point

Details: [`reports/calibration/README.md`](../reports/calibration/README.md).

- **Calibration:** temperature scaling fitted on the calibration cameras
  (51, 108): T = 2.01, so raw scores were overconfident. Held-out
  (policy-validation) expected calibration error falls from 0.274 to 0.062.
  Calibrated P(empty) is still overconfident on new cameras: images scored
  0.7-0.8 are empty only 42% of the time.
- **Rule** ([`operating_point.yaml`](../configs/experiments/operating_point.yaml),
  committed before any calibrated result): on the policy-validation cameras
  (90, 125), enable a threshold only if the worst case of its 95% Wilson
  interval meets the target: at most 2% of animal-containing events filtered
  as empty, at least 95% of accepted labels correct. v2
  ([`operating_point_v2.yaml`](../configs/experiments/operating_point_v2.yaml),
  also committed before its results) adds the versioned policy, unfamiliar
  flags if adopted, and the per-species gate; the outcome is unchanged.
- **Rule result:** empty filter at calibrated P(empty) >= 0.65; species
  acceptance disabled (no threshold reaches the precision bound; the few
  high-confidence acceptances are too few to prove 95%).
- **Released:** both disabled. The empty filter was switched off by a
  recorded deviation
  ([`operating_point_deviation.yaml`](../configs/experiments/operating_point_deviation.yaml)):
  it passed by a hair (upper bound 1.99%) but filters 9.8% of animal events on
  camera 51, and would take only 4% of events out of review.

## Decision policy

- **Policy `conservative/v1`** ([source](../src/wildinbox/policy/conservative.py)),
  released as `conservative/v1+fe20c7586555` in the versioned artifact
  [`policy.json`](../reports/calibration/policy.json), which also carries the
  calibration and unfamiliar-input versions.
- An event is filtered only when every frame has completed processing and looks
  empty; any pending or failed frame sends it to review (`processing_failure`).
  Any animal-looking frame keeps the event; conflicting frames, low confidence,
  possible unknowns, species without validation evidence, and disabled
  automation are each listed as machine-readable reasons.
- **Per-species gate:** a species is auto-accepted only if its own accepted
  events reach the precision bound. None does yet; opossum comes closest
  (26 of 26 correct at 0.7, lower bound 87%).
- **Reproducibility:** every development event's saved predictions and both
  decisions (released and the rule's) are committed; `wildinbox replay`
  reproduces all 1,847 exactly, and CI runs it.

## Unfamiliar inputs

Details: [`reports/unfamiliar/README.md`](../reports/unfamiliar/README.md).

- Score: mean cosine distance from an image's features to its 10 nearest
  training images, compared with ordinary confidence at the same false-flag
  rate (10% of supported animals).
- **Not adopted.** On the calibration cameras it detects 0.4% of squirrel and
  rodent images (confidence: 2.0%); both AUROCs are below 0.5. Exploratory
  check: every unseen-camera image is far from the training images (mean
  distance 0.27-0.41 against 0.08 within training), whatever it shows, so the
  distance measures "new camera" rather than "new species"; squirrels and
  rodents are small (median box ~1% of the frame) and mostly predicted empty,
  dog, or rabbit. Recognising unknown species likely needs animal-centred
  features (for example crops from an animal detector).
- Tuning species: squirrel, rodent. Held out for the final test: skunk, bird,
  badger, fox, deer (deer has no development or final-test events). All were
  unseen during fine-tuning; **all but deer appear in ImageNet-1k pretraining
  classes**, so no evaluated unknown species is new to the backbone.

## Promotion and rollback (*planned*)

1. Freeze the model, calibration, and thresholds, then open the final test once
   and publish the result, including failures.
2. A retrained candidate replaces the deployed model only if it passes the same
   evaluation. The API reports model, preprocessing, and decision-policy
   versions; rollback redeploys the retained previous version.
