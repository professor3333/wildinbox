# Split report: `cct20-splits-v1-7de740aba75f`

Inventory `cct20-396eb43cc6ce` · grouping rule `sequence_id/v1` · spec `configs/splits/cct20.yaml` · rules [`docs/dataset.md`](../../../docs/dataset.md)

## Leakage checks

| Check | Result | Detail |
|---|---|---|
| no sequence id in more than one partition | PASS | 22710 sequences, each in exactly one partition |
| no exact image duplicate (SHA-256) across partitions | PASS | 57855 distinct image hashes checked |
| no suspected near-duplicate pair across partitions | PASS | 0 inventory near-duplicate pairs checked |
| final-test cameras used in no other partition | PASS | 9 final-test cameras, none elsewhere |
| development cameras separate from training cameras | PASS | 4 development cameras, none used for training |
| calibration and policy-validation cameras separate | PASS | camera-disjoint, so their events are disjoint too |
| seen-camera diagnostic only uses training cameras | PASS | diagnostic cameras are a subset of training cameras |
| every inventory record belongs to exactly one event | PASS | 57864 records, 22719 events |
| every event is in exactly one partition or excluded with a reason | PASS | 22710 events kept, 9 excluded with reasons |
| fit examples are training-only, supported or all-empty, frame label = event label | PASS | unsupported, mixed, and non-animal events never become fit examples |

## Partitions

| Partition | Permitted use | Cameras | Events | Images | Fit images |
|---|---|---|---|---|---|
| train | Fit model parameters and training-derived statistics | 33, 38, 43, 61, 88, 115, 120 | 10135 | 24894 | 20287 |
| seen_camera_diagnostic | Diagnostic only: seen-camera vs unseen-camera comparison | 33, 38, 43, 61, 88, 115, 120 | 1746 | 4313 | 0 |
| calibration | Fit score calibration | 51, 108 | 908 | 2641 | 0 |
| policy_validation | Choose thresholds, aggregation rules, operating points | 90, 125 | 939 | 2732 | 0 |
| final_test | Measure the finished system once, after decisions are frozen | 0, 7, 28, 40, 46, 78, 100, 105, 130 | 8982 | 23275 | 0 |

Events by role:

| Partition | empty | supported_species | unsupported_animal | mixed_species | non_animal |
|---|---|---|---|---|---|
| train | 534 | 7061 | 1237 | 7 | 1296 |
| seen_camera_diagnostic | 101 | 1235 | 203 | 3 | 204 |
| calibration | 184 | 521 | 203 | 0 | 0 |
| policy_validation | 86 | 708 | 123 | 22 | 0 |
| final_test | 715 | 6548 | 594 | 12 | 1113 |

## Events per label

**Bold** labels are the supported classes.

| Label | train | seen_camera_diagnostic | calibration | policy_validation | final_test |
|---|---|---|---|---|---|
| **opossum** | 1984 | 347 | 213 | 186 | 1761 |
| **raccoon** | 577 | 107 | 46 | 62 | 1825 |
| car | 1296 | 204 | 0 | 0 | 1113 |
| **coyote** | 1310 | 228 | 58 | 39 | 964 |
| **bobcat** | 563 | 87 | 101 | 249 | 860 |
| **rabbit** | 1222 | 245 | 95 | 14 | 280 |
| **empty** | 534 | 101 | 184 | 86 | 715 |
| **cat** | 810 | 131 | 0 | 121 | 452 |
| **dog** | 595 | 90 | 8 | 37 | 406 |
| squirrel | 438 | 60 | 134 | 99 | 343 |
| bird | 297 | 62 | 47 | 3 | 80 |
| skunk | 203 | 30 | 4 | 21 | 144 |
| rodent | 199 | 35 | 17 | 0 | 18 |
| deer | 92 | 12 | 0 | 0 | 0 |
| (mixed) | 7 | 3 | 0 | 22 | 12 |
| badger | 2 | 3 | 1 | 0 | 8 |
| fox | 6 | 1 | 0 | 0 | 1 |

## Supported species selection

Rule, applied to single-species events in the **training partition only**: at least 500 training events and at least 3 training cameras with >= 20 events each.

Supported classes (model output order): `['empty', 'bobcat', 'cat', 'coyote', 'dog', 'opossum', 'rabbit', 'raccoon']`

| Species | Training events | Training cameras | Selected | Reason |
|---|---|---|---|---|
| opossum | 1984 | 5 | yes | meets rule |
| coyote | 1310 | 5 | yes | meets rule |
| rabbit | 1222 | 4 | yes | meets rule |
| cat | 810 | 3 | yes | meets rule |
| dog | 595 | 6 | yes | meets rule |
| raccoon | 577 | 5 | yes | meets rule |
| bobcat | 563 | 3 | yes | meets rule |
| squirrel | 438 | 4 | no | 438 < 500 training events |
| bird | 297 | 3 | no | 297 < 500 training events |
| skunk | 203 | 3 | no | 203 < 500 training events |
| rodent | 199 | 1 | no | 199 < 500 training events; 1 < 3 cameras |
| deer | 92 | 2 | no | 92 < 500 training events; 2 < 3 cameras |
| fox | 6 | 0 | no | 6 < 500 training events; 0 < 3 cameras |
| badger | 2 | 0 | no | 2 < 500 training events; 0 < 3 cameras |

Animals not selected stay identifiable as `unsupported_animal` events. They are never used as empty (negative) examples; they are evaluation cases for unsupported inputs. Species absent from training appear in no row above.

## Taxonomy mapping

| Source category | Kind | Role in this build |
|---|---|---|
| badger | animal | unsupported animal |
| bird | animal | unsupported animal |
| bobcat | animal | supported class |
| car | non_animal | non_animal |
| cat | animal | supported class |
| coyote | animal | supported class |
| deer | animal | unsupported animal |
| dog | animal | supported class |
| empty | empty | supported class |
| fox | animal | unsupported animal |
| opossum | animal | supported class |
| rabbit | animal | supported class |
| raccoon | animal | supported class |
| rodent | animal | unsupported animal |
| skunk | animal | unsupported animal |
| squirrel | animal | unsupported animal |

## Published benchmark partitions and deviations

Events by the published CCT20 file(s) their frames came from:

| Published file(s) | train | seen_camera_diagnostic | calibration | policy_validation | final_test |
|---|---|---|---|---|---|
| cis_test | 5032 | 821 | 463 | 102 | 0 |
| cis_val | 852 | 169 | 124 | 52 | 0 |
| cis_val+train | 176 | 39 | 7 | 2 | 0 |
| train | 4075 | 717 | 314 | 194 | 0 |
| trans_test | 0 | 0 | 0 | 0 | 8982 |
| trans_val | 0 | 0 | 0 | 589 | 0 |

Deviations from the published split:

1. **224 sequences have frames in more than one published file** (`train` and `cis_val`): sequence leakage in the published split. Partitions here are assigned per whole sequence, so those files are pooled on training cameras.
2. **Development cameras 51, 90, 108 come from the published cis cameras**, because the published `trans_val` has a single camera. All of their published train/val/test images move to development.
3. **Published `cis_test` is not used as a test**: its cameras are training cameras. Instead a deterministic sample of whole training-camera sequences is a seen-camera diagnostic (in-distribution performance only). Using all of `cis_test` would have removed about half of the training events.
4. **Policy validation uses cameras 90, 125**: the published `trans_val` camera(s) plus development cameras listed above.
5. **`trans_test` is the locked final test, unchanged.**

## Excluded events

| Event | Reason | Frames |
|---|---|---|
| cct20:700fb966-5567-11e8-a777-dca9047ef277 | no_usable_images | 1 |
| cct20:700ffa66-5567-11e8-bf31-dca9047ef277 | no_usable_images | 1 |
| cct20:700ffc59-5567-11e8-9ea9-dca9047ef277 | no_usable_images | 1 |
| cct20:70100ccc-5567-11e8-9115-dca9047ef277 | contains_quarantined_frame | 1 |
| cct20:70100d1c-5567-11e8-bc0b-dca9047ef277 | contains_quarantined_frame | 1 |
| cct20:70100da8-5567-11e8-a35b-dca9047ef277 | no_usable_images | 1 |
| cct20:7010106e-5567-11e8-b5a2-dca9047ef277 | no_usable_images | 1 |
| cct20:701013e8-5567-11e8-a0e8-dca9047ef277 | no_usable_images | 1 |
| cct20:7010178a-5567-11e8-a3d5-dca9047ef277 | no_usable_images | 1 |

2 kept event(s) carry notes, e.g. `cct20:6f14a363-5567-11e8-a98d-dca9047ef277`: 1 empty-annotated frame(s) in an animal event. Their empty-annotated frames are not used as fit examples.

## Upload grouping rule evaluated on CCT20

Uploads without sequence ids use `time_gap/v1(gap_s=5)`. Applied to all 57864 CCT20 images (ignoring their sequence ids):

|  | Count |
|---|---|
| True sequences | 22719 |
| Time-gap events | 18369 |
| Events matching exactly one whole sequence | 15159 |
| Events merging several sequences (back-to-back triggers) | 3210 |
| Sequences split across events | 0 |

The rule never splits a trigger burst but merges re-triggers that start within the gap. For uploads this errs toward fewer, longer events to review; event counts from time grouping are not comparable to sequence counts.

## Limitations

- **Few locations.** 20 cameras in total: 7 train, 2 calibration, 2 policy validation, 9 final test. Calibration and threshold choices rest on 2 cameras each and may not transfer; final-test results must be reported per camera, with camera-level variation.
- **Low empty rate.** CCT20 events are mostly animals; empty-filtering numbers here will not reflect a real memory card (~70% empty in full Caltech Camera Traps).
- **Uneven species coverage.** Some supported species are rare on development or test cameras (see Events per label), so per-species results there will be noisy.
- **Non-animal triggers.** `car` events occur on one training and one test camera.
