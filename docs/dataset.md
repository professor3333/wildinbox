# Dataset: events, labels, and evaluation splits

How WildInbox turns the validated CCT20 inventory into capture events,
event ground truth, supported classes, and leakage-resistant partitions.

- Build: `uv run wildinbox dataset build`
- Spec: [`configs/splits/cct20.yaml`](../configs/splits/cct20.yaml)
- Taxonomy: [`configs/taxonomy/cct20.yaml`](../configs/taxonomy/cct20.yaml)
- Results and leakage checks: [`reports/splits/cct20/README.md`](../reports/splits/cct20/README.md)
- Pin: [`manifests/cct20-splits-v1.lock.json`](../manifests/cct20-splits-v1.lock.json)

## Grouping rules

Events are **capture events**, not individual animals: one animal returning
several times produces several events.

| Rule | Used for | Definition |
|---|---|---|
| `sequence_id/v1` | Public data (CCT20) | Images sharing a sequence id form one event. |
| `time_gap/v1(gap_s=5)` | Uploads without sequence ids | Same camera, ordered by timestamp; a new event starts when the gap to the previous image exceeds 5 s. No timestamp or no camera means a single-image event. Different cameras are never grouped. |

The 5 s default comes from CCT20: every within-sequence gap is at most 3 s, so
the rule never splits a trigger burst. But 19% of consecutive sequences on the
same camera start within 5 s (the camera re-triggers while the animal is still
in view), and those get merged. For uploads that means fewer, longer events to
review; time-grouped event counts are not comparable with sequence counts. The
split report measures this on all CCT20 images.

Every event row records its grouping rule and the ids of all its frames.

## Event ground truth

Applied in order:

1. **Quarantined frame -> event excluded** (`contains_quarantined_frame`).
   Its ground truth can't be trusted.
2. **Ground truth uses every non-quarantined frame's annotations**, including
   exact duplicates that are dropped from the image set, so removing a
   duplicate image never turns an animal event into an empty one. An event
   whose frames were all dropped as duplicates is excluded (`no_usable_images`);
   its images exist elsewhere.
3. **Animal-containing if at least one frame contains an animal** (taxonomy
   kind `animal`).
4. **Exactly one animal species -> that species.** More than one -> role
   `mixed_species`, label unresolved. No resolution rule is applied; these
   events go to review.
5. **No animal but a non-animal category** (e.g. `car`) -> role `non_animal`.
   Neither an animal positive nor an empty negative.
6. **`empty` only when every frame is explicitly annotated empty.** Images
   with no annotation were already quarantined at ingestion, so they can't
   reach this rule.

Each event then gets a role: `supported_species`, `unsupported_animal`,
`mixed_species`, `non_animal`, or `empty`.

## Supported classes

Chosen from the **training partition only**, over single-species events:
at least 500 training events, and at least 3 training cameras with at least
20 events each. The thresholds were set before the post-split counts were
known and have not been tuned to reach a target number of species.

Result: `empty` plus bobcat, cat, coyote, dog, opossum, rabbit, raccoon.
Squirrel (438 training events) is just below the cut.

Animals not selected remain `unsupported_animal` events. **They are never used
as empty (negative) examples**; they are the evaluation cases for unsupported
inputs.

## Which images may fit the model

An image is a fit example only if its event is in the training partition, its
role is `supported_species` or `empty`, and its own annotation equals the event
label. Excluded as a result:

- unsupported, mixed-species, and non-animal events;
- empty-annotated frames inside animal events;
- everything outside the training partition.

`images.jsonl.gz` records `use_for_fit` and the reason for every image.

## Partitions

Assigned by camera, so every sequence stays whole.

| Partition | Permitted use | Cameras |
|---|---|---|
| `train` | Fit model parameters and training-derived statistics | 38, 120, 43, 33, 88, 61, 115 (minus the diagnostic sample) |
| `seen_camera_diagnostic` | Only the seen-camera vs unseen-camera comparison | 15% of training-camera sequences, sampled by hashed sequence id |
| `calibration` | Fit score calibration | 51, 108 |
| `policy_validation` | Choose thresholds, aggregation rules, operating points | 125, 90 |
| `final_test` | Measure the finished system once, after decisions are frozen | 46, 0, 100, 78, 105, 7, 130, 28, 40 |

- Final-test cameras are the published `trans_test` cameras, unchanged, so we
  did not choose them. They are held out of all development.
- Development cameras are separate from training cameras, and calibration
  cameras are separate from policy-validation cameras.
- Unsupported species on development cameras can be used to tune unfamiliar-
  input thresholds; those on final-test cameras are reserved for the final
  measurement.

### Deviations from the published CCT20 split

1. The published `train` and `cis_val` files share 224 sequences (sequence
   leakage). Partitions here are assigned per whole sequence.
2. The published `trans_val` has a single camera, so cameras 51, 108, and 90
   move from the published cis cameras into development.
3. The published `cis_test` uses training cameras, so it is not treated as a
   test. A hashed 15% sample of training-camera sequences serves as the
   seen-camera diagnostic instead; all of `cis_test` would have taken about
   half of the training events.

## Leakage checks

`wildinbox dataset build` fails, and does not update the lock, unless all of
these pass:

- no sequence id in more than one partition;
- no exact duplicate image (SHA-256) across partitions;
- no suspected near-duplicate pair across partitions;
- no final-test camera in any other partition;
- development cameras separate from training cameras;
- calibration and policy-validation cameras separate;
- diagnostic cameras are a subset of training cameras;
- every inventory record in exactly one event;
- every event in one partition or excluded with a reason;
- fit examples are training-only, supported or all-empty, and match their event label.

The build also refuses to run if the inventory doesn't match the pinned
manifest version, if any camera is unassigned or assigned twice, or if any
source category is missing from the taxonomy.

## Reproducibility

Every decision is in a committed file: grouping rule, camera assignment,
diagnostic salt and fraction, selection thresholds, and taxonomy. The build is
deterministic; its output hash is the split version pinned in the lock file,
and a re-run that produces anything different fails until reviewed with
`--update-lock`.

## Limitations

- **Few locations:** 7 training, 2 calibration, 2 policy-validation, and 9
  final-test cameras. Calibration and threshold choices rest on two cameras
  each. Final-test results must be reported per camera, with camera-level variation.
- **Low empty rate:** CCT20 events are mostly animals. Empty-filtering results
  will not reflect a real memory card (~70% empty in full Caltech Camera Traps).
- **Uneven species coverage:** some supported species are rare on development
  or test cameras (e.g. cat has no calibration events), so per-species results
  there will be noisy.
