# WildInbox — Requirements (v1)

Stage 1 deliverable: product contract, supported inputs, terminology, and
acceptance targets. Everything later is built and tested against this file.

## Scope of the first release

- **One workspace** (one organization or one person). No multi-tenant accounts.
- **Batch uploads** of still images from a collected memory card. No live
  camera hardware, no streaming, no video.
- **About six supported species plus `empty`.** The actual species are chosen
  in the dataset stage from training-set counts in Caltech Camera Traps
  (CCT20); they are recorded here once chosen.
- **One suggested species per event.** Mixed-species events stay reviewable.
- Species outside the supported set are never relabeled as `empty`.

Supported species: **bobcat, cat, coyote, dog, opossum, rabbit, raccoon**,
plus `empty`. Chosen from the training partition by a rule fixed before the
counts were known (at least 500 single-species training events, and at least 3
training cameras with at least 20 events each); seven species met it, one more
than the planned "about six" ([dataset](dataset.md#supported-classes)).
Every other CCT20 animal (squirrel, skunk, bird, rodent, badger, fox, deer) is
an **unsupported species**: its ground-truth label is kept, it is never used as
an `empty` example, and it is never fit.

## Core workflow

**Upload → validate → group → predict → review → correct → export.**

1. **Upload** — user submits a batch of images plus optional metadata. The API
   returns a job ID immediately.
2. **Validate** — each file is decoded and checked; failures are quarantined,
   not silently dropped.
3. **Group** — valid images are grouped into capture events.
4. **Predict** — each image is classified; predictions are saved per image,
   then aggregated per event under the decision policy.
5. **Review** — events needing review are queued for a human.
6. **Correct** — the human confirms or changes the label, or marks it
   unresolved. The original prediction is kept alongside the correction.
7. **Export** — the user downloads the observation log.

## Terminology

| Term | Meaning |
|---|---|
| Image | One uploaded photograph |
| Capture event | Related frames from one camera trigger or a documented time-grouping rule |
| Suggested label | The model's prediction |
| Confirmed label | A human-reviewed annotation |
| Automatically filtered | Excluded from the normal review queue, with originals still accessible |
| Needs review | The system cannot make an acceptable automatic decision |
| Unresolved | A reviewer also cannot determine the label |

**Event counts represent capture events, not distinct animals.** An animal
returning repeatedly may produce several events. WildInbox does not identify
individual animals or estimate population counts.

### Event dispositions

Every event receives exactly one:

| Disposition | Condition |
|---|---|
| Likely empty (automatically filtered) | All usable frames strongly support `empty` above the validated empty threshold, **and** automatic filtering is enabled for this operating point |
| Supported species identified | Frames agree on one supported species above the validated acceptance threshold, no frame is flagged as unfamiliar (only possible once an unfamiliar-input score is adopted; none is in v1), **and** automatic acceptance is enabled |
| Needs review | Anything else: low confidence, any frame suggesting an animal in an otherwise empty-looking event, conflicting species, possible unsupported input, or automation not enabled |

Automatic filtering and automatic acceptance are **off by default** and are
enabled only for operating points supported by evaluation. Until then every
event goes to review with its suggestion attached.

A random sample of automatically handled events is also sent for audit, so
confident mistakes are measured.

### Status in the v1 release

**Implemented.**
- The released policy (`conservative/v2`) has automatic filtering and
  automatic acceptance both off. Every event is *needs review* with reason
  `automation_disabled`, carrying its suggestion and confidence.
- Unsupported species keep their ground-truth labels in the dataset manifest
  and are never relabeled or trained as `empty`. Reviewers can name an
  unsupported species or mark an event *unresolved*, and the export keeps it.
- The policy sends any frame flagged as unfamiliar to review
  (`possible_unknown`), but serving computes no unfamiliar-input score, so v1
  never raises that reason.

**Demonstrated** (locked final test, 594 unsupported-species events): none was
automatically accepted as a known species, and every animal event, supported
or not, stayed out of the filtered bucket. Both follow from nothing being
automated. They are not evidence that the model recognises unfamiliar species.

**Not established.**
- **Recognising an unfamiliar species at inference.** The distance-based
  score was evaluated and not adopted: it mostly measures "new camera", not
  "new species" ([unfamiliar inputs](../reports/unfamiliar/README.md)).
  Confidence alone is not evidence that an input belongs to a supported
  class.
- **Protection once automation is enabled.** The one evaluated operating
  point, an empty filter at 0.65 (not released), would have filtered 17 of the
  594 unsupported-species events as empty (97.14% retained, below the 98%
  target) ([final evaluation](../reports/final_evaluation/README.md)).
  Today the protection for unsupported species is that a person reviews every
  event.

## Supported-input specification

### Files

| Property | Requirement |
|---|---|
| Formats | JPEG (`.jpg`, `.jpeg`) and PNG (`.png`) |
| Validity | The file must fully decode as an image |
| Content type | Detected from file contents, not trusted from the extension |
| Originals | Stored unchanged; thumbnails generated separately |
| Duplicates | Identified by content hash; an exact duplicate within the workspace is reported and not processed twice |

Any other file type, or a file that fails to decode, is **quarantined**: it is
recorded as a failed file with a reason, shown to the user, and does not stop
the rest of the batch.

### Metadata (optional)

| Field | Use |
|---|---|
| Camera ID | Grouping, per-camera filtering and monitoring |
| Timestamp | Grouping, timeline, "last night's visitors" |
| Sequence ID | When supplied (e.g. public dataset), used directly for grouping |

### Grouping rule

1. If a sequence ID is supplied, images sharing it form one capture event.
2. Otherwise, images from the same camera ordered by timestamp form one event
   while the gap between consecutive images is at most **G** seconds. **G** is
   editable by the user; its default is set from the dataset's real sequences.
3. An image with no usable timestamp forms its own single-image event.

## Required behavior — acceptance-gate scenarios

**A clear animal image (supported species).** It is validated, stored with a
thumbnail, grouped into its capture event, and classified. If every usable
frame agrees on the same supported species above the validated acceptance
threshold and automatic acceptance is enabled, the event is labeled with that
species and shown in the timeline and "last night's visitors". Otherwise it
goes to *needs review* with the suggested label. Either way, the user can
correct it; the original prediction is retained.

**An empty sequence.** Its frames are grouped into one event. If all usable
frames strongly support `empty` and automatic filtering is enabled, the event
is *automatically filtered*: removed from the normal review queue but still
viewable and recoverable, and eligible for the random audit sample. If any
frame suggests an animal, or filtering is not enabled, it goes to *needs
review*.

**An unfamiliar species.** The system does not claim to recognise it. If its
confidence (or, once one is adopted, its unfamiliar-input score) fails the
policy, the event goes to *needs review*; the reviewer labels it manually or
marks it *unresolved*. Its ground-truth label is never relabeled as `empty`.
How often such inputs are wrongly auto-accepted as a known species is measured
in evaluation, not assumed to be zero. In v1 no unfamiliar-input score is used
and automation is off, so these events reach review because every event does,
not because they were recognised as unfamiliar (see *Status in the v1
release*).

**A corrupted file.** It fails decoding during validation, is quarantined with
a reason, and is listed as a failed file for the batch. It is never passed to
the model and never creates an event. The remaining files process normally.

**An interrupted batch.** Progress is tracked in PostgreSQL per file. When a
worker stops mid-batch, the abandoned job is recovered and resumed; files
already processed are not processed again, transient failures are retried,
and permanently failed files are reported. Processing is idempotent, so the
result has no lost inputs and no duplicate events.

## Acceptance targets

Proposed targets, not achieved results. They are measured on **unseen camera
locations**.

| Measure | Target / evaluation method |
|---|---|
| Animal-event retention | ≥98% of animal-containing test events kept outside the automatically filtered bucket |
| Accepted species precision | ≥95% correct among automatically accepted species labels; report the fraction auto-labeled |
| Manual review reduction | 50% fewer reviewed events, including audit work, while meeting retention; measured against an already sequence-grouped manual workflow |
| Species quality | Macro-F1, per-class recall, confusion matrices |
| Unsupported-input handling | Rate at which withheld species receive an incorrect, automatically accepted known-species label |
| Operational performance | On declared hardware, 1,000 resized images processed within 10 minutes; metadata API p95 latency < 500 ms |
| Reliability | Worker restart during processing with no lost inputs or duplicate results |

## Milestones

- **Engineering complete** — uploads, processing, review, monitoring, and
  model updates work reliably.
- **Ready for automatic decisions** — evaluation supports automatically
  filtering empty events or accepting species labels at the chosen error
  limits.

The project is credible even if some predictions still require human
review; what matters is showing where automation works and enforcing those
boundaries.
