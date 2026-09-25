# Observation export format

`GET /batches/{id}/export?format=csv` (the review interface's **Export** page)
returns one row per **capture event**: the photos from one camera trigger. A
row is not an individual animal, and rows cannot be summed into population
counts. `?format=json` returns the same fields as
`{"batch_id": ..., "observations": [...]}` with lists and nulls kept as JSON.

Every event is exported, including events nobody has reviewed yet and events a
reviewer left unresolved.

## Columns

| Column | Meaning |
|---|---|
| `event_id` | The capture event's id (stable; use it to join exports). |
| `batch_id` | The upload (memory card) the event came from. |
| `camera_id` | Camera name from the upload metadata; empty if none was given. |
| `start_at`, `end_at` | First and last capture time of the event's photos, camera local time (ISO 8601); empty if the photos had no time. |
| `frames` | Number of photos in the event. |
| `filenames` | The photos' original file names, separated by `;`. |
| `observation` | **The current label**: the latest review's label if a person reviewed the event; otherwise the automatic label if automation decided it; otherwise empty. `empty` means no animal. A species outside the model's list appears as the reviewer typed it (lowercase). |
| `label_source` | Where `observation` comes from: `review`, `automatic`, `pending_review` (nobody has reviewed it yet), or `unresolved` (a reviewer could not tell). |
| `review_outcome` | The latest review: `confirmed` (same as the suggestion), `corrected` (different), `unresolved`; empty if not reviewed. |
| `reviewer`, `reviewed_at`, `review_id` | Who made the latest review, when (UTC), and its id. Earlier reviews stay in the event's history (`GET /events/{id}`). |
| `disposition` | What the system decided: `needs_review`, `likely_empty` (filtered automatically), or `species_identified` (labeled automatically). |
| `suggested_label` | The model's suggestion, never overwritten by reviews. |
| `confidence` | The calibrated confidence of that suggestion (0-1). |
| `reasons` | Why the event needed review, separated by `;`: `low_confidence`, `conflicting_frames`, `possible_unknown`, `processing_failure`, `species_not_validated`, `automation_disabled`. |
| `audit_selected` | `True` if this automatically handled event was sampled for a human audit. |
| `audit_rule` | For automatic decisions, how audit samples were chosen, e.g. `sha256-uniform(rate=0.05, seed=wildinbox-audit-v1)`; recomputable from the event id. |
| `model_release_id` | The immutable release that made the suggestion, e.g. `finetune-e3-deep-balanced@7a25aea97c76`. |
| `weights_sha256` | SHA-256 of that release's model weights. |
| `preprocessing_version`, `calibration_version`, `policy_version` | The versions of image preprocessing, score calibration, and decision policy behind the suggestion. |
| `exported_at` | When this export was made (UTC). |

## Reading it

- For a species list, use rows with `label_source = review`; add `automatic`
  rows only if you trust the release's automatic decisions (the model card says
  which are enabled; as released, none are).
- `pending_review` rows still need a person; `unresolved` rows had one who
  could not decide.
- The provenance columns identify exactly which model and settings produced a
  suggestion, so exports from different releases can be told apart.
