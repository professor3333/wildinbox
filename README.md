# WildInbox

**Find the wildlife. Skip the empty frames.**

A review system for trail-camera photos. It groups photos into capture events,
suggests a species for each event, and sends the photos it can't label
reliably to a human for review. See [`docs/requirements.md`](docs/requirements.md)
for the product contract.

> Status: early development. Data acquisition, evaluation splits, and a thin
> upload-to-result application are in place. **No model is trained yet:** the app
> runs a clearly labeled test predictor whose scores are meaningless.

## Setup

Requirements: [uv](https://docs.astral.sh/uv/) and Git. uv installs the pinned
Python version (3.12) and every dependency from `uv.lock`.

```bash
git clone https://github.com/professor3333/wildinbox.git
cd wildinbox
uv sync --locked          # create .venv with exact locked versions
cp .env.example .env      # local settings; contains no secrets
```

## Run the application

```bash
docker compose up -d --build --wait    # Postgres, Redis, SeaweedFS (S3), migrations, API, worker, UI
open http://localhost:8501             # review interface (API: http://localhost:8000)
uv run python scripts/smoke.py         # upload -> process -> results, end to end

# Register the trained model as an immutable release and make it the default
# (weights and policy artifact come from the training machine):
docker compose run --rm \
  -v "$PWD/models/finetune-e3-deep-balanced:/app/models/finetune-e3-deep-balanced:ro" \
  -v "$PWD/reports/calibration:/app/reports/calibration:ro" \
  worker wildinbox release register --activate
curl localhost:8000/version            # deploy check: release, weights, preprocessing, policy

uv run python scripts/demo.py          # the demo: upload the sample, review, export (docs/demo.md)

docker compose down                    # add -v to delete stored data
```

| Endpoint | Purpose |
|---|---|
| `POST /batches` | Multipart upload: `files` (JPEG/PNG) plus optional `metadata` JSON (`camera_id`, per-file `camera_id` / `captured_at` / `sequence_id`). Returns `202` with the batch and job id. Send `Idempotency-Key` to make retries safe; identical re-uploads are recognised without it. |
| `GET /batches/{id}` | Status, counts, progress, every failed file with its error, job lifecycle, pinned release |
| `GET /batches/{id}/export` | Current observations with provenance (`?format=csv` or `json`) |
| `GET /batches/{id}/images` | Every file with its validation and processing status |
| `GET /batches/{id}/view` | Batch-status page |
| `GET /jobs/{id}` | Job status, attempts, lease, next retry, error |
| `GET /events?batch_id=&camera_id=&disposition=&label=&reason=&reviewed=` | Paginated capture events with decisions, `total`, and `next_offset` |
| `GET /events/{id}` | Frames with status, raw and calibrated predictions, decision, review history |
| `POST /events/{id}/reviews` | Record a review (`confirmed` / `corrected` / `unresolved`); reviews are appended, never overwritten |
| `GET /version`, `GET /releases` | Active release and its versions; all registered releases |
| `GET /images/{id}/original` | The stored original file |

Full contract, job lifecycle, and recovery rules: [`docs/api.md`](docs/api.md).

**Review interface** (`http://localhost:8501`, or `uv run wildinbox ui` against a
running API). It talks only to the API, so every review takes the same
validated, append-only path.

| Page | What it does |
|---|---|
| Review queue | Events with their frames, suggestion, confidence, and why they need review. Accept the suggestion in one click, pick another label, type an unsupported species, or mark "can't tell"; the review outcome (confirmed / corrected / unresolved) follows from the choice. |
| Last night's visitors | The representative frame of every animal event in one night (18:00-06:00, camera time), opening on the newest night with activity. Unreviewed labels are marked as suggestions. |
| Upload | Upload a memory card's photos with a camera name and follow processing live; files that could not be used are listed with their errors. |
| Export | Download the observation CSV with provenance. |

**Processing.** Workers score images in bounded chunks under a lease and
commit progress per chunk; events and decisions are written only when the
whole batch is scored. A killed worker's job is recovered when its lease
expires and resumes where it stopped, without duplicates; failures retry with
backoff up to three attempts, then fail with a recorded error. Each job is
pinned to the release it was created with. Demonstration with a real worker
kill: `uv run python scripts/make_sample_batch.py` then
`uv run python scripts/crash_demo.py` (results:
[`reports/serving/README.md`](reports/serving/README.md)).

**Limits** (configurable, see `.env.example`): 2,000 files and 1 GiB per batch,
20 MiB per file. Batches over the limit are rejected whole with `413`.
Unsupported, empty, oversized, or corrupt files get individual error records
and the rest of the batch is processed.

**Test predictor.** Before a trained release is activated, batches use release
`test-predictor-v0`: pseudo-random scores derived from file hashes, used to
exercise the pipeline. It is flagged `is_test` in the database, every API
response and the status page carry a warning, and its policy sends every event
to review. Nothing it produces is ML performance.

## Checks

The same commands CI runs:

```bash
uv run ruff check                                   # lint
uv run ruff format --check                          # formatting
uv run mypy src/                                    # types
uv run wildinbox validate-config configs/*.yaml     # run configs
uv run wildinbox export-schemas && git diff --exit-code docs/schemas
uv run pytest -m "not slow"                         # unit tests
```

The API tests need a PostgreSQL they may wipe; without one they are skipped
locally (CI always runs them):

```bash
docker run -d --name wildinbox-test-pg -p 55432:5432 -e POSTGRES_USER=wildinbox \
  -e POSTGRES_PASSWORD=wildinbox -e POSTGRES_DB=wildinbox postgres:16-alpine
export WILDINBOX_TEST_DATABASE_URL=postgresql+psycopg://wildinbox:wildinbox@localhost:55432/wildinbox
uv run pytest
```

## Layout

```
src/wildinbox/
  class_map.py       single source of class names and output order
  config.py          run configuration schema and loader
  preprocessing.py   the only image preprocessing (training and serving)
  schemas.py         shared contracts: image metadata, predictions, decisions, reviews
  settings.py        deployment settings from environment variables
  cli.py             `wildinbox` command
  ingestion/         dataset download, validation, inventory
  datasets/          events, event labels, splits, leakage checks
  api/               FastAPI app, upload validation, HTML pages
  workers/           batch processing and job dispatch (Redis/RQ)
  ui/                review interface (Streamlit, talks to the API)
  inference/         model releases; the test predictor
  policy/            decision policies
  storage/           PostgreSQL models, object storage
  training/          frozen-embedding baseline, EfficientNet-B0 fine-tuning
  evaluation/        evaluation reports, experiment comparison, calibration
  monitoring/        later stage
configs/             run configurations (YAML)
tests/
migrations/          Alembic database migrations
scripts/smoke.py     end-to-end check against a running deployment
scripts/demo.py      the demo flow on the committed sample batch
samples/cct-dev/     37-photo sample batch (Caltech Camera Traps, CDLA-Permissive)
docs/                requirements, dataset rules, model card, generated JSON Schemas
reports/             committed evaluation, experiment, and calibration reports
```

## Configuration

A run configuration fixes the dataset version, model architecture, seed,
preprocessing, supported classes (in model output order), and decision
thresholds. Validate one with:

```bash
uv run wildinbox validate-config configs/example.yaml
```

Invalid files fail with the file path and each bad field, e.g.
`thresholds.empty_filter: Input should be less than or equal to 1 (got 1.5)`.
Unknown fields are rejected, so a typo can't silently fall back to a default.

`configs/example.yaml` uses **placeholder** species and an unpinned dataset
manifest until the dataset stage selects the supported species from
training-set counts. Automatic filtering and acceptance are disabled by default.

## Data acquisition (CCT20)

```bash
uv run wildinbox data acquire      # download -> extract -> ingest -> lock check -> report
```

or step by step: `data download`, `data ingest`, `data report`. Sources, sizes,
and MD5s are pinned in [`configs/sources/cct20.yaml`](configs/sources/cct20.yaml).

- **Resumable.** Downloads go to `*.part` and continue from the last byte on
  re-run (HTTP Range). A file is only renamed to its final name after its size
  and MD5 match; extraction writes each file under a temporary name first.
- **Idempotent.** Re-running ingestion upserts by source id and reuses checks
  for unchanged files; the same inputs always give the same manifest version.
- **Every record accounted for.** Each source image ends up `accepted`,
  `quarantined` (missing/unreadable file, no annotation, unknown category,
  conflicting annotations, missing camera or sequence), or `excluded`
  (exact duplicate, configured exclusion), with reasons.
- **Never silently empty.** A label is assigned only to accepted records with
  annotations; the inventory database enforces this with CHECK constraints.
- **Originals preserved.** Raw image and annotation records are stored
  unchanged; normalized labels live in separate columns with their rule.

Outputs (under `data/`, not committed): the SQLite inventory
`data/inventory/cct20.sqlite` and the manifest
`data/manifests/<version>.jsonl.gz`. Committed: the pin
[`manifests/cct20.lock.json`](manifests/cct20.lock.json) (manifest version,
archive checksums, counts) and the
[data-quality report](reports/data_quality/cct20/README.md). If a re-run
produces a different inventory, `data ingest` fails until you inspect the
change and pass `--update-lock`.

## Events, labels, and splits

```bash
uv run wildinbox dataset build     # events -> ground truth -> species -> partitions -> leakage checks
```

Groups images into capture events, applies conservative event labels, selects
supported species from the training partition, and assigns camera-disjoint
partitions (train, calibration, policy validation, locked final test). The
build fails unless every leakage check passes. Rules and rationale:
[`docs/dataset.md`](docs/dataset.md); results:
[`reports/splits/cct20/README.md`](reports/splits/cct20/README.md); pin:
[`manifests/cct20-splits-v1.lock.json`](manifests/cct20-splits-v1.lock.json).

Supported classes: `empty`, bobcat, cat, coyote, dog, opossum, rabbit, raccoon.

## Baseline and evaluation

```bash
uv run wildinbox baseline train      # frozen EfficientNet-B0 embeddings + logistic regression
uv run wildinbox evaluate            # development partitions -> reports/baseline/
```

The baseline is the reference every later model must beat. Evaluation reports
image-level macro-F1 and per-species precision/recall, confusion matrices,
event-level decisions under the conservative policy (animal events wrongly
filtered as empty, accepted-label precision, review load), per-camera and
day/night slices, inference time and memory, and an error gallery. It only
reads training and development partitions; the locked final test is refused in
code. Results: [`reports/baseline/README.md`](reports/baseline/README.md).
Experiment runs are tracked in MLflow (`mlruns/`, local, not committed):
`uvx mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db` (the project
installs only the lightweight tracking client).

## Fine-tuning experiments

```bash
uv run wildinbox finetune inspect-augmentation --config configs/experiments/finetune-e1-top-lossweight.yaml
uv run wildinbox finetune train --config configs/experiments/finetune-e1-top-lossweight.yaml
uv run wildinbox evaluate --model models/finetune-e1-top-lossweight \
  --report-dir reports/experiments/finetune-e1-top-lossweight
uv run wildinbox compare reports/experiments/finetune-* --out reports/experiments/comparison.md
```

Each experiment is one YAML file in `configs/experiments/`. Models train only
on the training partition, with box-safe crops (a crop may not cut more than
10% of any annotated animal). The selection rule in
[`configs/experiments/selection.yaml`](configs/experiments/selection.yaml) was
committed before any results existed. Results, failure analysis, and the
decision: [`reports/experiments/README.md`](reports/experiments/README.md);
model card: [`docs/model_card.md`](docs/model_card.md).

## Calibration and operating point

```bash
uv run wildinbox unfamiliar     # unfamiliar-input score vs confidence -> reports/unfamiliar/
uv run wildinbox calibrate      # calibration, decision policy, operating point -> reports/calibration/
uv run wildinbox replay         # recompute every saved event decision (no data download needed)
```

`unfamiliar` evaluates a distance-from-training-images score against ordinary
confidence under
[`configs/experiments/unfamiliar.yaml`](configs/experiments/unfamiliar.yaml):
squirrel and rodent tune it; skunk, bird, badger, fox, and deer are held out
for the final test. `calibrate` fits temperature scaling on the calibration
cameras, then applies the rule in
[`configs/experiments/operating_point_v2.yaml`](configs/experiments/operating_point_v2.yaml)
(committed before any v2 result existed) with the versioned `conservative/v1`
policy on the policy-validation cameras: a threshold is enabled only if the worst case of its 95% interval
meets the target (at most 2% of animal events filtered as empty; at least 95%
of accepted labels correct). A recorded deviation
([`configs/experiments/operating_point_deviation.yaml`](configs/experiments/operating_point_deviation.yaml))
can only switch automation off. Species also need their own precision
evidence before they can be accepted. It writes the versioned policy artifact
([`reports/calibration/policy.json`](reports/calibration/policy.json)) and every
development event's saved predictions and decisions, which `replay` (and CI)
reproduces exactly. Events with a pending or failed frame are never filtered;
review reasons are machine-readable. Current release: **automatic filtering and
automatic acceptance are both off**; every event goes to review with its
suggested label. Results and error-versus-coverage curves:
[`reports/calibration/README.md`](reports/calibration/README.md),
[`reports/unfamiliar/README.md`](reports/unfamiliar/README.md).

## Final test

```bash
uv run wildinbox final-test     # once, under configs/experiments/final_test.yaml
```

The locked final test (9 cameras never used for training, calibration, or
thresholds) was opened once after every artifact was frozen and pinned by hash
in a protocol committed beforehand. `final-test` is the only code path that
reads it; a re-run must reproduce the recorded numbers exactly. Results:
[`reports/final_test/README.md`](reports/final_test/README.md). In short: E3
macro-F1 0.447 [0.435, 0.458] vs 0.285 for the frozen-embedding baseline (0.747
on held-out sequences from training cameras); as released, every event goes to
review; the 50% review-reduction target is not met.

## Update cycle

```bash
uv run wildinbox snapshot build          # reviewed events -> versioned training snapshot
uv run wildinbox finetune train --config configs/experiments/finetune-e3-update1.yaml
uv run wildinbox update gate             # candidate vs deployed, pre-registered checks
uv run python scripts/release_rollback.py
```

Reviewed corrections become a versioned snapshot (originals checked by
SHA-256; each camera split at its median time into training data and a
holdout). A candidate trained with the deployed recipe is released only if it
passes the gate in
[`configs/experiments/update_cycle.yaml`](configs/experiments/update_cycle.yaml);
releases are immutable, activation is append-only, and rollback is
`wildinbox release activate <previous>`. One full cycle, with simulated
reviews: [`reports/update/README.md`](reports/update/README.md).

## Data and weights

Downloaded datasets, uploads, thumbnails, and trained weights are never
committed (see `.gitignore` and `data/README.md`). They are reproduced from a
pinned dataset manifest. The one exception is the 37-photo demo sample in
[`samples/cct-dev`](samples/cct-dev), committed with its license and
attribution so the demo runs without downloading the dataset.

Dataset: [Caltech Camera Traps](https://lila.science/datasets/caltech-camera-traps),
released under the Community Data License Agreement (permissive variant).
