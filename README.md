# WildInbox

**Find the wildlife. Skip the empty frames.**

[![ci](https://github.com/professor3333/wildinbox/actions/workflows/ci.yml/badge.svg)](https://github.com/professor3333/wildinbox/actions/workflows/ci.yml)

A review system for trail-camera photos. Upload a memory card; WildInbox groups
the photos into capture events, suggests a species for each event, says why an
event needs a human, and lets you review, correct, and export an observation
log with the provenance of every label.

```mermaid
flowchart TD
    U[Memory card upload / public dataset] --> A[FastAPI: validate files, build manifest]
    A --> S3[(Object storage, SeaweedFS S3:<br/>originals, thumbnails, release weights)]
    A --> PG[(PostgreSQL: batches, images, jobs,<br/>events, predictions, decisions, reviews)]
    PG --> Q[Redis / RQ queue]
    Q --> W[Workers: lease, bounded chunks,<br/>retries, stale-job recovery]
    R[Immutable release: weights, classes,<br/>preprocessing, calibration, policy] --> W
    W --> PG
    PG --> UI[Streamlit review interface]
    S3 --> UI
    UI --> H[Reviews: append-only corrections]
    H --> PG
    PG --> SN[Versioned training snapshot]
    S3 --> SN
    SN --> T[Train offline, MLflow tracking]
    T --> G[Pre-registered gate vs deployed model]
    G --> R
    W --> M[Monitoring: operations, label-free signals,<br/>review-based accuracy]
    H --> M
```

## The problem

Trail cameras fire on wind, heat, and animals walking out of frame, so a memory
card is mostly empty or repeated photos that someone still has to sort. About
70% of the full Caltech Camera Traps dataset is labeled empty. WildInbox aims to
cut that reviewing while never losing a real sighting, and the central question
is measured, not assumed: **how much manual review can be removed without losing
animals when a camera is somewhere new?**

## What it does today

- **Upload and validate** JPEG/PNG batches with optional camera, time, and
  sequence metadata; files are checked by content, duplicates recognised by
  hash, and every unusable file gets an explicit error.
- **Group** photos into capture events (supplied sequence ids, or camera and
  time gaps).
- **Suggest** a species per event with a fine-tuned EfficientNet-B0, calibrated,
  and decide each event with a versioned policy: likely empty, species
  identified, or needs review, with machine-readable reasons.
- **Review** in the browser: accept, correct, name an unsupported species, or
  mark "can't tell"; see last night's visitors; export a CSV with provenance.
- **Process reliably**: asynchronous workers with leases, bounded retries,
  stale-job recovery, and idempotent writes; a worker killed mid-batch loses
  nothing and duplicates nothing.
- **Release safely**: immutable model releases pinned per job, a pre-registered
  promotion gate, append-only activation, and one-command rollback.
- **Monitor** operations, label-free drift signals per camera, and accuracy
  measured from reviews, kept apart.

### Results, honestly

Measured once on the **locked final test** (9 cameras never used for training,
calibration, or thresholds), under a protocol committed beforehand
([report](reports/final_test/README.md)):

| | Fine-tuned (released) | Frozen-embedding baseline |
|---|---|---|
| Macro-F1, supported classes (95% CI) | **0.447** [0.435, 0.458] | 0.285 |
| Same model on held-out sequences of *training* cameras | 0.747 | 0.724 |

- **Automation is off.** No empty-filter or species-acceptance threshold held
  up on new cameras with the required confidence, so every event goes to
  review with its suggestion and reasons ([calibration](reports/calibration/README.md)).
- **The 50% review-reduction target is not met.** Grouping turns 2.6 photos
  into one event; the model itself removes no reviews as released.
- The gap between 0.747 and 0.447 is the new-camera problem this project set out
  to measure. Reviewing a camera helps that camera: one update cycle raised
  label quality on later photos of the reviewed cameras from 0.486 to 0.548
  ([update cycle](reports/update/README.md), reviews simulated from ground truth).

Supported classes: **empty, bobcat, cat, coyote, dog, opossum, rabbit,
raccoon**, chosen from training-set counts. Other species (squirrel, skunk,
bird, rodent, badger, fox, deer) are kept as unsupported-input cases, never
relabeled as empty. See the [model card](docs/model_card.md).

## Tech stack

Python 3.12, uv · PyTorch and torchvision (EfficientNet-B0), scikit-learn ·
FastAPI, Streamlit · PostgreSQL 16 (SQLAlchemy, Alembic) · Redis and RQ ·
SeaweedFS (S3-compatible object storage) · MLflow tracking · Docker Compose ·
GitHub Actions · ruff, mypy, pytest.

## Requirements

- [uv](https://docs.astral.sh/uv/) and Git (uv installs Python 3.12 and every
  locked dependency).
- Docker with Compose v2 for the deployment.
- For reproducing training: about 20 GB of disk for the dataset, and a GPU or
  Apple silicon (training runs on CPU too, much more slowly).

## Installation

```bash
git clone https://github.com/professor3333/wildinbox.git
cd wildinbox
uv sync --locked          # .venv with the exact locked versions
cp .env.example .env      # local settings; contains no secrets
```

## Quick start: deploy and run the demo

```bash
docker compose up -d --build --wait    # Postgres, Redis, SeaweedFS, migrations, API, worker, UI
uv run python scripts/smoke.py         # upload -> process -> results, end to end
uv run python scripts/demo.py          # upload the sample, list uncertain events, correct one, export
open http://localhost:8501             # review interface (API: http://localhost:8000)
```

Without a registered model the deployment uses the clearly labeled **test
predictor** (pseudo-random scores that exercise the pipeline, flagged on every
response). To serve the trained model, register it once as a release (weights
and policy artifact come from the training machine, see Reproduce below):

```bash
docker compose run --rm \
  -v "$PWD/models/finetune-e3-deep-balanced:/app/models/finetune-e3-deep-balanced:ro" \
  -v "$PWD/reports/calibration:/app/reports/calibration:ro" \
  worker wildinbox release register --activate
curl localhost:8000/version            # deploy check: release, weights, preprocessing, calibration, policy
```

The demo walk-through, with a real run's output: [`docs/demo.md`](docs/demo.md).
`docker compose down` stops the stack (`-v` also deletes its data).

## Usage

### Review interface (`http://localhost:8501`)

| Page | What it does |
|---|---|
| Review queue | Frames, suggestion, confidence, and why the event needs review. Accept, pick another label, type an unsupported species, or "can't tell"; the outcome (confirmed / corrected / unresolved) follows from the choice. |
| Last night's visitors | The best frame of every animal event in one night, opening on the newest night with activity. |
| Upload | Upload photos with a camera name; follow processing; see unusable files and why. |
| Export | The observation CSV with provenance. |
| Monitoring | Alerts, operations, label-free signals, and review-based accuracy. |

### API (`http://localhost:8000`, schema at `/docs`)

| Endpoint | Purpose |
|---|---|
| `POST /batches` | Upload `files` plus optional `metadata`; returns `202` with the batch and job. `Idempotency-Key` makes retries safe. |
| `GET /batches`, `GET /batches/{id}` | Recent batches; progress, counts, every failed file, job lifecycle, pinned release |
| `GET /batches/{id}/export` | Observations with provenance (`?format=csv` or `json`) |
| `GET /events`, `GET /events/{id}` | Paginated, filterable events; frames, predictions, decision, reviews |
| `POST /events/{id}/reviews` | Append a review |
| `GET /version`, `GET /releases` | Active release; all releases and the activation history |
| `GET /monitoring`, `GET /metrics` | Monitoring as JSON and in Prometheus text format |

Full contract, job lifecycle, and recovery rules: [`docs/api.md`](docs/api.md).
Limits (see `.env.example`): 2,000 files and 1 GiB per batch, 20 MiB per file.

### Command line (`uv run wildinbox --help`)

| Area | Commands |
|---|---|
| Data | `data acquire`, `dataset build`, `validate-config`, `export-schemas` |
| Models | `baseline train`, `finetune train`, `evaluate`, `compare`, `unfamiliar`, `calibrate`, `replay`, `final-test` |
| Releases | `release register`, `release activate`, `release list` |
| Update cycle | `snapshot build`, `update gate` |
| Operations | `api`, `worker`, `ui`, `jobs recover`, `monitoring summary`, `monitoring backfill-quality` |

## Reproduce the data and models

Everything is rebuilt from the pinned manifests; nothing downloaded or trained
is committed.

```bash
uv run wildinbox data acquire      # download (resumable, MD5-checked) -> extract -> ingest -> lock check -> report
uv run wildinbox dataset build     # events -> labels -> species -> camera-disjoint partitions -> leakage checks
uv run wildinbox baseline train    # frozen EfficientNet-B0 embeddings + logistic regression
uv run wildinbox evaluate --compare-to reports/baseline/metrics.json
uv run wildinbox finetune train --config configs/experiments/finetune-e3-deep-balanced.yaml
uv run wildinbox evaluate --model models/finetune-e3-deep-balanced \
  --report-dir reports/experiments/finetune-e3-deep-balanced \
  --compare-to reports/experiments/finetune-e3-deep-balanced/metrics.json
uv run wildinbox unfamiliar        # unfamiliar-input score vs confidence (evaluated, not adopted)
uv run wildinbox calibrate         # temperature, pre-registered operating point, policy artifact
uv run wildinbox replay            # every saved decision reproduces from saved predictions
```

`--compare-to` fails unless the fresh run matches the committed report within
the documented tolerances. The other experiments, the selection rule, and the
final test have their own commands and reports:
[experiments](reports/experiments/README.md),
[unfamiliar inputs](reports/unfamiliar/README.md),
[final test](reports/final_test/README.md) (`wildinbox final-test` refuses to
run unless every pinned artifact matches, and a re-run must reproduce the
recorded numbers). MLflow runs are local:
`uvx mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`.

## Data source and manifest

[Caltech Camera Traps](https://lila.science/datasets/caltech-camera-traps)
(CCT20 subset, resized images, about 6.5 GB), released under the Community Data
License Agreement (permissive variant); archive sizes and MD5s are pinned in
[`configs/sources/cct20.yaml`](configs/sources/cct20.yaml).

- [`manifests/cct20.lock.json`](manifests/cct20.lock.json) pins the inventory:
  manifest version, archive checksums, and record counts (57,864 source
  records: 57,855 accepted, 7 excluded duplicates, 2 quarantined).
- Each manifest record (`data/manifests/<version>.jsonl.gz`) holds
  `source_id`, `file_name`, `storage_path`, `sha256`, `width`, `height`,
  `camera_id`, `sequence_id`, `frame_num`, `seq_num_frames`, `captured_at`,
  `original_labels`, `normalized_label` with its `label_rule`, `status`
  (accepted / quarantined / excluded), `reasons`, and `warnings`.
- [`manifests/cct20-splits-v1.lock.json`](manifests/cct20-splits-v1.lock.json)
  pins the camera-disjoint partitions; rules in [`docs/dataset.md`](docs/dataset.md),
  leakage checks in [`reports/splits/cct20/README.md`](reports/splits/cct20/README.md).

### What is and is not committed

| Committed | Not committed (reproduced or local) |
|---|---|
| Code, configs, migrations, tests, pinned manifests (lock files) | Downloaded archives and extracted images (`data/`) |
| Reports, metrics, curves, policy artifacts, saved development decisions | Trained weights (`models/`), MLflow runs (`mlruns/`) |
| The 37-photo demo sample [`samples/cct-dev`](samples/cct-dev) with its license | Uploads, thumbnails, and database contents |
| JSON Schemas of the shared contracts (`docs/schemas`) | `.env`, `.venv/` |

## Testing

No network is needed.

```bash
uv run ruff check && uv run ruff format --check && uv run mypy src/
uv run wildinbox validate-config configs/*.yaml
uv run pytest -m "not slow"
```

The API, worker, and monitoring tests need a PostgreSQL they may wipe; without
one they are skipped locally (CI always runs them):

```bash
docker run -d --name wildinbox-test-pg -p 55432:5432 -e POSTGRES_USER=wildinbox \
  -e POSTGRES_PASSWORD=wildinbox -e POSTGRES_DB=wildinbox postgres:16-alpine
export WILDINBOX_TEST_DATABASE_URL=postgresql+psycopg://wildinbox:wildinbox@localhost:55432/wildinbox
uv run pytest -m "not slow"
```

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs lint, format,
types, config and schema checks, migrations up/check/down, the tests
(including preprocessing consistency between training and serving, API
contracts, and retry behaviour), then builds the Compose stack and runs the
smoke test and the demo.

## Development

The Docker Compose loop: edit code, then `docker compose up -d --build --wait`
rebuilds and restarts the API, worker, and UI (migrations run first). Useful
while developing:

- `uv run wildinbox ui --api-url http://localhost:8000` runs the review UI
  against any API.
- `docker compose logs -f worker` follows processing.

## Deployment and rollback

One VM with Docker Compose runs the API, worker(s), UI, PostgreSQL, Redis, and
SeaweedFS; training runs on a separate machine.

1. Train and evaluate offline; `wildinbox calibrate` writes the policy
   artifact for the weights.
2. Copy `models/<name>/` and the policy artifact to the VM and run
   `wildinbox release register --activate` (as in Quick start). Registration
   refuses a policy artifact that belongs to other weights.
3. Check the deploy: `curl localhost:8000/version` must report the expected
   release, weights SHA-256, preprocessing, calibration, and policy versions.
4. **Roll back:** `docker compose exec api wildinbox release activate <previous-release-id>`.
   New batches use it immediately; running jobs keep the release they started
   with; the activation history is in `GET /releases`.

Scale by running more workers (`docker compose up -d --scale worker=3`): jobs
are claimed under leases, so workers never process the same job at once.

## Measured processing cost

On an Apple M1 (8 GB), Docker VM with 8 vCPUs and 3.9 GB, CPU inference, one
worker ([serving report](reports/serving/README.md)):

| Measure | Result | Target |
|---|---|---|
| 1,000 images scored, grouped, and decided | 78.5 s (12.7 images/s) | within 10 minutes |
| Metadata API p95 latency | 3.5-139 ms | < 500 ms |
| Worker killed mid-batch (real `SIGKILL`) | resumed; no lost inputs, duplicates, or early events | |

Training the released model: 70 minutes on the M1's GPU (Metal).

## Limitations

- **New cameras remain hard:** macro-F1 falls from 0.75 on training cameras to
  0.45 on new ones and varies from 0.24 to 0.57 by camera; small animals (36%
  recall), night frames, and blur are the main failure modes.
- **No automation is released:** evaluation did not support a threshold, so
  review reduction comes only from grouping.
- **Unfamiliar species are not reliably flagged:** the distance-based score
  was not adopted; it mostly measures "new camera", not "new species".
- **Evaluated species were in pretraining:** every unsupported species that
  could be evaluated (except deer) appears in ImageNet-1k.
- **Reviews in the update cycle were simulated** from ground truth; no timed
  study with real reviewers has been run yet.
- **One machine, one worker** measured; the event list is not optimised for very
  large pages.
- Capture events are not individual animals; nothing here estimates population
  counts.

## Project structure

```
src/wildinbox/
  api/          FastAPI app, upload validation, export
  workers/      job lifecycle: leases, retries, recovery, processing
  inference/    releases, serving scorer, calibration, unfamiliar-input score
  policy/       versioned event policies and decision replay
  ui/           Streamlit review interface (talks only to the API)
  monitoring/   operations, signals, review-based accuracy, Prometheus text
  training/     baseline, fine-tuning, snapshots, promotion gate
  evaluation/   evaluation, comparison, calibration, final test, reports
  datasets/     events, labels, splits, leakage checks
  ingestion/    download, validation, inventory
  storage/      PostgreSQL models, object storage
  preprocessing.py, quality.py, class_map.py, schemas.py, config.py, settings.py, cli.py
configs/        run config, dataset sources, splits, taxonomy, experiments, monitoring
manifests/      pinned dataset and split locks
migrations/     Alembic migrations
reports/        every committed evaluation, experiment, serving, and monitoring report
samples/        the demo sample batch
scripts/        smoke, demo, crash demo, sample batches, simulated deployment, release/rollback
docs/           requirements, dataset rules, API, demo, model card, JSON Schemas
tests/
```

## Data license and attribution

Images and annotations: Caltech Camera Traps (Beery, Van Horn, and Perona,
"Recognition in Terra Incognita", ECCV 2018), distributed by LILA BC under the
[Community Data License Agreement - Permissive, Version 1.0](https://cdla.dev/permissive-1-0/).
