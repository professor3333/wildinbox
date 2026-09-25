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
    A --> S3[(Object storage, S3 or SeaweedFS:<br/>originals, release weights, backups)]
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
- **Deploy to staging** on one AWS VM ([guide](docs/deployment.md)): bearer
  tokens on every data endpoint, readiness that requires the expected model
  to load, JSON logs, upload limits, nightly database backups to S3 with a
  tested restore, and a load-tested release
  ([report](reports/staging/README.md)).
- **Monitor** in two views ([rules](docs/monitoring.md)). Operational health
  covers queue, failures, latency, API errors, worker memory and deaths,
  storage, and batch cost. Model behavior covers filtered and review shares,
  label and confidence mixes, each camera's latest batch against its earlier
  ones, time periods with the camera-mix effect separated, and audited
  false-empty and species errors. Signals and measured errors are never
  mixed: a camera with no audit labels shows unknown quality.
- **Learn from corrections, under control** ([workflow](docs/retraining.md)).
  Snapshots use approved reviews only, exclude protected evaluation records
  (with any corrections made to them), and keep per-label provenance.
  Candidates are compared on development data only, and every comparison is
  logged. Rolling back restores the previous release's predictions exactly.

### Results, honestly

Measured on the **locked final test** (9 cameras never used for training,
calibration, or thresholds; 23,275 photos, 8,982 capture events), under
protocols committed beforehand, with the release frozen
([final test](reports/final_test/README.md),
[final evaluation](reports/final_evaluation/README.md)). Intervals are 95%;
event metrics use a cluster bootstrap over camera-nights, because events from
one camera on one night are not independent.

| | Released (automation off) | Rule's empty filter (not released) | Target |
|---|---|---|---|
| Animal-event retention | 100% | 98.95% [98.68, 99.22] | ≥ 98% |
| · unsupported species | 100% | 97.14% [95.20, 98.66] | |
| Accepted species precision | undefined: nothing accepted | undefined | ≥ 95% |
| Review reduction vs grouped workflow, with audits | 0% | 5.99% [5.21, 6.76] | ≥ 50% |
| Unsupported species accepted as known | 0 / 594 | 0 / 594 | |

| | Fine-tuned (released) | Frozen-embedding baseline |
|---|---|---|
| Macro-F1 on unseen cameras (95% CI) | **0.447** [0.435, 0.458] | 0.285 |
| Same model on held-out sequences of *training* cameras | 0.747 | 0.724 |

- **The 50% review-reduction target is not met, so automation stays off.**
  The only automation development evidence supported, the empty filter,
  would save 6% of reviews, and it falls below 98% retention for unsupported
  species, by day, and on one camera (94.2%). Every event goes to review with
  its suggestion, confidence, and reasons.
- **What helps as released is grouping:** 23,275 photos become 8,982 events
  to review (61.4% fewer items), and each comes with a suggestion.
- The gap between 0.747 and 0.447 is the new-camera problem this project set out
  to measure. Reviewing a camera helps that camera: one update cycle raised
  label quality on later photos of the reviewed cameras from 0.486 to 0.548
  ([update cycle](reports/update/README.md), reviews simulated from ground truth).

Supported classes: **empty, bobcat, cat, coyote, dog, opossum, rabbit,
raccoon**, chosen from training-set counts. Other species (squirrel, skunk,
bird, rodent, badger, fox, deer) are kept as unsupported-input cases, never
relabeled as empty. See the [model card](docs/model_card.md).

## For reviewers

| Deliverable | Where |
|---|---|
| Runnable demo | [Quick start](#quick-start-deploy-and-run-the-demo), [docs/demo.md](docs/demo.md), video [docs/media/demo.webm](docs/media/demo.webm) |
| Architecture | [docs/architecture.md](docs/architecture.md) |
| Dataset and training instructions | [Reproduce the data and models](#reproduce-the-data-and-models), [docs/dataset.md](docs/dataset.md) |
| Model card | [docs/model_card.md](docs/model_card.md) |
| Evaluation report | [reports/final_evaluation/](reports/final_evaluation/README.md), [reports/final_test/](reports/final_test/README.md) |
| Error gallery | [final evaluation gallery](reports/final_evaluation/README.md#error-gallery), [development gallery](reports/baseline/README.md) |
| Operational benchmark | [reports/staging/](reports/staging/README.md) (AWS VM), [reports/serving/](reports/serving/README.md) (laptop) |
| Deployment, backup, rollback | [docs/deployment.md](docs/deployment.md), [docs/retraining.md](docs/retraining.md) |

Every headline claim and where it comes from:

| Claim | Evidence | Reproduce |
|---|---|---|
| Retention, review reduction, precision, coverage, unsupported acceptance on unseen cameras | `reports/final_evaluation/metrics.json` (plan `configs/experiments/final_evaluation.yaml`, committed first) | `uv run wildinbox final-evaluation` |
| Macro-F1 0.447 vs baseline 0.285; random-image vs unseen-camera gap | `reports/final_test/metrics.json` (protocol `configs/experiments/final_test.yaml`) | `uv run wildinbox final-test` (refuses unless it reproduces exactly) |
| Thresholds chosen on development cameras only; automation off | `reports/calibration/`, `configs/experiments/operating_point*.yaml` | `uv run wildinbox calibrate`, `uv run wildinbox replay` |
| Model selection | `reports/experiments/comparison.md`, `configs/experiments/selection.yaml` | `uv run wildinbox compare reports/baseline reports/experiments/finetune-e*` |
| 1,000 images in 111 s; metadata p95 < 360 ms (2-vCPU VM) | `reports/staging/workers-1.json` | `scripts/loadtest.py` ([how](reports/staging/README.md#reproduce)) |
| Worker crash or restart loses and duplicates nothing | `reports/serving/crash-demo.json`, `reports/staging/workers-1.json` | `scripts/crash_demo.py`, `scripts/loadtest.py --scenarios restart` |
| Update cycle, promotion gate, rollback restores predictions | `reports/update/`, `reports/monitoring/acceptance.json` | `wildinbox update gate`, `scripts/rollback_restores.py` |
| Backup and restore | `reports/staging/backup-restore.log` | `deploy/staging/restore_drill.sh` |
| A stranger can deploy the pinned release | `reports/staging/rehearsal/` | [docs/deployment.md](docs/deployment.md) |

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
response). To serve the trained model, register it once as a release. The
weights and policy are published with the
[v1.5.0 release](https://github.com/professor3333/wildinbox/releases/tag/v1.5.0)
(15 MB, SHA-256 checked; no GitHub account needed):

```bash
deploy/fetch_release.sh                # download the trained release (v1.5.0 asset), verify, unpack
docker compose run --rm -v "$PWD/dist/release:/release:ro" worker \
  wildinbox release register --activate --model-dir /release/model --policy /release/policy.json
curl localhost:8000/version            # deploy check: release, weights, preprocessing, calibration, policy
```

The demo walk-through, with a real run's output: [`docs/demo.md`](docs/demo.md).
`docker compose down` stops the stack (`-v` also deletes its data).

## Usage

### Review interface (`http://localhost:8501`)

| Page | What it does |
|---|---|
| Getting started | A one-screen guide for first-time users (no other documentation needed). |
| Upload | Upload photos with a camera name or the card's metadata file (capture times, sequences, cameras per file); follow processing; see unusable files and why. |
| Batches | Progress, counts, and failed files for every upload. |
| Review queue | Events that need a person: frames, suggestion, confidence, and why. Accept, pick another species, choose empty, type an unsupported species, or "can't tell"; every event opens to all frames, per-frame predictions, and its full review history. |
| Timeline | Events by day and camera. |
| Last night's visitors | The best frame of every animal event in one night. |
| Automatically filtered | Events automation set aside as empty; label one to recover it. |
| Audit queue | A random sample (default 5%, `WILDINBOX_AUDIT_RATE`) of automatic decisions, each with the rule that chose it. |
| Export | The observation CSV ([format](docs/export_format.md)). |
| Monitoring | Two views: **Operational health** (queue, failures, latency, API errors, workers, storage, batch cost) and **Model behavior** (decision shares, label and confidence mixes, latest batch per camera, time periods with the camera mix separated, audited errors, review-based accuracy). Rules: [docs/monitoring.md](docs/monitoring.md). |
| Study | The timed review study for participants ([guide](docs/review_study.md)). |

Labels are visibly different by source: ✅ confirmed by a person, ⚙️ decided
automatically, 🤖 a suggestion nobody has reviewed, ❔ unresolved. Reviews are
appended; the model's suggestion is never overwritten.

### API (`http://localhost:8000`, schema at `/docs`)

| Endpoint | Purpose |
|---|---|
| `POST /batches` | Upload `files` plus optional `metadata`; returns `202` with the batch and job. `Idempotency-Key` makes retries safe. |
| `GET /batches`, `GET /batches/{id}` | Recent batches; progress, counts, every failed file, job lifecycle, pinned release |
| `GET /batches/{id}/export` | Observations with provenance (`?format=csv` or `json`); columns in [`docs/export_format.md`](docs/export_format.md) |
| `GET /events`, `GET /events/{id}` | Paginated, filterable events; frames, predictions, decision, reviews |
| `POST /events/{id}/reviews` | Append a review |
| `GET /version`, `GET /releases` | Active release; all releases and the activation history |
| `GET /health`, `GET /ready` | Liveness; readiness (database, queue, object store, expected release loaded) |
| `GET /whoami` | The principal behind the token |
| `GET /monitoring`, `GET /metrics` | Monitoring as JSON and in Prometheus text format |

Full contract, job lifecycle, and recovery rules: [`docs/api.md`](docs/api.md).
With `WILDINBOX_AUTH=tokens` (the default; staging) every endpoint except
`/health`, `/ready`, `/docs`, and the upload page needs
`Authorization: Bearer <token>`; create tokens with `wildinbox token new NAME`.
The local Compose stack sets `WILDINBOX_AUTH=disabled`.
Limits (see `.env.example`): 2,000 files and 1 GiB per batch, 20 MiB per file.

### Command line (`uv run wildinbox --help`)

| Area | Commands |
|---|---|
| Data | `data acquire`, `dataset build`, `validate-config`, `export-schemas` |
| Models | `baseline train`, `finetune train`, `evaluate`, `compare`, `unfamiliar`, `calibrate`, `replay`, `final-test`, `final-evaluation` |
| Releases | `release register`, `release activate`, `release list` |
| Update cycle | `snapshot build`, `update gate` |
| Operations | `api`, `worker`, `ui`, `jobs recover`, `token new`, `monitoring summary`, `monitoring backfill-quality` |
| Review study | `study plan`, `study analyze` |

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

**Staging** runs on one AWS VM with Docker Compose, originals and weights in
S3, and access only over SSH: [docs/deployment.md](docs/deployment.md) covers
creating the stack, secrets, deploying a pinned release, the deploy check,
backups and restore, and teardown. The steps below apply to both staging and
the local stack.

1. Train and evaluate offline; `wildinbox calibrate` writes the policy
   artifact for the weights.
2. Copy `models/<name>/` and the policy artifact to the VM and run
   `wildinbox release register --activate` (as in Quick start). Registration
   refuses a policy artifact that belongs to other weights.
3. Check the deploy: `curl localhost:8000/version` must report the expected
   release, weights SHA-256, preprocessing, calibration, and policy versions.
4. **Roll back:** `docker compose exec api wildinbox release activate <previous-release-id>`.
   New batches use it immediately; running jobs keep the release they started
   with; the activation history is in `GET /releases`. Reverting restores the
   previous release's predictions exactly (`scripts/rollback_restores.py`).

Model updates follow [docs/retraining.md](docs/retraining.md): approved
corrections, then `wildinbox snapshot build`, a candidate trained with the
unchanged recipe, `wildinbox update gate` (development data only, comparisons
logged), release, monitoring, and rollback if needed. Alerts and what to do
about each: [docs/monitoring.md](docs/monitoring.md).

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
- **No automation is released:** evaluation did not support a threshold, and
  the one candidate (the empty filter) would save 6% of reviews on unseen
  cameras against a 50% target, so review reduction comes only from grouping
  ([final evaluation](reports/final_evaluation/README.md)).
- **The final test has been used:** it was measured twice without any choice
  made from it. A genuinely fresh assessment, or any automation restricted to
  particular cameras or hours, needs new held-out cameras.
- **Unfamiliar species are not reliably flagged:** the distance-based score
  was not adopted; it mostly measures "new camera", not "new species".
- **Evaluated species were in pretraining:** every unsupported species that
  could be evaluated (except deer) appears in ImageNet-1k.
- **Reviews in the update cycle were simulated** from ground truth. The timed
  review study is built and pre-registered
  ([`docs/review_study.md`](docs/review_study.md)) but has not been run with
  people, so whether suggestions speed up review is not yet measured.
- **Measured on two machines:** a laptop and a 2-vCPU AWS VM with one and two
  workers. One batch is always processed by one worker. `GET /events` (100 per
  page) is the slowest metadata call (p95 360 ms).
- **Staging access is single-workspace** (every token holder sees every batch)
  and reached over SSH; a public endpoint needs a TLS proxy.
- Capture events are not individual animals; nothing here estimates population
  counts.

## Project structure

```
src/wildinbox/
  api/          FastAPI app, upload validation, export
  workers/      job lifecycle: leases, retries, recovery, processing, worker liveness
  inference/    releases, serving scorer, calibration, unfamiliar-input score
  policy/       versioned event policies and decision replay
  ui/           Streamlit review interface (talks only to the API)
  monitoring/   operational health, model behavior, audits, alert rules, Prometheus text
  training/     baseline, fine-tuning, snapshots, promotion gate
  evaluation/   evaluation, comparison, calibration, final test, reports
  datasets/     events, labels, splits, leakage checks
  ingestion/    download, validation, inventory
  storage/      PostgreSQL models, object storage
  preprocessing.py, quality.py, class_map.py, schemas.py, config.py, settings.py, cli.py
configs/        run config, dataset sources, splits, taxonomy, experiments, monitoring
manifests/      pinned dataset and split locks
migrations/     Alembic migrations
reports/        every committed evaluation, experiment, final test and final evaluation,
                serving, staging, and monitoring report
samples/        the demo sample batch
scripts/        smoke, demo, crash demo, sample batches, simulated deployment, release/rollback,
                monitoring acceptance demo, rollback-restores check, staging load test,
                demo video recording
deploy/         AWS CloudFormation template, staging compose, deploy/backup/restore scripts
docs/           architecture, requirements, dataset rules, API, deployment, export format,
                monitoring and alert rules, retraining workflow, review study, demo (and video),
                model card, JSON Schemas
tests/
```

## Data license and attribution

Images and annotations: Caltech Camera Traps (Beery, Van Horn, and Perona,
"Recognition in Terra Incognita", ECCV 2018), distributed by LILA BC under the
[Community Data License Agreement - Permissive, Version 1.0](https://cdla.dev/permissive-1-0/).
