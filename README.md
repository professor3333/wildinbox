<div align="center">

<img src="docs/media/banner.svg" alt="WildInbox: find the wildlife, skip the empty frames" width="100%">

### Turn a trail-camera memory card into a diary of animal sightings, and send a person only what needs one.

[![ci](https://github.com/professor3333/wildinbox/actions/workflows/ci.yml/badge.svg)](https://github.com/professor3333/wildinbox/actions/workflows/ci.yml)
[![release](https://img.shields.io/github/v/release/professor3333/wildinbox?color=2e7d5b&label=release)](https://github.com/professor3333/wildinbox/releases)
![python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
[![data: CDLA-Permissive-1.0](https://img.shields.io/badge/data-CDLA--Permissive--1.0-6a737d)](https://cdla.dev/permissive-1-0/)

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL_16-4169E1?logo=postgresql&logoColor=white)
![Redis](https://img.shields.io/badge/Redis_%2B_RQ-DC382D?logo=redis&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker_Compose-2496ED?logo=docker&logoColor=white)

[**Quick start**](#quick-start) ·
[**Results**](#results-honestly) ·
[**Demo video**](docs/media/demo.webm) ·
[**Model card**](docs/model_card.md) ·
[**Architecture**](docs/architecture.md) ·
[**API**](docs/api.md)

</div>

---

WildInbox is a review system for trail-camera photos. Upload a memory card and it
groups the photos into **capture events**, suggests a species for each event,
says why an event needs a human, and exports a corrected observation log with
the provenance of every label.

<p align="center">
  <img src="docs/media/demo.gif" alt="Uploading a memory card, reviewing events, and exporting the observation log" width="100%">
</p>

<table>
<tr>
<td width="33%"><a href="docs/media/demo-upload.png"><img src="docs/media/demo-upload.png" alt="Upload a memory card"></a></td>
<td width="33%"><a href="docs/media/demo-review.png"><img src="docs/media/demo-review.png" alt="Review queue"></a></td>
<td width="33%"><a href="docs/media/demo-visitors.png"><img src="docs/media/demo-visitors.png" alt="Last night's visitors"></a></td>
</tr>
<tr>
<td align="center"><b>Upload</b><br><sub>Frames of one trigger become one capture event</sub></td>
<td align="center"><b>Review queue</b><br><sub>Suggestion, confidence, and why it needs a person</sub></td>
<td align="center"><b>Last night's visitors</b><br><sub>The best frame of every animal event in a night</sub></td>
</tr>
</table>

<p align="center">▶️ <a href="docs/media/demo.webm"><b>Full-length video</b></a> · walk-through with a real run's output in <a href="docs/demo.md"><code>docs/demo.md</code></a></p>

## At a glance

<table>
<tr>
<td align="center" width="20%"><h3>61.4%</h3>fewer items to review, from grouping alone<br><sub>23,275 photos → 8,982 events</sub></td>
<td align="center" width="20%"><h3>0.447</h3>macro-F1 on 9 unseen cameras<br><sub>vs 0.285 frozen baseline</sub></td>
<td align="center" width="20%"><h3>100%</h3>animal events retained, as released<br><sub>automation off</sub></td>
<td align="center" width="20%"><h3>78.5 s</h3>for 1,000 images, one worker<br><sub>laptop CPU; 111 s on a 2-vCPU VM</sub></td>
<td align="center" width="20%"><h3>0 / 0</h3>lost / duplicated inputs<br><sub>after a worker <code>SIGKILL</code> mid-batch</sub></td>
</tr>
</table>

> [!IMPORTANT]
> **The model does not yet remove review work, and the release says so.** The
> 50% review-reduction target was not met on unseen cameras, so automatic
> filtering and species labels are **off**: every event goes to a person with
> its suggestion, confidence, and reasons. See [Results, honestly](#results-honestly).

## The problem

Trail cameras fire on wind, heat, and animals walking out of frame, so a memory
card is mostly empty or repeated photos; about **70% of Caltech Camera Traps is
labeled empty**. WildInbox asks one measurable question:

> **How much manual review can be removed without losing animals when a camera
> is somewhere new?**

## Architecture

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

    classDef ingest fill:#e3f2e9,stroke:#2e7d5b,color:#123;
    classDef store fill:#e8eef9,stroke:#3b5ba5,color:#123;
    classDef serve fill:#fff4e0,stroke:#c77d00,color:#123;
    classDef human fill:#fde8e6,stroke:#c0392b,color:#123;
    classDef ml fill:#efe7f8,stroke:#6f42c1,color:#123;
    class U,A ingest;
    class S3,PG,Q store;
    class W,R,M serve;
    class UI,H human;
    class SN,T,G ml;
```

<sub>🟩 ingest · 🟦 storage and queue · 🟧 serving and monitoring · 🟥 human review · 🟪 offline training and release gate</sub>

Each event gets one disposition from a versioned policy with machine-readable
reasons (in the current release, every event is **needs review**):

| Disposition | When | What you do |
|---|---|---|
| 🍃 **Likely empty** | Every usable frame strongly supports empty | Skip normal review; still recoverable and audited |
| 🦝 **Species identified** | One supported species, confident enough under the policy | Browse or accept the label |
| 🔍 **Needs review** | Uncertain, conflicting, unsupported, or automation off | Inspect frames and correct the label |

## Features

- 📥 **Upload and validate** JPEG/PNG batches with optional camera, time, and sequence metadata; content checks, hash-based duplicate detection, and an explicit error for every unusable file.
- 🧩 **Group** photos into capture events by sequence id, or by camera and a per-upload time gap (default 5 s).
- 🧠 **Suggest** a species per event with a calibrated, fine-tuned EfficientNet-B0.
- 👀 **Review** in the browser: accept, correct, name an unsupported species, or mark "can't tell"; export a CSV with provenance.
- 🔁 **Process reliably**: leased workers, bounded retries, stale-job recovery, idempotent writes.
- 🚀 **Release safely**: immutable releases pinned per job, a pre-registered promotion gate, one-command rollback.
- 📈 **Monitor** operational health and model behavior separately; signals are never presented as measured accuracy ([rules](docs/monitoring.md)).
- 🎓 **Retrain under control** from approved reviews only, with protected evaluation data excluded ([workflow](docs/retraining.md)).
- ☁️ **Deploy to staging** on one AWS VM with tokens, readiness checks, and tested S3 backups ([guide](docs/deployment.md)).

## Results, honestly

Measured on the **locked final test**: 9 cameras never used for training,
calibration, or thresholds (23,275 photos, 8,982 events), under protocols
committed beforehand ([final test](reports/final_test/README.md),
[final evaluation](reports/final_evaluation/README.md)). Intervals are 95%,
cluster-bootstrapped over camera-nights.

| | Released (automation off) | Rule's empty filter (not released) | Target |
|---|---|---|---|
| Animal-event retention | ✅ 100% | 98.95% [98.68, 99.22] | ≥ 98% |
| · unsupported species | ✅ 100% | ⚠️ 97.14% [95.20, 98.66] | |
| Accepted species precision | undefined: nothing accepted | undefined | ≥ 95% |
| Review reduction vs grouped workflow, with audits | ❌ 0% | ❌ 5.99% [5.21, 6.76] | ≥ 50% |
| Unsupported species accepted as known | 0 / 594 | 0 / 594 | |

| | Fine-tuned (released) | Frozen-embedding baseline |
|---|---|---|
| Macro-F1 on unseen cameras (95% CI) | **0.447** [0.435, 0.458] | 0.285 |
| Same model on held-out sequences of *training* cameras | 0.747 | 0.724 |

- **Automation stays off.** The only supported candidate, the empty filter, saves 6% of reviews and drops below 98% retention for unsupported species, by day, and on one camera (94.2%).
- **Grouping and the model are credited separately.** Grouping alone gives 61.4% fewer items; the released model adds 0%. Fewer items is not measured reviewer time.
- **Suggestions are not shown to help reviewers.** A 5-person pilot found no reliable speed or accuracy gain ([study](reports/study/study-1/FINDINGS.md)).
- **0.747 → 0.447 is the new-camera problem.** One update cycle raised label quality on later photos of reviewed cameras from 0.486 to 0.548 ([update cycle](reports/update/README.md), simulated reviews).

**Camera adaptation (v2), pre-registered, failed.** Adapting to each new
camera from its first 50 reviewed events reached **88.6%** (505 / 570)
[85.7, 91.0] accepted-label precision on 12 fresh cameras, against a ≥ 95%
target, with 12.8% review reduction and 100% retention. One camera gave 39 of
the 65 errors, so automatic acceptance stays off
([plan](docs/v2_experiment.md), [report](reports/adaptation/fresh_test/README.md)).

| Supported (classifier output) | Unsupported (kept for evaluation, never relabeled as empty) |
|---|---|
| empty · bobcat · cat · coyote · dog · opossum · rabbit · raccoon | squirrel · skunk · bird · rodent · badger · fox · deer |

<details>
<summary><b>📷 What the supported classes look like in the dataset</b></summary>
<br>

| | |
|---|---|
| **empty**<br><img src="reports/data_quality/cct20/images/class_empty.jpg" alt="empty"> | **bobcat**<br><img src="reports/data_quality/cct20/images/class_bobcat.jpg" alt="bobcat"> |
| **cat**<br><img src="reports/data_quality/cct20/images/class_cat.jpg" alt="cat"> | **coyote**<br><img src="reports/data_quality/cct20/images/class_coyote.jpg" alt="coyote"> |
| **dog**<br><img src="reports/data_quality/cct20/images/class_dog.jpg" alt="dog"> | **opossum**<br><img src="reports/data_quality/cct20/images/class_opossum.jpg" alt="opossum"> |
| **rabbit**<br><img src="reports/data_quality/cct20/images/class_rabbit.jpg" alt="rabbit"> | **raccoon**<br><img src="reports/data_quality/cct20/images/class_raccoon.jpg" alt="raccoon"> |

</details>

<details>
<summary><b>🔎 Every headline claim, its evidence, and how to reproduce it</b></summary>
<br>

| Claim | Evidence | Reproduce |
|---|---|---|
| Retention, review reduction, precision, unsupported acceptance on unseen cameras; grouping 61.4% vs model 0% | `reports/final_evaluation/metrics.json` (plan `configs/experiments/final_evaluation.yaml`, committed first) | `uv run wildinbox final-evaluation` |
| Macro-F1 0.447 vs baseline 0.285; random-image vs unseen-camera gap | `reports/final_test/metrics.json` (protocol `configs/experiments/final_test.yaml`) | `uv run wildinbox final-test` (refuses unless it reproduces; [CPU record](reports/final_test/cpu-reproduction.json)) |
| Thresholds chosen on development cameras only; automation off | `reports/calibration/`, `configs/experiments/operating_point*.yaml` | `uv run wildinbox calibrate`, `uv run wildinbox replay` |
| Model selection | `reports/experiments/comparison.md`, `configs/experiments/selection.yaml` | `uv run wildinbox compare reports/baseline reports/experiments/finetune-e*` |
| 1,000 images in 111 s; metadata p95 < 360 ms (2-vCPU VM) | `reports/staging/workers-1.json` | `scripts/loadtest.py` ([how](reports/staging/README.md#reproduce)) |
| Worker crash or restart loses and duplicates nothing | `reports/serving/crash-demo.json`, `reports/staging/workers-1.json` | `scripts/crash_demo.py`, `scripts/loadtest.py --scenarios restart` |
| Update cycle, promotion gate, rollback restores predictions | `reports/update/`, `reports/monitoring/acceptance.json`, `reports/rehearsal/` | `wildinbox update gate`, `scripts/rollback_restores.py` |
| Backup and restore; a stranger can deploy the pinned release | `reports/staging/backup-restore.log`, `reports/staging/rehearsal/` | `deploy/staging/restore_drill.sh`, [docs/deployment.md](docs/deployment.md) |
| v2 adaptation fails its target (88.6% vs 95%) | `reports/adaptation/fresh_test/metrics.json` (protocol committed before download) | `uv run wildinbox adaptation fresh-test` |
| Pilot study: no demonstrated time saving | `reports/study/study-1/` | `wildinbox study posthoc --dir reports/study/study-1 --without author` |

Error galleries: [final evaluation](reports/final_evaluation/README.md#error-gallery),
[development](reports/baseline/README.md).

</details>

## Tech stack

| Layer | Tools |
|---|---|
| Language and packaging | Python 3.12, uv |
| Machine learning | PyTorch and torchvision (EfficientNet-B0), scikit-learn, MLflow tracking |
| Serving | FastAPI, Streamlit, Redis and RQ workers |
| Storage | PostgreSQL 16 (SQLAlchemy, Alembic), SeaweedFS (S3-compatible object storage) |
| Delivery | Docker Compose, GitHub Actions |
| Quality | ruff, mypy, pytest |

## Quick start

**Requires** [uv](https://docs.astral.sh/uv/), Git, and Docker with Compose v2.
Reproducing training also needs about 20 GB of disk and ideally a GPU or Apple silicon.

```bash
git clone https://github.com/professor3333/wildinbox.git && cd wildinbox
uv sync --locked                       # exact locked dependencies
cp .env.example .env                   # local settings; contains no secrets
docker compose up -d --build --wait    # Postgres, Redis, SeaweedFS, migrations, API, worker, UI
uv run python scripts/smoke.py         # upload -> process -> results, end to end
uv run python scripts/demo.py          # upload the sample, list uncertain events, correct one, export
open http://localhost:8501             # review interface (API: http://localhost:8000)
```

> [!TIP]
> Without a registered model the stack runs a clearly labeled **test predictor**
> (pseudo-random scores, flagged on every response). To serve the trained model
> (15 MB, published with [v1.5.0](https://github.com/professor3333/wildinbox/releases/tag/v1.5.0)
> and unchanged in every release since), register it once:

```bash
deploy/fetch_release.sh                # download, verify SHA-256, unpack
docker compose run --rm -v "$PWD/dist/release:/release:ro" worker \
  wildinbox release register --activate --model-dir /release/model --policy /release/policy.json
curl localhost:8000/version            # deploy check: release, weights, preprocessing, calibration, policy
```

`docker compose down` stops the stack (`-v` also deletes its data).

## Usage

**Review interface** (`http://localhost:8501`): Getting started, Upload,
Batches, Review queue, Timeline, Last night's visitors, Automatically filtered,
Audit queue (random 5% of automatic decisions), Export
([CSV format](docs/export_format.md)), Monitoring, and the review Study
([guide](docs/review_study.md)). Labels show their origin: ✅ confirmed by a
person · ⚙️ decided automatically · 🤖 unreviewed suggestion · ❔ unresolved.
Reviews are appended; the model's suggestion is never overwritten.

**API** (`http://localhost:8000`, schema at `/docs`; full contract in [`docs/api.md`](docs/api.md)):

| Endpoint | Purpose |
|---|---|
| `POST /batches` | Upload `files` plus optional `metadata`; `202` with batch and job. `Idempotency-Key` makes retries safe. |
| `GET /batches`, `GET /batches/{id}` | Batches, progress, counts, failed files, pinned release |
| `GET /batches/{id}/export` | Observations with provenance (`?format=csv` or `json`) |
| `GET /events`, `GET /events/{id}` | Paginated, filterable events with frames, predictions, decision, reviews |
| `POST /events/{id}/reviews` | Append a review |
| `GET /cameras`, `GET /version`, `GET /releases` | Cameras; active release; release and activation history |
| `GET /health`, `GET /ready`, `GET /whoami` | Liveness; readiness; the principal behind the token |
| `GET /monitoring`, `GET /metrics` | Monitoring as JSON and Prometheus text |

> [!NOTE]
> With `WILDINBOX_AUTH=tokens` (the default; staging) data endpoints need
> `Authorization: Bearer <token>` (`wildinbox token new NAME`); the local stack
> disables auth. Limits: 2,000 files and 1 GiB per batch, 20 MiB per file.

<details>
<summary><b>Command line</b> (<code>uv run wildinbox --help</code>)</summary>
<br>

| Area | Commands |
|---|---|
| Data | `data acquire`, `dataset build`, `validate-config`, `export-schemas` |
| Models | `baseline train`, `finetune train`, `evaluate`, `compare`, `unfamiliar`, `calibrate`, `replay`, `final-test`, `final-evaluation` |
| Releases | `release register`, `release activate`, `release list` |
| Update cycle | `snapshot build`, `update gate` |
| Operations | `api`, `worker`, `ui`, `jobs recover`, `token new`, `monitoring summary`, `monitoring backfill-quality` |
| Review study | `study plan`, `study analyze`, `study posthoc` |
| Camera adaptation (v2) | `adaptation develop`, `adaptation fresh-test` |

</details>

## Reproduce the data and models

Everything is rebuilt from pinned manifests; nothing downloaded or trained is committed.

```bash
uv run wildinbox data acquire      # download (resumable, MD5-checked) -> extract -> ingest -> lock check
uv run wildinbox dataset build     # events -> labels -> camera-disjoint partitions -> leakage checks
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

`--compare-to` fails unless the run matches the committed report within
documented tolerances. Evaluation commands take `--device auto|cpu|mps|cuda`.
MLflow runs are local: `uvx mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`.
More: [experiments](reports/experiments/README.md),
[unfamiliar inputs](reports/unfamiliar/README.md).

## Data source and manifest

[Caltech Camera Traps](https://lila.science/datasets/caltech-camera-traps)
(CCT20 subset, resized, about 6.5 GB), Community Data License Agreement
(permissive variant); archive sizes and MD5s are pinned in
[`configs/sources/cct20.yaml`](configs/sources/cct20.yaml).

- [`manifests/cct20.lock.json`](manifests/cct20.lock.json) pins the inventory (57,864 records: 57,855 accepted, 7 duplicates excluded, 2 quarantined).
- Each record (`data/manifests/<version>.jsonl.gz`) holds `source_id`, `file_name`, `storage_path`, `sha256`, `width`, `height`, `camera_id`, `sequence_id`, `frame_num`, `seq_num_frames`, `captured_at`, `original_labels`, `normalized_label`, `label_rule`, `status`, `reasons`, `warnings`.
- [`manifests/cct20-splits-v1.lock.json`](manifests/cct20-splits-v1.lock.json) pins the camera-disjoint splits ([rules](docs/dataset.md), [leakage checks](reports/splits/cct20/README.md)).

| Committed | Not committed (reproduced or local) |
|---|---|
| Code, configs, migrations, tests, pinned manifests | Downloaded archives and images (`data/`) |
| Reports, metrics, policy artifacts, saved decisions | Trained weights (`models/`), MLflow runs (`mlruns/`) |
| 37-photo demo sample [`samples/cct-dev`](samples/cct-dev) with its license | Uploads, thumbnails, database contents, `.env`, `.venv/` |

## Testing and development

No network is needed:

```bash
uv run ruff check && uv run ruff format --check && uv run mypy src/
uv run wildinbox validate-config configs/*.yaml
uv run pytest -m "not slow"
```

<details>
<summary><b>Running the database-backed tests locally</b></summary>
<br>

API, worker, and monitoring tests need a PostgreSQL they may wipe; without one
they are skipped locally (CI always runs them):

```bash
docker run -d --name wildinbox-test-pg -p 55432:5432 -e POSTGRES_USER=wildinbox \
  -e POSTGRES_PASSWORD=wildinbox -e POSTGRES_DB=wildinbox postgres:16-alpine
export WILDINBOX_TEST_DATABASE_URL=postgresql+psycopg://wildinbox:wildinbox@localhost:55432/wildinbox
uv run pytest -m "not slow"
```

</details>

[CI](.github/workflows/ci.yml) runs lint, types, config and schema checks,
migrations, the tests (preprocessing consistency, API contracts, retry
behaviour), then builds the Compose stack and runs the smoke test and demo.

**Development loop:** edit code, then `docker compose up -d --build --wait`
rebuilds the API, worker, and UI. `docker compose logs -f worker` follows
processing; `uv run wildinbox ui --api-url http://localhost:8000` runs the UI
against any API.

## Deployment and rollback

Staging runs on one AWS VM with Docker Compose and S3 ([docs/deployment.md](docs/deployment.md));
the same steps apply locally.

1. Train and evaluate offline; `wildinbox calibrate` writes the policy artifact.
2. Register it: `wildinbox release register --activate` (as in [Quick start](#quick-start)); a policy for other weights is refused.
3. Check the deploy: `curl localhost:8000/version` reports release, weights SHA-256, preprocessing, calibration, and policy.
4. **Roll back:** `docker compose exec api wildinbox release activate <previous-release-id>`. New batches switch immediately; running jobs keep their release; history is in `GET /releases`.

Model updates follow [docs/retraining.md](docs/retraining.md); alerts follow
[docs/monitoring.md](docs/monitoring.md). Scale with
`docker compose up -d --scale worker=3`; leases keep workers off each other's jobs.

## Measured processing cost

Apple M1 (8 GB), Docker VM with 8 vCPUs, CPU inference, one worker
([serving report](reports/serving/README.md)):

| Measure | Result | Target |
|---|---|---|
| 1,000 images scored, grouped, and decided | ✅ 78.5 s (12.7 images/s) | within 10 minutes |
| Metadata API p95, a year of history (~115,000 events), idle | ✅ 145-208 ms | < 500 ms |
| The same while one batch processes | ⚠️ 495-871 ms: **not reliably within target** | < 500 ms |
| The same during two batches or a worker restart | ❌ 538-923 ms: **target missed** | < 500 ms |
| Worker killed mid-batch (real `SIGKILL`) | ✅ resumed; no lost inputs, duplicates, or early events | |

Latency was measured on a laptop deep in swap, not the declared server
hardware ([concurrency report](reports/history/concurrency/README.md)).
Training the released model took 70 minutes on the M1's GPU.

## Limitations

- **New cameras remain hard:** macro-F1 falls from 0.75 to 0.45 on new cameras (0.24-0.57 by camera); small animals (36% recall), night frames, and blur fail most.
- **No automation is released;** review reduction comes only from grouping.
- **Both test sets are spent:** the final test (measured twice, no choices made from it) and the v2 fresh test. New claims need new held-out cameras.
- **Unfamiliar species are not flagged:** the distance score mostly measured "new camera", so it was not adopted. Every event being reviewed is the protection today ([requirements](docs/requirements.md#status-in-the-v1-release)).
- **Evaluated unsupported species** (except deer) appear in ImageNet-1k pretraining.
- **Reviewer benefit is not demonstrated:** the pilot had 5 of 8 required participants; both time and accuracy intervals include zero, and its projected 57% saving reflects accepting unchecked suggestions ([findings](reports/study/study-1/FINDINGS.md)).
- **Update-cycle reviews were simulated** from ground truth.
- **Metadata latency under load** misses target on the laptop; server hardware is unmeasured.
- **Staging is single-workspace** (every token sees every batch) and SSH-only; a public endpoint needs TLS.
- Capture events are not individual animals; nothing estimates population counts.

## Project structure

```
src/wildinbox/  api, workers, inference, policy, ui, monitoring, training,
                evaluation, datasets, ingestion, storage, shared preprocessing, cli
configs/        run config, dataset sources, splits, taxonomy, experiments, monitoring
manifests/      pinned dataset and split locks
migrations/     Alembic migrations
reports/        every evaluation, experiment, final test, serving, staging, monitoring report
samples/        the demo sample batch
scripts/        smoke, demo, crash demo, load test, release/rollback checks, demo recording
deploy/         CloudFormation template, staging compose, deploy/backup/restore scripts
docs/           architecture, API, dataset, deployment, monitoring, retraining, model card, schemas
tests/
```

## Data license and attribution

Images and annotations: Caltech Camera Traps (Beery, Van Horn, and Perona,
"Recognition in Terra Incognita", ECCV 2018), distributed by LILA BC under the
[Community Data License Agreement - Permissive, Version 1.0](https://cdla.dev/permissive-1-0/).

---

<div align="center">
<sub>Built on <a href="https://lila.science/datasets/caltech-camera-traps">Caltech Camera Traps</a> · Measured on cameras the model never saw · Capture events are not individual animals</sub>
</div>
