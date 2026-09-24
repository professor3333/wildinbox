# WildInbox

**Find the wildlife. Skip the empty frames.**

A review system for trail-camera photos. It groups photos into capture events,
suggests a species for each event, and sends the photos it can't label
reliably to a human for review. See [`docs/requirements.md`](docs/requirements.md)
for the product contract.

> Status: early development. Foundation (package, contracts, configuration, CI)
> and data acquisition are in place; training and serving are not built yet.

## Setup

Requirements: [uv](https://docs.astral.sh/uv/) and Git. uv installs the pinned
Python version (3.12) and every dependency from `uv.lock`.

```bash
git clone https://github.com/professor3333/wildinbox.git
cd wildinbox
uv sync --locked          # create .venv with exact locked versions
cp .env.example .env      # local settings; contains no secrets
```

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

## Layout

```
src/wildinbox/
  class_map.py       single source of class names and output order
  config.py          run configuration schema and loader
  preprocessing.py   the only image preprocessing (training and serving)
  schemas.py         shared contracts: image metadata, predictions, decisions, reviews
  settings.py        deployment settings from environment variables
  cli.py             `wildinbox` command
  ingestion/ datasets/ training/ evaluation/ inference/
  policy/ api/ workers/ monitoring/                  components (later stages)
ui/                  review interface
configs/             run configurations (YAML)
tests/
migrations/          database migrations
docs/                requirements, generated JSON Schemas
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

## Data and weights

Downloaded datasets, uploads, thumbnails, and trained weights are never
committed (see `.gitignore` and `data/README.md`). They are reproduced from a
pinned dataset manifest.

Dataset: [Caltech Camera Traps](https://lila.science/datasets/caltech-camera-traps),
released under the Community Data License Agreement (permissive variant).
