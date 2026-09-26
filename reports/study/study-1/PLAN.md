# Review study 1: plan record and session log

Recorded on 2026-09-26, **before any participant took part**. The protocol is
[`configs/study/review_study.yaml`](../../../configs/study/review_study.yaml)
(version 2, SHA-256 `fbb861f9b4aff490ebddac4058b41aa558bfc65c20a63c14dd00d58f0ec598bc`).
The plan stores this hash, and `wildinbox study analyze` refuses to run if the
file changes. The procedure is [docs/review_study.md](../../../docs/review_study.md).
The analysis will write `README.md`, `metrics.json`, and `export.json` next to
this file.

## Plan

| | |
|---|---|
| Study code (plan id) | `eddabd58-402e-4c86-ae73-ed5b2a622c08` |
| Deployment | local Docker Compose stack at `v1.5.3` (`f399dd8`), no authentication |
| Release and policy | `finetune-e3-deep-balanced@7a25aea97c76`, `conservative/v2+378312635a29` (automation disabled: every event needs review) |
| Study batch | `data/samples/review-study-1`: 400 calibration-camera images, 140 sequences, none previously uploaded to this stack (seed `review-study`, `--exclude-hashes`) |
| Batches | cct-108 `d4ed1a5c-f37a-4a2b-8c56-ee5b6b10cf64` (64 events), cct-51 `e9a83046-b90b-4a76-8e85-b23cad42cf4a` (76 events); all decided by the release above, grouping `sequence_id/v2` |
| Sets | A and B: 40 events each, identical ground-truth mix: empty 7, opossum 7, rabbit 7, squirrel 7, bobcat 4, bird 3, coyote 2, rodent 2, raccoon 1. Squirrel, bird, and rodent are unsupported, so the correct answer is "other". Practice: 6 events, one each of bird, bobcat, coyote, empty, opossum, rabbit. |

The same stack also holds two tooling dry runs, which are not study data and
are never analysed with this plan:
- `4d13d14a-…`: protocol v1, 1 participant;
- `53194942-…`: 8 synthetic participants on these batches, used to check the
  flow and the analysis on `v1.5.3`.

## Before each session

1. `docker compose up -d --wait`, then check that `curl -s localhost:8000/version`
   reports the release and policy above.
2. Same laptop, screen, and browser zoom for everyone. Close other windows and
   silence notifications.
3. Open <http://localhost:8501>, choose **Study** in the sidebar, then close the
   sidebar, because the other pages show model suggestions.
4. Enter the study code yourself (it is long). Then hand over the laptop.

## During the session

- The participant chooses a code: no names, and not reused across sessions.
  They read the consent text and agree.
- The session runs: practice, block 1, a difficulty rating, block 2, and a
  final rating. It takes about 15 minutes.
- Stay nearby. Don't help, comment, or answer questions about species. If
  asked, say "choose what you think it is, or can't tell."
- If they are interrupted, let them continue. A decision over 180 s is
  excluded automatically.

## After each session

1. Fill in the log below: the code, date, arm, and anything unusual. Record
   nothing identifying.
2. Back up the study data. It exists only in this stack's database:

   ```bash
   docker compose exec -T postgres pg_dump -U wildinbox -Fc wildinbox > ~/wildinbox-study-backup-$(date +%Y%m%dT%H%M).dump
   ```
3. Don't run `docker compose down -v`, and don't reset the database, until the
   study is analysed.

## Participants

Aim for 8 (a multiple of 4 keeps the arms balanced). Arms are assigned in the
order people join. Report who took part in general terms, for example
"volunteers, not trained field reviewers".

| # | Participant code | Date | Arm | Notes |
|---|---|---|---|---|
| 1 | | | 0 | |
| 2 | | | 1 | |
| 3 | | | 2 | |
| 4 | | | 3 | |
| 5 | | | 0 | |
| 6 | | | 1 | |
| 7 | | | 2 | |
| 8 | | | 3 | |

## Analysis (after the last session)

```bash
uv run wildinbox study analyze --plan-id eddabd58-402e-4c86-ae73-ed5b2a622c08 --out reports/study/study-1
```
