# Running the timed review study

Does the model actually make review faster? CLAUDE.md asks for this to be
measured with people, against an already grouped workflow, so grouping is not
credited to the model. The protocol, fixed before anyone took part, is
[`configs/study/review_study.yaml`](../configs/study/review_study.yaml).

## Design in one paragraph

Each participant labels two matched sets of 40 capture events, one set per
condition: **grouped** (frames only) and **suggested** (frames plus the model's
suggestion, confidence, review reasons, and a one-click Accept). Condition
order and set pairing rotate over four arms, assigned in the order people
join; three practice events per condition come first and are not analysed. The
primary measure is seconds per event; accuracy against ground truth,
interactions, and a 1-5 difficulty rating per block are secondary. The
analysis (per-participant medians, paired difference, bootstrap interval,
exclusion rules) is fixed in the protocol; it reports only descriptive numbers
until at least 8 participants finished. Amendment 1 (protocol version 2, made
before any participant) adds the time saved **including audit effort**:
projected reviewer minutes per 1,000 events for a grouped workflow versus
WildInbox (events needing review plus the audit share of automatic ones).

Study choices are stored in the study log only; they never become reviews, so
participants cannot overwrite each other's labels or the production queue.

## Before running it with people

The deployment must run policy `conservative/v2` (release
`finetune-e3-deep-balanced@7a25aea97c76`): under v1, every animal event said
"model is unsure", even next to a 95% suggestion, which would have skewed the
suggested condition. Check with `curl localhost:8000/version`, and upload the
study batch **after** activating v2 (decisions are made when a batch is
processed).

## 1. Prepare a study batch (organizer)

Use photos the deployment has not seen, and whose ground truth you know.
Already uploaded photos are recognised as duplicates and form no events, and a
sequence with even one duplicate frame is incomplete. So list the hashes the
deployment holds and skip every sequence that contains one:

```bash
docker compose exec -T postgres psql -U wildinbox -d wildinbox -tAc \
  "select distinct sha256 from images where sha256 is not null" > seen.txt
uv run python scripts/make_sample_batch.py --partition calibration --images 400 \
  --seed review-study --exclude-hashes seen.txt --out data/samples/review-study-1
uv run python scripts/simulate_deployment.py --batch-dir data/samples/review-study-1 --upload-only
```

The second command uploads one batch per camera and prints the batch ids. Its
upload `Idempotency-Key` comes from the directory name. Give every study batch
a new directory (`review-study-1`, `review-study-2`, ...); reusing one gets
`409 idempotency_conflict`. The sets need about 100 events with enough of each
label; the plan command says so if the batch is too small. Check that the
events were decided by the release and policy above:

```bash
curl -s "localhost:8000/events?batch_id=<batch id>&limit=500" \
  | jq -r '.events[].decision | "\(.model_release_id) \(.policy_version)"' | sort | uniq -c
```

## 2. Create the plan

```bash
uv run wildinbox study plan --name "study-1" \
  --batch-id <batch id> --batch-id <batch id> \
  --truth data/samples/review-study/truth.csv
```

It prints the study code (plan id) and records the protocol's SHA-256, so the
analysis refuses to run if the protocol file is edited afterwards.

Against a deployment with token authentication, `export WILDINBOX_TOKEN=<token>`
first: `study plan` and `study analyze` send it as a bearer token, and stop
with an error naming the refused request if it is missing or wrong.

## 3. Run sessions

For each participant, on a laptop with the deployment running:

1. Open `http://localhost:8501`, choose **Study** in the sidebar, and close the
   sidebar (the other pages show model suggestions).
2. The participant enters the study code and a participant code of their choice
   (no names), reads the consent text, and agrees.
3. Practice events, then block 1, a difficulty rating, block 2, and a final
   rating; about 15 minutes. Stay nearby but do not help or comment.

Keep conditions the same for everyone: same machine and screen, same time of
day if possible, and no interruptions (a decision over 180 s is excluded as an
interruption). Aim for at least 8 participants; with 4 arms, multiples of 4
keep the design balanced.

## 4. Analyse

```bash
uv run wildinbox study analyze --plan-id <study code> --out reports/study/study-1
```

Writes `README.md` (verdict, per-participant table, exclusions), `metrics.json`,
and `export.json` (every trial and rating, with ground truth). The verdict is
one of: descriptive only (too few participants); suggestions made review faster
without an accuracy drop beyond 5 points; faster but less accurate; or no
reliable speed gain.

## What it can and cannot tell you

- It measures the **released** system, where every event is reviewed: the
  question is whether showing suggestions speeds a person up, not whether
  automation removes reviews.
- Participants are not trained field reviewers unless you recruit them; report
  who took part.
- A small sample (8-12 people) detects only fairly large differences; the
  interval says how large a difference the data can rule out.
