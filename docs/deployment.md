# Staging deployment

How to deploy a pinned WildInbox release to one AWS VM with Docker Compose,
check it, back it up, restore it, and tear it down. Everything below was run
as written for the Stage 12 staging deployment; the measurements are in
[`reports/staging/`](../reports/staging/README.md).

```
 your machine                          AWS (one region)
 ─────────────                         ───────────────────────────────────────────────
 browser / scripts ──SSH tunnel──▶ VM (Ubuntu 24.04, Docker Compose)
   localhost:8000  (API)              api ─ worker(s) ─ ui ─ postgres ─ redis
   localhost:8501  (review UI)          │        │                  │
                                        └────────┴──── instance role ▼
                                                       S3 bucket (private, versioned)
                                                       originals/  releases/weights/
                                                       backups/postgres/
```

- **Only SSH (port 22) is open, and only to one address.** The API and UI
  listen on the VM's localhost; you reach them through an SSH tunnel, so
  access tokens never cross the internet in plain HTTP.
- **Large artifacts live in S3**, not on the VM: uploaded originals and model
  weights (content-addressed, SHA-256 checked when loaded), and database
  backups. The VM's disk holds PostgreSQL, Redis, and the Docker images.
- **No long-lived AWS keys.** Containers use the VM's instance role, which
  can read and write only this stack's bucket.

## What you need

| | |
|---|---|
| AWS account | permission to create a CloudFormation stack with an IAM role (`CAPABILITY_IAM`), EC2, and S3 |
| AWS CLI v2 | configured for the account (`aws sts get-caller-identity` works) |
| An SSH key pair | e.g. `~/.ssh/id_ed25519`; imported into EC2 below |
| The release bundle | `wildinbox-release-finetune-e3-deep-balanced-7a25aea97c76.tar.gz` (see [Release bundle](#release-bundle)) |
| This repository | checked out at the tag you deploy, on your machine (for the template and scripts) |

## 1. Create the stack

The template [`deploy/aws/staging.yaml`](../deploy/aws/staging.yaml) creates
the VM (`m7i-flex.large`: 2 vCPU, 8 GiB, 40 GB encrypted gp3; pass `InstanceType=` for another), the bucket, the
instance role, and a security group allowing SSH from `AllowedCidr` only.
User data installs Docker, Compose, Git, and the AWS CLI, clones the
repository at `GitRef` into `/opt/wildinbox`, and writes the stack's
non-secret facts to `/etc/wildinbox/stack.env`.

```bash
export AWS_REGION=us-east-1 TAG=v0.2.0
aws ec2 import-key-pair --key-name wildinbox-staging \
  --public-key-material fileb://$HOME/.ssh/id_ed25519.pub
aws cloudformation deploy --stack-name wildinbox-staging \
  --template-file deploy/aws/staging.yaml --capabilities CAPABILITY_IAM \
  --parameter-overrides KeyName=wildinbox-staging GitRef=$TAG \
    AllowedCidr=$(curl -s https://checkip.amazonaws.com)/32
aws cloudformation describe-stacks --stack-name wildinbox-staging \
  --query 'Stacks[0].Outputs' --output table     # PublicIp, BucketName, InstanceId
```

Add the VM to `~/.ssh/config` (the forwards are the tunnel):

```
Host wildinbox-staging
  HostName <PublicIp>
  User ubuntu
  IdentityFile ~/.ssh/id_ed25519
  LocalForward 8000 localhost:8000
  LocalForward 8501 localhost:8501
  ServerAliveInterval 30
```

Wait for first-boot setup to finish (about two minutes):

```bash
ssh wildinbox-staging 'cloud-init status --wait && ls /var/lib/cloud/instance/wildinbox-ready'
```

## 2. Configure secrets on the VM

Secrets exist only in `/etc/wildinbox/secrets.env` on the VM (mode 600). They
are never committed, never copied off the VM, and never in the image. The
template is [`deploy/staging/secrets.env.example`](../deploy/staging/secrets.env.example).

```bash
ssh wildinbox-staging
cd /opt/wildinbox
install -m 600 deploy/staging/secrets.env.example /etc/wildinbox/secrets.env
sed -i "s/^POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=$(openssl rand -hex 24)/" /etc/wildinbox/secrets.env
```

Create tokens on your own machine (in the repository checkout), one per
person or service:

```bash
uv run wildinbox token new owner
```

`token new` prints the token (give it to that person; it is stored nowhere)
and a `{"owner": "<sha256>"}` entry. Put every entry into one JSON object on
the `WILDINBOX_API_TOKENS=` line of `/etc/wildinbox/secrets.env`. Only hashes
are on the VM, so a copy of the file does not grant access.

| Variable | Meaning |
|---|---|
| `POSTGRES_PASSWORD` | database password (random, generated above) |
| `WILDINBOX_API_TOKENS` | JSON object of principal name → SHA-256 of that principal's token |
| `WILDINBOX_EXPECTED_RELEASE` | the release this deployment must serve; readiness fails otherwise |
| `WILDINBOX_WORKERS` | worker processes (default 1) |
| `WILDINBOX_TORCH_THREADS` | PyTorch threads per worker (unset: one per vCPU) |

`/etc/wildinbox/stack.env` (bucket, region, Git ref) is written by the
template and holds nothing secret. Compose refuses to start if any required
value is missing.

**Rotating a token:** create a new one, replace the principal's hash, and run
`deploy/staging/wi up -d` (the API restarts with the new set). Removing an
entry revokes that token.

## 3. Deploy the release

Copy the release bundle to the VM and run the deploy script:

```bash
scp wildinbox-release-finetune-e3-deep-balanced-7a25aea97c76.tar.gz wildinbox-staging:/tmp/
ssh wildinbox-staging 'cd /opt/wildinbox && deploy/staging/deploy.sh /tmp/wildinbox-release-*.tar.gz'
```

[`deploy.sh`](../deploy/staging/deploy.sh) builds the image from the checked
out tag, starts PostgreSQL and Redis, applies migrations, verifies the
bundle's checksums, registers the release (weights go to S3; registering an
existing release is a no-op, a different release under the same id is
refused), activates `WILDINBOX_EXPECTED_RELEASE` if it is not already active,
starts the API, workers, and UI, and **waits for readiness**. It then prints
`/ready` and the release list, and installs the nightly backup.

### Deploy check

Readiness is the API's health check in staging, so `deploy.sh` only succeeds
when the expected release is active **and its weights load** (SHA-256
verified). Check it again at any time, and confirm every version the API
reports:

```bash
ssh -fN wildinbox-staging                          # opens the tunnel in the background
curl -s localhost:8000/ready | jq                  # database, queue, object_store, model: ok
export WILDINBOX_TOKEN=<your token>
curl -s -H "Authorization: Bearer $WILDINBOX_TOKEN" localhost:8000/version | jq
```

`/version` must report release `finetune-e3-deep-balanced@7a25aea97c76`,
weights SHA-256 `3ab6fec2…9b360`, preprocessing `6d9a950a6543`, calibration
`55cdb7daad08`, and policy `conservative/v2+378312635a29`.

Open the review UI at <http://localhost:8501> and sign in with your token.

### Operating the stack

`deploy/staging/wi` is `docker compose` with the staging file and env files
loaded; use it for everything on the VM:

```bash
deploy/staging/wi ps
deploy/staging/wi logs -f --since 10m worker | jq -c 'select(.job_id)'   # JSON logs
deploy/staging/wi up -d --scale worker=2 --no-recreate                  # add a worker
deploy/staging/wi restart worker
```

Logs are one JSON object per line (`ts`, `level`, `logger`, `message`, plus
fields such as `request_id`, `route`, `status`, `duration_ms`, `principal`,
`job_id`, `batch_id`, `release`). Docker rotates them at 5 × 20 MB per
container. Every response carries `X-Request-ID` (sent by the client or
generated), which is the key to find it in the logs.

## Limits

Enforced by the API before anything is stored (see `.env.example` to change):

| Limit | Default | Response |
|---|---|---|
| Files per batch | 2,000 | `413 too_many_files` |
| Bytes per batch | 1 GiB | `413 batch_too_large` (checked from `Content-Length` before reading) |
| Bytes per file | 20 MiB | that file gets an `invalid` record; the rest of the batch proceeds |
| File types | JPEG, PNG, detected from content | others get an `invalid` record with the reason |

## Backups and restore

**What is backed up.** PostgreSQL (jobs, images, events, predictions,
decisions, reviews, releases) is dumped nightly at 02:30 UTC by
[`backup.sh`](../deploy/staging/backup.sh) to
`s3://<bucket>/backups/postgres/`, kept 30 days. A dump is only uploaded after
`pg_restore --list` reads it. Originals and model weights are already in S3;
the bucket is versioned, so an overwritten or deleted object stays
recoverable for 30 days. Redis holds only queue messages, and a lost queue
message is recovered from PostgreSQL, so Redis is not backed up.

```bash
deploy/staging/backup.sh                       # on demand; prints the new key
aws s3 ls s3://<bucket>/backups/postgres/      # what exists
tail /var/log/wildinbox-backup.log             # nightly runs
```

**Restore** replaces the database with a dump, applies migrations, and waits
for readiness:

```bash
deploy/staging/restore.sh latest
deploy/staging/restore.sh backups/postgres/wildinbox-20260925T123456Z.dump
```

The API, workers, and UI stop during the restore. Anything written after the
backup is lost from the database. Originals uploaded after it stay in S3 but
are no longer referenced. A job that was running at backup time is recovered
by its lease after the restore.

**Rehearse it** with the drill. It adds a review, backs up, uploads a batch
after the backup, restores, and checks that the counts match the backup, the
later batch is gone, originals still load, and new uploads process. It replaces
the database, so run it on staging only:

```bash
WILDINBOX_TOKEN=<token> deploy/staging/restore_drill.sh
```

The recorded run is in the
[staging report](../reports/staging/README.md#backup-and-restore).

## Upgrade and rollback

- **New code:** `git fetch --tags && git checkout <new tag>` in
  `/opt/wildinbox`, update `WILDINBOX_GIT_REF` in `/etc/wildinbox/stack.env`,
  run `deploy/staging/backup.sh`, then `deploy/staging/deploy.sh`.
- **New model:** copy its bundle, set `WILDINBOX_EXPECTED_RELEASE` to the new
  id, and run `deploy/staging/deploy.sh <bundle>`.
- **Roll back the model:** set `WILDINBOX_EXPECTED_RELEASE` back to the
  previous id, then
  `deploy/staging/wi run --rm --no-deps worker wildinbox release activate <previous-id>`
  and `deploy/staging/wi up -d`. Running jobs keep the release they started
  with (see [README](../README.md#deployment-and-rollback)).
- **Roll back the code:** check out the previous tag and run `deploy.sh`. If
  the new release ran a migration, restore the backup taken before the
  upgrade.

## Release bundle

A bundle is exactly what `wildinbox release register` reads, plus checksums:

```
model/meta.json  model/model.pt  policy.json  SHA256SUMS
```

Built on the training machine with

```bash
deploy/release_bundle.sh models/finetune-e3-deep-balanced \
  reports/policy/finetune-e3-deep-balanced-v2/policy.json dist/
```

The model and policy are reproduced from the pinned manifests as described
in the [README](../README.md#reproduce-the-data-and-models). Registration
re-checks every cross-reference (the calibration was fitted to these weights,
the class order and preprocessing match), so a mismatched bundle is refused.

## Tear down

```bash
aws cloudformation delete-stack --stack-name wildinbox-staging
aws s3 rb s3://<bucket> --force        # the bucket is retained on purpose; this deletes all data
aws ec2 delete-key-pair --key-name wildinbox-staging
```

The versioned bucket keeps old object versions; delete them too (or let the
30-day lifecycle rule expire them) before `rb` succeeds.

## Cost

On-demand prices in us-east-1 when measured; see the
[staging report](../reports/staging/README.md#cost-assumptions).
