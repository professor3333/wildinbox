#!/usr/bin/env bash
# Restore PostgreSQL from a backup in the stack's bucket.
#
#   deploy/staging/restore.sh latest
#   deploy/staging/restore.sh backups/postgres/wildinbox-20260925T023000Z.dump
#
# Stops the API, workers, and UI; replaces the database with the dump; applies
# migrations (a dump from an older release is upgraded); starts everything and
# waits for readiness. Jobs that were running are recovered by their leases.
set -euo pipefail
# cron runs with PATH=/usr/bin:/bin; the AWS CLI is a snap.
export PATH="/snap/bin:/usr/local/bin:$PATH"
cd "$(dirname "$0")/../.."
set -a; source /etc/wildinbox/stack.env; set +a
wi=deploy/staging/wi
key="${1:?usage: restore.sh latest|<s3 key>}"
if [[ $key == latest ]]; then
  key="backups/postgres/$(aws s3 ls "s3://$WILDINBOX_BUCKET/backups/postgres/" | awk '{print $4}' | sort | tail -1)"
fi
file="$(mktemp --suffix=.dump)"; trap 'rm -f "$file"' EXIT
aws s3 cp --only-show-errors "s3://$WILDINBOX_BUCKET/$key" "$file"
echo "== restoring $key ($(stat -c %s "$file") bytes)"
$wi stop api worker ui
$wi exec -T postgres dropdb -U wildinbox --if-exists --force wildinbox
$wi exec -T postgres createdb -U wildinbox wildinbox
$wi exec -T postgres pg_restore -U wildinbox -d wildinbox --no-owner --exit-on-error < "$file"
$wi run --rm migrate
$wi up -d --wait
echo "== restored $key"
