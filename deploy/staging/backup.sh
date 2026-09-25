#!/usr/bin/env bash
# Back up PostgreSQL to the stack's S3 bucket (backups/postgres/, kept 30 days).
#
#   deploy/staging/backup.sh            # prints the S3 key of the new backup
#
# Originals and model weights already live in the same bucket, which is
# versioned, so the database dump is the only thing that needs copying.
set -euo pipefail
# cron runs with PATH=/usr/bin:/bin; the AWS CLI is a snap.
export PATH="/snap/bin:/usr/local/bin:$PATH"
cd "$(dirname "$0")/../.."
set -a; source /etc/wildinbox/stack.env; set +a
wi=deploy/staging/wi
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
dir=/var/tmp/wildinbox-backups; mkdir -p "$dir"
file="$dir/wildinbox-$stamp.dump"
$wi exec -T postgres pg_dump -U wildinbox -d wildinbox --format=custom --no-owner > "$file"
# A dump that pg_restore cannot list is not a backup.
tables=$($wi exec -T postgres pg_restore --list < "$file" | grep -c "TABLE DATA")
key="backups/postgres/wildinbox-$stamp.dump"
aws s3 cp --only-show-errors "$file" "s3://$WILDINBOX_BUCKET/$key" \
  --metadata "tables=$tables,git=$(git rev-parse --short HEAD)"
find "$dir" -name 'wildinbox-*.dump' -mtime +3 -delete
echo "{\"key\": \"$key\", \"bytes\": $(stat -c %s "$file"), \"tables\": $tables, \"at\": \"$stamp\"}"
