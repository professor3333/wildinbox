#!/usr/bin/env bash
# Backup and restore drill on staging (docs/deployment.md#backups-and-restore).
#
#   WILDINBOX_TOKEN=<token> deploy/staging/restore_drill.sh
#
# Counts before the backup must be exactly what the restore brings back, a
# batch uploaded after the backup must be gone, reviews and originals must
# survive, and the restored stack must process new uploads. Needs at least one
# processed event. Replaces the database: run it on staging only.
set -euo pipefail
cd "$(dirname "$0")/../.."
wi=deploy/staging/wi
TOKEN=${WILDINBOX_TOKEN:?set WILDINBOX_TOKEN}; H="Authorization: Bearer $TOKEN"; API=http://localhost:8000
counts() { $wi exec -T postgres psql -U wildinbox -d wildinbox -At -c "SELECT json_build_object(
  'batches',(SELECT count(*) FROM batches), 'images',(SELECT count(*) FROM images),
  'events',(SELECT count(*) FROM events), 'predictions',(SELECT count(*) FROM predictions),
  'decisions',(SELECT count(*) FROM decisions), 'reviews',(SELECT count(*) FROM reviews),
  'jobs',(SELECT count(*) FROM jobs), 'releases',(SELECT count(*) FROM model_releases))"; }
echo "== $(date -u +%FT%TZ) add a review so the backup holds a human correction"
ev=$(curl -fsS -H "$H" "$API/events?limit=1" | jq -r ".events[0].id")
curl -fsS -H "$H" -H "Content-Type: application/json" -X POST "$API/events/$ev/reviews" \
  -d "{\"reviewer\": \"restore-drill\", \"outcome\": \"unresolved\", \"note\": \"backup and restore drill\"}" | jq -c "{id, event_id, confirmed_label}"
before=$(counts); echo "before backup: $before"
echo "== backup"; time deploy/staging/backup.sh | tee /tmp/backup.json
key=$(jq -r .key /tmp/backup.json)
echo "== upload one batch AFTER the backup (the restore must remove it)"
img=$(curl -fsS -H "$H" "$API/events?limit=1" | jq -r ".events[0].image_ids[0]")
curl -fsS -H "$H" -o /tmp/source.jpg "$API/images/$img/original"
python3 - > /tmp/after.jpg <<PY
import sys; d=open("/tmp/source.jpg","rb").read()
import time; p=b"restore-drill " + str(time.time()).encode(); sys.stdout.buffer.write(d[:2]+b"\xff\xfe"+(len(p)+2).to_bytes(2,"big")+p+d[2:])
PY
after_batch=$(curl -fsS -H "$H" -F "files=@/tmp/after.jpg;filename=after.jpg" "$API/batches" | jq -r .id)
echo "after-backup batch: $after_batch"; echo "after upload: $(counts)"
echo "== restore $key"; time deploy/staging/restore.sh "$key"
restored=$(counts); echo "after restore: $restored"
[[ $restored == "$before" ]] && echo "ok   counts identical to the backup" || { echo "FAIL counts differ"; exit 1; }
code=$(curl -s -o /dev/null -w "%{http_code}" -H "$H" "$API/batches/$after_batch")
[[ $code == 404 ]] && echo "ok   after-backup batch is gone (404)" || { echo "FAIL after-backup batch: $code"; exit 1; }
curl -fsS -H "$H" "$API/events/$ev" | jq -e ".reviews[-1].reviewer == \"restore-drill\"" >/dev/null && echo "ok   the review survived"
img_id=$(curl -fsS -H "$H" "$API/events/$ev" | jq -r ".images[0].id")
curl -fsS -H "$H" -o /tmp/thumb.jpg "$API/images/$img_id/thumbnail" && echo "ok   restored image loads from S3 ($(stat -c %s /tmp/thumb.jpg) byte thumbnail)"
curl -fsS "$API/ready" | jq -c "{status, model: .checks.model.detail}"
echo "== the restored system still processes new uploads"
new=$(curl -fsS -H "$H" -F "files=@/tmp/after.jpg;filename=after-restore.jpg" "$API/batches" | jq -r .id)
for i in $(seq 60); do s=$(curl -fsS -H "$H" "$API/batches/$new" | jq -r .status); [[ $s == completed* ]] && break; sleep 2; done
echo "new batch $new: $s"; [[ $s == completed ]] && echo "ok   processed after restore"
