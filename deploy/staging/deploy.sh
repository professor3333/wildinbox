#!/usr/bin/env bash
# Deploy (or redeploy) the checked-out release on the staging VM.
#
#   deploy/staging/deploy.sh [release-bundle.tar.gz]
#
# Builds the image, applies migrations, registers and activates the model
# release from the bundle (idempotent), starts everything, waits for
# readiness, and prints the versions the API serves. Run from /opt/wildinbox.
set -euo pipefail
cd "$(dirname "$0")/../.."
wi=deploy/staging/wi
for f in /etc/wildinbox/stack.env /etc/wildinbox/secrets.env; do
  [[ -r $f ]] || { echo "missing $f (see docs/deployment.md)" >&2; exit 1; }
done
[[ $(stat -c %a /etc/wildinbox/secrets.env) == 600 ]] || {
  echo "/etc/wildinbox/secrets.env must be chmod 600" >&2; exit 1; }
set -a; source /etc/wildinbox/stack.env; source /etc/wildinbox/secrets.env; set +a

echo "== build $(git rev-parse --short HEAD) ($(git describe --tags --always))"
$wi build
$wi up -d --wait postgres redis
$wi run --rm migrate

if [[ $# -ge 1 ]]; then
  bundle="$(realpath "$1")"
  echo "== register release from $bundle"
  tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
  tar -xzf "$bundle" -C "$tmp"
  (cd "$tmp" && sha256sum -c SHA256SUMS)
  chmod -R a+rX "$tmp"
  $wi run --rm --no-deps -v "$tmp:/release:ro" worker \
    wildinbox release register --model-dir /release/model --policy /release/policy.json \
      --note "staging deploy $(git describe --tags --always)"
  current="$($wi run --rm --no-deps worker wildinbox release list | awk '/^\*/ {print $2}')"
  if [[ $current != "$WILDINBOX_EXPECTED_RELEASE" ]]; then
    $wi run --rm --no-deps worker wildinbox release activate "$WILDINBOX_EXPECTED_RELEASE" \
      --note "staging deploy $(git describe --tags --always)"
  fi
fi

echo "== start (waits for readiness: expected release active and loaded)"
$wi up -d --wait --remove-orphans
curl -fsS localhost:8000/ready | jq .
$wi run --rm --no-deps worker wildinbox release list

# Nightly database backup at 02:30 UTC.
cron=/etc/cron.d/wildinbox-backup
line="30 2 * * * ubuntu cd $PWD && deploy/staging/backup.sh >> /var/log/wildinbox-backup.log 2>&1"
if [[ ! -f $cron ]] || ! grep -qF "$line" "$cron"; then
  echo "$line" | sudo tee "$cron" >/dev/null
  sudo touch /var/log/wildinbox-backup.log && sudo chown ubuntu /var/log/wildinbox-backup.log
fi
echo "== deployed"
