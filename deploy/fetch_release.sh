#!/usr/bin/env bash
# Download the trained release bundle from GitHub, verify it, and unpack it.
#
#   deploy/fetch_release.sh [tag] [dir]      # defaults: v1.5.0, dist/release
#
# Checks the archive against its published SHA-256, then every file against
# the bundle's own SHA256SUMS. Needs curl and shasum (or sha256sum); no GitHub
# account. Then register it:
#   docker compose run --rm -v "$PWD/dist/release:/release:ro" worker \
#     wildinbox release register --activate --model-dir /release/model --policy /release/policy.json
set -euo pipefail
tag="${1:-v1.5.0}"
dir="${2:-dist/release}"
name=wildinbox-release-finetune-e3-deep-balanced-7a25aea97c76.tar.gz
url="https://github.com/professor3333/wildinbox/releases/download/$tag/$name"
sha() { if command -v shasum >/dev/null; then shasum -a 256 "$@"; else sha256sum "$@"; fi; }
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
(cd "$tmp" && curl -fsSLO "$url" && curl -fsSLO "$url.sha256" && sha -c "$name.sha256")
rm -rf "$dir" && mkdir -p "$dir"
tar -xzf "$tmp/$name" -C "$dir"
(cd "$dir" && sha -c SHA256SUMS)
chmod -R a+rX "$dir"   # readable by the containers' non-root user
echo "release bundle $tag verified -> $dir"
