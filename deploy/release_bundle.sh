#!/usr/bin/env bash
# Package a trained model and its policy artifact for deployment.
#
#   deploy/release_bundle.sh models/finetune-e3-deep-balanced \
#     reports/policy/finetune-e3-deep-balanced-v2/policy.json dist/
#
# The bundle holds exactly what `wildinbox release register` reads (meta.json,
# model.pt, policy.json) plus SHA256SUMS; registration re-checks every
# cross-reference, so a mismatched bundle is refused on the VM.
set -euo pipefail
export COPYFILE_DISABLE=1  # no macOS ._ files in the archive
model_dir="${1:?model dir}"; policy="${2:?policy.json}"; out="${3:-dist}"
name="$(jq -r .name "$model_dir/meta.json")@$(jq -r .artifact_version "$policy")"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/model" "$out"
cp "$model_dir/meta.json" "$model_dir/model.pt" "$tmp/model/"
cp "$policy" "$tmp/policy.json"
(cd "$tmp" && shasum -a 256 model/meta.json model/model.pt policy.json > SHA256SUMS)
bundle="$out/wildinbox-release-${name//@/-}.tar.gz"
tar -czf "$bundle" -C "$tmp" .
echo "$name"
shasum -a 256 "$bundle"
