#!/usr/bin/env bash
# Delete every object version and delete marker in the stack's bucket, then the
# bucket. The bucket is versioned and retained by the stack, so
# `aws s3 rb --force` alone leaves old versions behind and fails.
#
#   deploy/aws/empty_bucket.sh <bucket>        # needs the AWS CLI v2 and jq
set -euo pipefail
bucket="${1:?usage: empty_bucket.sh <bucket>}"
while :; do
  page=$(aws s3api list-object-versions --bucket "$bucket" --max-keys 1000 --no-paginate --output json)
  batch=$(jq -c '{Objects: [((.Versions // [])[], (.DeleteMarkers // [])[]) | {Key, VersionId}], Quiet: true}' <<<"$page")
  n=$(jq '.Objects | length' <<<"$batch")
  [[ $n == 0 ]] && break
  aws s3api delete-objects --bucket "$bucket" --delete "$batch" >/dev/null
  echo "deleted $n object versions"
done
aws s3api delete-bucket --bucket "$bucket"
echo "deleted bucket $bucket"
