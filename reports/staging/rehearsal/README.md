# Rehearsal: a fresh deployment of v1.4.0 from the guide

Acceptance gate: *another person can deploy the pinned release using the
instructions.* On 2026-09-25 a second, empty stack (`wildinbox-rehearsal`) was
created from the `v1.4.0` tag and deployed by following
[docs/deployment.md](../../../docs/deployment.md) step by step, then torn down.

| Step in the guide | Result |
|---|---|
| 1. Create the stack (`GitRef=v1.4.0`) | `CREATE_COMPLETE`; the VM checked out `v1.4.0` and wrote `/etc/wildinbox/stack.env` |
| Wait for first boot | `cloud-init status: done`; marker present |
| 2. Secrets (example file, generated password, `token new owner`) | `secrets.env` mode 600, hashes only |
| 3. `deploy.sh <bundle>` | [deploy.log](deploy.log): bundle checksums OK; release registered and activated; ready; **3 min 33 s** from an empty VM |
| Deploy check | `/ready` all four checks ok; `401` without a token; `/version` matched the guide (release, weights, preprocessing, calibration, policy); review UI up |
| Use it | [demo.log](demo.log): committed sample uploaded through the tunnel, 13 events to review, a correction, export with provenance |
| Backup under cron's environment | [backup.log](backup.log): dump in S3, 15 tables |
| Tear down | stack deleted; 39 object versions and the bucket deleted |

## What the rehearsal changed in the guide

- **Tokens:** assembling the JSON line by hand was the one error-prone step.
  `token new` now takes several names and prints the finished
  `WILDINBOX_API_TOKENS='…'` line.
- **Tear down:** the bucket is versioned, so `aws s3 rb --force` alone would
  fail. The rehearsal deleted the versions with the S3 API calls that
  `deploy/aws/empty_bucket.sh` now makes (list object versions, delete objects,
  delete bucket).
- **Load-test results** go outside the checkout. Results written inside it
  blocked the documented `git checkout <new tag>` upgrade of the main staging
  VM until they were moved.

The main staging VM was upgraded to `v1.4.0` through the guide's upgrade
path: backup, check out the tag, `deploy.sh`, ready. It serves the same release.
