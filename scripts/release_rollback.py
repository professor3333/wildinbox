"""Release a candidate that passed the update gate, then roll back.

    uv run python scripts/make_sample_batch.py --partition calibration --images 60 \\
        --seed release-demo --out data/samples/release-demo
    uv run python scripts/release_rollback.py

1. Refuse unless the gate report says promote.
2. Register the candidate inside the deployment and activate it.
3. Process batch A: it must run on the candidate.
4. Roll back: re-activate the previous release.
5. Process batch B: it must run on the previous release, while batch A keeps
   the candidate's predictions and decisions (provenance never rewritten).
6. Write reports/update/release-log.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx


def auth_headers() -> dict[str, str]:
    """Bearer token from WILDINBOX_TOKEN, for deployments that require one."""
    token = os.environ.get("WILDINBOX_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def check(cond: bool, message: str) -> None:
    print(("ok   " if cond else "FAIL ") + message)
    if not cond:
        sys.exit(1)


def compose(*args: str) -> str:
    return subprocess.run(
        ["docker", "compose", *args], check=True, capture_output=True, text=True
    ).stdout


def upload(api: httpx.Client, files: list[Path], meta: dict[str, Any], key: str) -> dict[str, Any]:
    res = api.post(
        "/batches",
        files=[("files", (p.name, p.read_bytes(), "image/jpeg")) for p in files],
        data={"metadata": json.dumps({"files": {p.name: meta[p.name] for p in files}})},
        headers={"Idempotency-Key": key},
        timeout=300,
    )
    res.raise_for_status()
    batch: dict[str, Any] = res.json()
    for _ in range(600):
        s = api.get(f"/batches/{batch['id']}").json()
        if s["progress"]["finished"]:
            return s
        time.sleep(1)
    sys.exit(f"batch {batch['id']} did not finish")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--gate", type=Path, default=Path("reports/update/finetune-e3-update1"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/finetune-e3-update1"))
    parser.add_argument("--batch-dir", type=Path, default=Path("data/samples/release-demo"))
    parser.add_argument("--log", type=Path, default=Path("reports/update/release-log.json"))
    args = parser.parse_args()
    api = httpx.Client(base_url=args.url, timeout=60, headers=auth_headers())

    gate = json.loads((args.gate / "metrics.json").read_text())
    check(gate["promote"], f"gate passed for {gate['candidate_release']}")
    previous = api.get("/version").json()["active_release"]["id"]
    check(previous == gate["deployed_release"], f"deployed release is {previous}")

    out = compose(
        "run",
        "--rm",
        "-v",
        f"{args.model_dir.resolve()}:/app/{args.model_dir}:ro",
        "-v",
        f"{(args.gate / 'policy.json').resolve()}:/app/candidate-policy.json:ro",
        "worker",
        "wildinbox",
        "release",
        "register",
        "--model-dir",
        str(args.model_dir),
        "--policy",
        "/app/candidate-policy.json",
        "--activate",
        "--note",
        "promoted: passed the update gate",
    )
    print("  " + "\n  ".join(line for line in out.splitlines() if line.strip()))
    version = api.get("/version").json()["active_release"]
    check(
        version["id"] == gate["candidate_release"],
        f"active release is the candidate: {version['id']}",
    )

    meta = json.loads((args.batch_dir / "metadata.json").read_text())["files"]
    sequences = sorted({m["sequence_id"] for m in meta.values()})
    half = set(sequences[: len(sequences) // 2])
    images = sorted((args.batch_dir / "images").iterdir())
    files_a = [p for p in images if meta[p.name]["sequence_id"] in half]
    files_b = [p for p in images if meta[p.name]["sequence_id"] not in half]

    a = upload(api, files_a, meta, "release-demo-a")
    check(
        a["release"]["id"] == gate["candidate_release"],
        f"batch A ran on the candidate ({len(files_a)} photos)",
    )

    compose(
        "exec",
        "-T",
        "api",
        "wildinbox",
        "release",
        "activate",
        previous,
        "--note",
        "rollback demonstration",
    )
    check(
        api.get("/version").json()["active_release"]["id"] == previous, f"rolled back to {previous}"
    )

    b = upload(api, files_b, meta, "release-demo-b")
    check(
        b["release"]["id"] == previous,
        f"batch B ran on the previous release ({len(files_b)} photos)",
    )

    events_a = api.get("/events", params={"batch_id": a["id"], "limit": 500}).json()["events"]
    check(
        all(e["decision"]["model_release_id"] == gate["candidate_release"] for e in events_a),
        f"batch A's {len(events_a)} decisions still name the candidate after the rollback",
    )
    history = api.get("/releases").json()["activations"]
    log = {
        "gate": str(args.gate / "README.md"),
        "candidate_release": gate["candidate_release"],
        "previous_release": previous,
        "batch_a": {"id": a["id"], "release": a["release"]["id"], "events": a["counts"]["events"]},
        "batch_b": {"id": b["id"], "release": b["release"]["id"], "events": b["counts"]["events"]},
        "activations": history,
    }
    args.log.write_text(json.dumps(log, indent=2) + "\n")
    print(f"release log -> {args.log}")


if __name__ == "__main__":
    main()
