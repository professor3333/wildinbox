"""Stage 9 acceptance demo against the Docker Compose deployment.

    uv run python scripts/crash_demo.py                # kill + restart halfway
    uv run python scripts/crash_demo.py --no-kill      # timing only

Uploads a batch (plus one corrupt file), SIGKILLs the worker container once
about half the images are scored, restarts it, and waits for completion. Then
checks, through the API and directly in PostgreSQL, that no input was lost, no
prediction or decision was duplicated, no event was finalized before the batch
completed, and the corrupt file has an explicit error. Requires a registered,
active non-test release (see docs/api.md).
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


CORRUPT = ("corrupt.jpg", b"\xff\xd8\xff\xe0" + b"this is not image data" * 50)


def check(cond: bool, message: str) -> None:
    print(("ok   " if cond else "FAIL ") + message)
    if not cond:
        sys.exit(1)


def compose(*args: str) -> str:
    return subprocess.run(
        ["docker", "compose", *args], check=True, capture_output=True, text=True
    ).stdout


def sql(query: str) -> list[list[str]]:
    out = compose(
        "exec",
        "-T",
        "postgres",
        "psql",
        "-U",
        "wildinbox",
        "-d",
        "wildinbox",
        "-At",
        "-F",
        "|",
        "-c",
        query,
    )
    return [line.split("|") for line in out.strip().splitlines() if line]


def upload(client: httpx.Client, batch_dir: Path) -> dict[str, Any]:
    images = sorted((batch_dir / "images").iterdir())
    files = [("files", (p.name, p.read_bytes(), "application/octet-stream")) for p in images]
    files.append(("files", (CORRUPT[0], CORRUPT[1], "application/octet-stream")))
    meta = (batch_dir / "metadata.json").read_text()
    res = client.post("/batches", files=files, data={"metadata": meta}, timeout=600)
    res.raise_for_status()
    return res.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--batch-dir", type=Path, default=Path("data/samples/dev-1000"))
    parser.add_argument("--kill-at", type=float, default=0.5, help="fraction scored before kill")
    parser.add_argument("--no-kill", action="store_true")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--report", type=Path, default=None, help="write results as JSON")
    args = parser.parse_args()

    client = httpx.Client(base_url=args.url, timeout=30, headers=auth_headers())
    version = client.get("/version").json()["active_release"]
    check(
        version is not None and not version["is_test"],
        f"active release {version and version['id']}",
    )

    t0 = time.monotonic()
    batch = upload(client, args.batch_dir)
    t_uploaded = time.monotonic()
    if batch.get("duplicate_request"):
        sys.exit(
            "FAIL this exact batch was already uploaded to this deployment; the API returned "
            "the existing one. Reset the stack (docker compose down -v) or use another batch."
        )
    bid = batch["id"]
    print(f"batch {bid}: {batch['counts']['images']} files uploaded in {t_uploaded - t0:.1f}s")

    killed: dict[str, Any] | None = None
    last = None
    while time.monotonic() - t0 < args.timeout:
        s = client.get(f"/batches/{bid}").json()
        p = s["progress"]
        if p["images_scored"] != last:
            elapsed = time.monotonic() - t0
            print(f"  {elapsed:6.1f}s scored {p['images_scored']}/{p['images_to_score']}")
            last = p["images_scored"]
        if (
            not args.no_kill
            and killed is None
            and not p["finished"]
            and p["images_scored"] >= args.kill_at * p["images_to_score"]
        ):
            events_before = s["counts"]["events"]
            compose("kill", "-s", "SIGKILL", "worker")
            killed = {"at_seconds": time.monotonic() - t0, "scored": p["images_scored"]}
            print(f"  SIGKILL worker at {p['images_scored']} scored")
            time.sleep(10)
            s = client.get(f"/batches/{bid}").json()
            frozen = s["progress"]["images_scored"]
            check(
                s["job"]["status"] == "running", "job still marked running (lease not yet expired)"
            )
            check(
                events_before == 0 and s["counts"]["events"] == 0,
                "no event finalized before completion",
            )
            killed["scored_after_kill"] = frozen
            compose("start", "worker")
            killed["restarted_at_seconds"] = time.monotonic() - t0
            print("  worker restarted; waiting for its lease to expire and the job to resume")
        if p["finished"]:
            break
        time.sleep(1)
    t_done = time.monotonic()

    s = client.get(f"/batches/{bid}").json()
    p = s["progress"]
    check(
        p["finished"] and s["status"] == "completed_with_errors", f"batch finished: {s['status']}"
    )
    check(
        p["images_scored"] == p["images_to_score"], f"no lost inputs: {p['images_scored']} scored"
    )
    failures = s["failures"]
    check(
        [f["filename"] for f in failures] == [CORRUPT[0]]
        and failures[0]["error"].startswith("unreadable"),
        f"corrupt file has an explicit error: {failures[0]['error'][:60] if failures else None}",
    )
    preds = sql(
        "select count(*), count(distinct p.image_id) from predictions p join images i "
        f"on i.id = p.image_id where i.batch_id = '{bid}'"
    )[0]
    check(preds[0] == preds[1] == str(p["images_scored"]), f"no duplicate predictions: {preds}")
    dec = sql(
        "select count(*), count(distinct d.event_id), (select count(*) from events where "
        f"batch_id = '{bid}') from decisions d join events e on e.id = d.event_id "
        f"where e.batch_id = '{bid}'"
    )[0]
    check(dec[0] == dec[1] == dec[2], f"one decision per event: {dec}")
    if killed:
        check(s["job"]["attempts"] == 2, f"resumed as attempt 2 (attempts={s['job']['attempts']})")

    result = {
        "batch_id": bid,
        "release": version["id"],
        "files": s["counts"]["images"],
        "images_scored": p["images_scored"],
        "events": s["counts"]["events"],
        "upload_seconds": round(t_uploaded - t0, 1),
        "processing_seconds": round(t_done - t_uploaded, 1),
        "images_per_second": round(p["images_scored"] / (t_done - t_uploaded), 2),
        "killed": killed,
        "job": {k: s["job"][k] for k in ("attempts", "status", "error")},
    }
    print(json.dumps(result, indent=2))
    if args.report:
        args.report.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
