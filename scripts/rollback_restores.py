"""Reverting to a previous release restores its predictions.

    uv run python scripts/rollback_restores.py

Against the Docker Compose deployment, with the same photos each time:

Each upload re-encodes the photos identically (same pixels) with a per-batch
JPEG comment, since the deployment skips exact duplicates.

1. Batch 1 on the active release (the "previous" release).
2. Register (if needed) and activate the update candidate; batch 2 on it.
3. Roll back: re-activate the previous release; batch 3 on it.
4. Check that batch 3 reproduces batch 1 (every frame's calibrated
   probabilities and label, every event's decision), that batch 2 differs,
   and that no batch's stored results were rewritten by the switches.

Writes reports/update/rollback-restore.json.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from PIL import Image


def auth_headers() -> dict[str, str]:
    """Bearer token from WILDINBOX_TOKEN, for deployments that require one."""
    token = os.environ.get("WILDINBOX_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


TOLERANCE = 1e-5


def check(cond: bool, message: str) -> None:
    print(("ok   " if cond else "FAIL ") + message)
    if not cond:
        sys.exit(1)


def compose(*args: str) -> str:
    return subprocess.run(
        ["docker", "compose", *args], check=True, capture_output=True, text=True
    ).stdout


def tagged(data: bytes, tag: str) -> bytes:
    """The same pixels every time (quality 95), new bytes per batch (a JPEG
    comment), because the deployment skips photos it already holds."""
    buf = io.BytesIO()
    Image.open(io.BytesIO(data)).convert("RGB").save(buf, "JPEG", quality=95, comment=tag.encode())
    return buf.getvalue()


def upload(api: httpx.Client, batch_dir: Path, key: str) -> dict[str, Any]:
    meta = json.loads((batch_dir / "metadata.json").read_text())["files"]
    files = sorted((batch_dir / "images").iterdir())
    res = api.post(
        "/batches",
        files=[("files", (p.name, tagged(p.read_bytes(), key), "image/jpeg")) for p in files],
        data={"metadata": json.dumps({"files": {p.name: meta[p.name] for p in files}})},
        headers={"Idempotency-Key": key},
        timeout=300,
    )
    res.raise_for_status()
    batch: dict[str, Any] = res.json()
    for _ in range(900):
        s = api.get(f"/batches/{batch['id']}").json()
        if s["progress"]["finished"]:
            return s
        time.sleep(1)
    sys.exit(f"batch {batch['id']} did not finish")


def results(api: httpx.Client, batch_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """(frames by file name, decisions by the event's sorted file names)."""
    frames: dict[str, Any] = {}
    decisions: dict[str, Any] = {}
    events = api.get("/events", params={"batch_id": batch_id, "limit": 500}).json()["events"]
    for e in events:
        d = api.get(f"/events/{e['id']}").json()
        names = []
        for i in d["images"]:
            names.append(i["filename"])
            if i["prediction"]:
                frames[i["filename"]] = {
                    "label": i["prediction"]["suggested_label"],
                    "probs": i["prediction"]["calibrated_probabilities"],
                    "release": i["prediction"]["model_release_id"],
                }
        dec = d["decision"]
        decisions["|".join(sorted(names))] = {
            k: dec[k]
            for k in ("disposition", "suggested_label", "confidence", "reasons", "model_release_id")
        }
    return frames, decisions


def max_diff(a: dict[str, Any], b: dict[str, Any]) -> float:
    return max(abs(a[s]["probs"][c] - b[s]["probs"][c]) for s in a for c in a[s]["probs"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--gate", type=Path, default=Path("reports/update/finetune-e3-update1"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/finetune-e3-update1"))
    parser.add_argument("--batch-dir", type=Path, default=Path("data/samples/release-demo"))
    parser.add_argument("--out", type=Path, default=Path("reports/update/rollback-restore.json"))
    args = parser.parse_args()
    api = httpx.Client(base_url=args.url, timeout=60, headers=auth_headers())
    run = time.strftime("%Y%m%dT%H%M%S")
    gate = json.loads((args.gate / "metrics.json").read_text())
    candidate = gate["candidate_release"]
    previous = api.get("/version").json()["active_release"]["id"]
    check(previous != candidate, f"previous release: {previous}")

    b1 = upload(api, args.batch_dir, f"rollback-restore-{run}-1")
    check(b1["release"]["id"] == previous, f"batch 1 ran on {previous}")
    frames1, dec1 = results(api, b1["id"])

    registered = {r["id"] for r in api.get("/releases").json()["releases"]}
    if candidate in registered:
        compose(
            "exec",
            "-T",
            "api",
            "wildinbox",
            "release",
            "activate",
            candidate,
            "--note",
            "rollback-restore demonstration: candidate",
        )
    else:
        compose(
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
            "rollback-restore demonstration: candidate",
        )
    check(
        api.get("/version").json()["active_release"]["id"] == candidate,
        f"activated the candidate {candidate}",
    )
    b2 = upload(api, args.batch_dir, f"rollback-restore-{run}-2")
    check(b2["release"]["id"] == candidate, "batch 2 ran on the candidate")
    frames2, dec2 = results(api, b2["id"])

    compose(
        "exec",
        "-T",
        "api",
        "wildinbox",
        "release",
        "activate",
        previous,
        "--note",
        "rollback-restore demonstration: roll back",
    )
    check(
        api.get("/version").json()["active_release"]["id"] == previous, f"rolled back to {previous}"
    )
    b3 = upload(api, args.batch_dir, f"rollback-restore-{run}-3")
    check(b3["release"]["id"] == previous, "batch 3 ran on the previous release")
    frames3, dec3 = results(api, b3["id"])

    check(set(frames1) == set(frames3) == set(frames2), f"{len(frames1)} frames scored each time")
    restored = max_diff(frames1, frames3)
    labels_same = sum(frames1[s]["label"] == frames3[s]["label"] for s in frames1)
    check(restored <= TOLERANCE, f"rollback restores probabilities (max difference {restored:.2e})")
    check(labels_same == len(frames1), f"rollback restores all {len(frames1)} frame labels")
    check(dec1 == dec3, f"rollback restores all {len(dec1)} event decisions")
    changed = sum(frames1[s]["label"] != frames2[s]["label"] for s in frames1)
    cand_diff = max_diff(frames1, frames2)
    dec_changed = sum(dec1[k] != dec2[k] for k in dec1)
    check(
        cand_diff > TOLERANCE,
        f"the candidate's predictions differ (max {cand_diff:.3f}; "
        f"{changed} frame labels, {dec_changed} decisions)",
    )
    again1, _ = results(api, b1["id"])
    again2, _ = results(api, b2["id"])
    check(
        again1 == frames1 and again2 == frames2,
        "stored results of batches 1 and 2 were not rewritten by the switches",
    )

    report = {
        "run": run,
        "photos": len(frames1),
        "previous_release": previous,
        "candidate_release": candidate,
        "batches": {"1": b1["id"], "2": b2["id"], "3": b3["id"]},
        "rollback": {
            "max_probability_difference": restored,
            "frame_labels_identical": labels_same,
            "event_decisions_identical": sum(dec1[k] == dec3[k] for k in dec1),
            "events": len(dec1),
        },
        "candidate_vs_previous": {
            "max_probability_difference": cand_diff,
            "frame_labels_changed": changed,
            "event_decisions_changed": dec_changed,
        },
        "tolerance": TOLERANCE,
        "activations": api.get("/releases").json()["activations"][:3],
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
