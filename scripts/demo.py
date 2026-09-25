"""The WildInbox demo, end to end, against a running deployment.

    docker compose up -d --build --wait
    uv run python scripts/demo.py                    # then open http://localhost:8501

1. Upload the committed sample batch (samples/cct-dev) with its metadata.
2. Follow processing to completion.
3. Show the most uncertain events and why they need review.
4. Record a correction (the demo reviewer knows the ground truth).
5. Download the observation export and show the corrected row.

Re-running reuses the already uploaded batch.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

REASONS = {
    "low_confidence": "model is unsure",
    "conflicting_frames": "frames disagree",
    "possible_unknown": "may be an unsupported species",
    "processing_failure": "a frame could not be processed",
    "species_not_validated": "species not validated for automation",
    "automation_disabled": "automation is off",
}


def step(n: int, text: str) -> None:
    print(f"\n{n}. {text}")


def truth_by_file(sample: Path) -> dict[str, str]:
    """Event-level truth per file: the animal in its sequence, else empty."""
    rows = list(csv.DictReader((sample / "truth.csv").open()))
    by_seq: dict[str, list[str]] = {}
    for r in rows:
        by_seq.setdefault(r["sequence_id"], []).append(r["image_label"] or "")
    label = {
        seq: next((x for x in labels if x and x != "empty"), "empty")
        for seq, labels in by_seq.items()
    }
    return {r["filename"]: label[r["sequence_id"]] for r in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--sample", type=Path, default=Path("samples/cct-dev"))
    parser.add_argument("--out", type=Path, default=Path("demo-observations.csv"))
    parser.add_argument("--reviewer", default="demo-reviewer")
    args = parser.parse_args()
    api = httpx.Client(base_url=args.url, timeout=60)

    release = api.get("/version").json()["active_release"]
    print(f"WildInbox at {args.url}, release {release['id']}")
    if release["is_test"]:
        print("  (TEST predictor: suggestions are pseudo-random, not model output)")

    step(1, "Upload the sample batch")
    images = sorted((args.sample / "images").iterdir())
    res = api.post(
        "/batches",
        files=[("files", (p.name, p.read_bytes(), "image/jpeg")) for p in images],
        data={"metadata": (args.sample / "metadata.json").read_text()},
        headers={"Idempotency-Key": f"demo-{args.sample.name}"},
        timeout=300,
    )
    res.raise_for_status()
    batch = res.json()
    reused = " (already uploaded; reusing it)" if batch.get("duplicate_request") else ""
    print(f"  {len(images)} photos -> batch {batch['id']}{reused}")

    step(2, "Process")
    for _ in range(600):
        s = api.get(f"/batches/{batch['id']}").json()
        p = s["progress"]
        print(f"\r  scored {p['images_scored']}/{p['images_to_score']}", end="", flush=True)
        if p["finished"]:
            break
        time.sleep(1)
    print()
    if not s["progress"]["finished"]:
        sys.exit("  processing did not finish in 10 minutes")
    print(
        f"  {s['status']}: {s['counts']['events']} capture events from "
        f"{s['counts']['valid']} photos; {len(s['failures'])} unusable file(s)"
    )
    if s["counts"]["duplicate"]:
        print(
            f"  {s['counts']['duplicate']} photo(s) were already in this workspace and are not "
            "processed twice; run the demo on a fresh deployment (docker compose down -v)."
        )
        if not s["counts"]["events"]:
            sys.exit(1)

    step(3, "Uncertain events (lowest confidence first)")
    events = api.get("/events", params={"batch_id": batch["id"], "limit": 500}).json()["events"]
    todo = sorted(
        (e for e in events if e["decision"]["disposition"] == "needs_review"),
        key=lambda e: e["decision"]["confidence"] or 0,
    )
    dispositions = Counter(e["decision"]["disposition"] for e in events)
    print(f"  dispositions: {dict(dispositions)}")
    for e in todo[:5]:
        d = e["decision"]
        why = "; ".join(REASONS.get(r, r) for r in d["reasons"])
        conf = f"{d['confidence']:.0%}" if d["confidence"] is not None else "n/a"
        print(
            f"  {e['start_at']}  {e['camera_id']:8}  {len(e['image_ids'])} frames  "
            f"suggested {d['suggested_label']} ({conf}): {why}"
        )

    step(4, "Record a correction")
    truth = truth_by_file(args.sample)
    target: dict[str, Any] | None = None
    for e in todo:
        if e["latest_review"]:
            continue
        detail = api.get(f"/events/{e['id']}").json()
        actual = truth[detail["images"][0]["filename"]]
        if actual != e["decision"]["suggested_label"]:
            target = {**e, "truth": actual}
            break
    if target is None:
        print("  every wrong suggestion is already reviewed (demo was run before)")
    else:
        body = {
            "reviewer": args.reviewer,
            "outcome": "corrected",
            "confirmed_label": target["truth"],
            "note": "demo: corrected from the sample's ground truth",
        }
        r = api.post(f"/events/{target['id']}/reviews", json=body)
        r.raise_for_status()
        print(
            f"  event {target['id'][:8]}: suggested {target['decision']['suggested_label']}, "
            f"corrected to {target['truth']} by {args.reviewer}"
        )

    step(5, "Export observations")
    export = api.get(f"/batches/{batch['id']}/export", params={"format": "csv"})
    export.raise_for_status()
    args.out.write_bytes(export.content)
    rows = list(csv.DictReader(export.text.splitlines()))
    sources = Counter(r["label_source"] for r in rows)
    print(f"  {len(rows)} rows -> {args.out}  ({dict(sources)})")
    if target is not None:
        row = next(r for r in rows if r["event_id"] == target["id"])
        print(
            f"  corrected row: observation={row['observation']} source={row['label_source']} "
            f"reviewer={row['reviewer']} release={row['model_release_id']} "
            f"policy={row['policy_version']}"
        )
    print("\nOpen http://localhost:8501 to review the rest in the browser.")


if __name__ == "__main__":
    main()
