"""Simulate a deployment for the update cycle: upload a batch, then review every
event from the dataset's ground truth through the API.

    uv run python scripts/simulate_deployment.py --batch-dir data/samples/deploy-cams-90-125

Uploads one batch per camera (one memory card each).

Reviews are SIMULATED (reviewer "simulated-ground-truth", with a note saying
so): no person looked at these photos. Event labels follow
configs/experiments/update_cycle.yaml: supported species -> that species,
empty -> empty, unsupported species -> its name, mixed species -> unresolved.
The outcome (confirmed / corrected / unresolved) follows from the suggestion,
as in the review interface.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import httpx

from wildinbox.ui.logic import review_for

NOTE = "simulated review from Caltech Camera Traps ground truth (update-cycle protocol)"


def event_label(row: dict[str, str]) -> str | None:
    role, label = row["event_role"], row["event_label"] or None
    if role == "empty":
        return "empty"
    if role in ("supported_species", "unsupported_animal"):
        return label
    return None  # mixed species (or anything else): unresolved


def review_batch(
    api: httpx.Client,
    batch_id: str,
    truth: dict[str, dict[str, str]],
    reviewer: str,
    outcomes: Counter[str],
) -> None:
    offset: int | None = 0
    while offset is not None:
        page = api.get(
            "/events", params={"batch_id": batch_id, "limit": 200, "offset": offset}
        ).json()
        for e in page["events"]:
            if e["latest_review"]:
                outcomes["already reviewed"] += 1
                continue
            detail = api.get(f"/events/{e['id']}").json()
            label = event_label(truth[detail["images"][0]["filename"]])
            outcome, confirmed = review_for(e["decision"]["suggested_label"], label)
            r = api.post(
                f"/events/{e['id']}/reviews",
                json={
                    "reviewer": reviewer,
                    "outcome": outcome,
                    "confirmed_label": confirmed,
                    "note": NOTE,
                },
            )
            r.raise_for_status()
            outcomes[outcome] += 1
        offset = page["next_offset"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--reviewer", default="simulated-ground-truth")
    parser.add_argument(
        "--upload-only", action="store_true", help="upload and process; post no reviews"
    )
    args = parser.parse_args()
    api = httpx.Client(base_url=args.url, timeout=120)
    truth = {r["filename"]: r for r in csv.DictReader((args.batch_dir / "truth.csv").open())}

    meta = json.loads((args.batch_dir / "metadata.json").read_text())["files"]
    by_camera: dict[str, list[Path]] = {}
    for p in sorted((args.batch_dir / "images").iterdir()):
        by_camera.setdefault(meta[p.name]["camera_id"], []).append(p)
    outcomes: Counter[str] = Counter()
    for camera, images in sorted(by_camera.items()):  # one memory card per camera
        res = api.post(
            "/batches",
            files=[("files", (p.name, p.read_bytes(), "image/jpeg")) for p in images],
            data={"metadata": json.dumps({"files": {p.name: meta[p.name] for p in images}})},
            headers={"Idempotency-Key": f"deployment-{args.batch_dir.name}-{camera}"},
            timeout=900,
        )
        res.raise_for_status()
        batch = res.json()
        print(f"{camera}: batch {batch['id']}, {len(images)} photos uploaded")
        for _ in range(3600):
            s = api.get(f"/batches/{batch['id']}").json()
            if s["progress"]["finished"]:
                break
            time.sleep(2)
        else:
            sys.exit("processing did not finish")
        c = s["counts"]
        print(f"  {s['status']}: {c['events']} events, {c['duplicate']} duplicates")
        if not args.upload_only:
            review_batch(api, batch["id"], truth, args.reviewer, outcomes)
    if not args.upload_only:
        print(f"simulated reviews: {dict(outcomes)}")


if __name__ == "__main__":
    main()
