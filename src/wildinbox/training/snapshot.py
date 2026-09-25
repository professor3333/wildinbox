"""`wildinbox snapshot build`: turn reviewed deployment events into a versioned
training snapshot, following configs/experiments/update_cycle.yaml.

Reads through the API (as an offline training machine would): reviewed events
on the protocol's cameras, their frames, and their originals. Per camera, events
before the median reviewed start time go to `train.jsonl` (frames of events
whose reviewed label is a supported class); later events go to
`holdout.jsonl` for the promotion gate and are never trained on.

The snapshot version is a hash of both manifests, so the same reviews always
give the same version.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

from wildinbox.class_map import EMPTY_CLASS


def load_protocol(path: Path) -> dict[str, Any]:
    raw: dict[str, Any] = yaml.safe_load(path.read_text())
    return raw


def _events(api: httpx.Client, camera: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    offset: int | None = 0
    while offset is not None:
        page = api.get(
            "/events",
            params={"camera_id": camera, "reviewed": True, "limit": 500, "offset": offset},
        ).json()
        out.extend(page["events"])
        offset = page["next_offset"]
    return out


def split_by_time(
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """(before, after, cutoff): events starting before the median start time,
    and the rest. Events without a start time cannot be placed and are dropped."""
    timed = sorted((e for e in events if e.get("start_at")), key=lambda e: e["start_at"])
    if not timed:
        return [], [], ""
    cutoff = statistics.median_low([e["start_at"] for e in timed])
    return (
        [e for e in timed if e["start_at"] < cutoff],
        [e for e in timed if e["start_at"] >= cutoff],
        cutoff,
    )


def review_label(event: dict[str, Any]) -> tuple[str | None, str]:
    """(label, kind) from the latest review: kind is supported, unsupported, or unresolved."""
    review = event["latest_review"]
    if review["outcome"] == "unresolved":
        return None, "unresolved"
    return review["confirmed_label"], "reviewed"


def build(protocol_path: Path, api_url: str, out_root: Path) -> Path:
    protocol = load_protocol(protocol_path)
    api = httpx.Client(base_url=api_url, timeout=120)
    deployed = protocol["deployed_release"]
    releases = {r["id"]: r for r in api.get("/releases").json()["releases"]}
    if deployed not in releases:
        raise RuntimeError(f"deployed release {deployed} is not registered at {api_url}")
    classes = set(releases[deployed]["class_names"])
    reviewer = protocol["deployment"]["reviewer"]

    train_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    cutoffs: dict[str, str] = {}
    to_download: dict[str, str] = {}  # sha256 -> image id
    for cam in protocol["deployment"]["cameras"]:
        camera_id = f"cct-{cam}"
        events = [e for e in _events(api, camera_id) if e["latest_review"]["reviewer"] == reviewer]
        before, after, cutoff = split_by_time(events)
        cutoffs[camera_id] = cutoff
        for group, rows in ((before, train_rows), (after, holdout_rows)):
            for e in group:
                detail = api.get(f"/events/{e['id']}").json()
                label, kind = review_label(e)
                frames = [
                    {"image_id": i["id"], "sha256": i["sha256"], "filename": i["filename"]}
                    for i in detail["images"]
                ]
                if rows is train_rows:
                    if label not in classes:
                        continue  # unsupported species and unresolved events are not fit
                    for f in frames:
                        rows.append(
                            {
                                **f,
                                "label": label,
                                "event_id": e["id"],
                                "camera_id": camera_id,
                                "start_at": e["start_at"],
                                "review_id": e["latest_review"]["id"],
                            }
                        )
                        to_download[f["sha256"]] = f["image_id"]
                else:
                    rows.append(
                        {
                            "event_id": e["id"],
                            "camera_id": camera_id,
                            "start_at": e["start_at"],
                            "label": label,
                            "kind": (
                                kind
                                if kind == "unresolved"
                                else ("supported" if label in classes else "unsupported")
                            ),
                            "frames": frames,
                            "review_id": e["latest_review"]["id"],
                        }
                    )
                    for f in frames:
                        to_download[f["sha256"]] = f["image_id"]

    def dumps(rows: list[dict[str, Any]]) -> str:
        return "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)

    train_text, holdout_text = dumps(train_rows), dumps(holdout_rows)
    version = hashlib.sha256((train_text + holdout_text).encode()).hexdigest()[:12]
    out = out_root / f"update1-{version}"
    (out / "images").mkdir(parents=True, exist_ok=True)
    for sha, image_id in sorted(to_download.items()):
        dest = out / "images" / f"{sha}.jpg"
        if dest.exists():
            continue
        data = api.get(f"/images/{image_id}/original").content
        if hashlib.sha256(data).hexdigest() != sha:
            raise RuntimeError(f"original for image {image_id} does not match its SHA-256")
        dest.write_bytes(data)
    (out / "train.jsonl").write_text(train_text)
    (out / "holdout.jsonl").write_text(holdout_text)
    per_class: dict[str, int] = {}
    for r in train_rows:
        per_class[r["label"]] = per_class.get(r["label"], 0) + 1
    summary = {
        "version": version,
        "protocol": str(protocol_path),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "deployed_release": deployed,
        "reviewer": reviewer,
        "cutoffs": cutoffs,
        "train": {
            "images": len(train_rows),
            "events": len({r["event_id"] for r in train_rows}),
            "per_class": dict(sorted(per_class.items())),
        },
        "holdout": {
            "events": len(holdout_rows),
            "by_kind": {
                k: sum(1 for r in holdout_rows if r["kind"] == k)
                for k in ("supported", "unsupported", "unresolved")
            },
        },
        "created_at": datetime.now(UTC).isoformat(),
    }
    (out / "snapshot.json").write_text(json.dumps(summary, indent=2) + "\n")
    assert all(r["label"] != "" for r in train_rows)
    assert EMPTY_CLASS in classes
    return out
