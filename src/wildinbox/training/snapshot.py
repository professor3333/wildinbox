"""`wildinbox snapshot build`: turn approved reviews into a versioned training
snapshot, following an update-cycle protocol (configs/experiments/update_cycle*.yaml).

Reads through the API (as an offline training machine would): reviewed events
on the protocol's cameras, their frames, and their originals.

- **Approved labels only**: an event counts when its latest review is by an
  approved reviewer. Automatic labels and unreviewed suggestions never do.
- **Protected evaluation records are excluded**, with every review on them
  (corrections included): frames of the protected dataset partitions (final
  test, calibration, seen-camera diagnostic by default), matched by SHA-256 or
  source file name, and events of earlier snapshots' holdouts. Exclusion
  happens before the time split, so protected events never shape a cutoff.
- Per camera, events before the median reviewed start time go to `train.jsonl`
  (frames of events whose reviewed label is a supported class); later events
  go to `holdout.jsonl` for the promotion gate and are never trained on.
- **Provenance**: `labels.jsonl` records, for every considered event, where its
  label came from (review id, reviewer, time, outcome, the suggestion and the
  release that made it, the full review chain) and where it went (train,
  holdout, not fit, or excluded and why).

The snapshot version is a hash of the train and holdout manifests, so the
same approved reviews always give the same version.

`snapshot.json` carries `schema` (`SNAPSHOT_SCHEMA`); `load_summary` is the one
reader, and training calls it before any work starts. Snapshots written before
the field existed (`reviewer`, no provenance) are read as `snapshot/v1`.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

from wildinbox.class_map import EMPTY_CLASS

SNAPSHOT_SCHEMA = "snapshot/v2"
LEGACY_SNAPSHOT_SCHEMA = "snapshot/v1"


class SnapshotError(ValueError):
    """A snapshot directory that training cannot use as it stands."""


def load_summary(snapshot_dir: Path) -> dict[str, Any]:
    """`snapshot.json`, validated and normalised to `SNAPSHOT_SCHEMA` fields.

    v1 summaries named one `reviewer`; they are read with `approved_reviewers`
    set to that name and no provenance record. For v2 the provenance file must
    match the SHA-256 the summary recorded."""
    path = snapshot_dir / "snapshot.json"
    if not path.is_file():
        raise SnapshotError(f"{path} does not exist")
    summary: dict[str, Any] = json.loads(path.read_text())
    schema = summary.get("schema", LEGACY_SNAPSHOT_SCHEMA if "reviewer" in summary else None)
    if schema == LEGACY_SNAPSHOT_SCHEMA:
        summary = {**summary, "schema": schema, "approved_reviewers": [summary.get("reviewer")]}
        summary.pop("reviewer", None)
        summary.setdefault("provenance", None)
    elif schema != SNAPSHOT_SCHEMA:
        raise SnapshotError(
            f"{path}: unknown snapshot schema {schema!r} (expected {SNAPSHOT_SCHEMA}); "
            "rebuild it with `wildinbox snapshot build`"
        )
    problems = []
    if not isinstance(summary.get("version"), str) or not summary["version"]:
        problems.append("version")
    reviewers = summary.get("approved_reviewers")
    if not (
        isinstance(reviewers, list)
        and reviewers
        and all(isinstance(r, str) and r for r in reviewers)
    ):
        problems.append("approved_reviewers")
    train = summary.get("train")
    if not (
        isinstance(train, dict) and all(isinstance(train.get(k), int) for k in ("images", "events"))
    ):
        problems.append("train.images/train.events")
    if not (snapshot_dir / "train.jsonl").is_file():
        problems.append("train.jsonl")
    if schema == SNAPSHOT_SCHEMA:
        prov = summary.get("provenance")
        labels = snapshot_dir / str((prov or {}).get("file", "labels.jsonl"))
        if not (isinstance(prov, dict) and isinstance(prov.get("sha256"), str)):
            problems.append("provenance")
        elif not labels.is_file():
            problems.append(labels.name)
        elif hashlib.sha256(labels.read_bytes()).hexdigest() != prov["sha256"]:
            problems.append(f"{labels.name} (SHA-256 differs from the summary)")
    if problems:
        raise SnapshotError(f"{path} ({schema}) is missing or has invalid: {', '.join(problems)}")
    return summary


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


DEFAULT_PROTECTED = {
    "splits": "data/splits/cct20-splits-v1/images.jsonl.gz",
    "partitions": ["final_test", "calibration", "seen_camera_diagnostic"],
    "snapshot_holdouts": [],
}


@dataclass
class Protected:
    """Evaluation records that must never enter training data."""

    sha256: set[str] = field(default_factory=set)
    source_ids: set[str] = field(default_factory=set)
    holdout_events: set[str] = field(default_factory=set)
    description: dict[str, Any] = field(default_factory=dict)

    def reason(self, event_id: str, frames: list[dict[str, Any]]) -> str | None:
        if event_id in self.holdout_events:
            return "earlier_snapshot_holdout"
        for f in frames:
            if f["sha256"] in self.sha256:
                return "protected_partition_sha256"
            if Path(f["filename"]).stem in self.source_ids:
                return "protected_partition_filename"
        return None


def protected_records(protocol: dict[str, Any], root: Path = Path(".")) -> Protected:
    spec = {**DEFAULT_PROTECTED, **(protocol.get("protected") or {})}
    out = Protected()
    splits = root / spec["splits"]
    parts = set(spec["partitions"])
    counts: dict[str, int] = {}
    if parts:
        with gzip.open(splits, "rt") as fh:
            for line in fh:
                row = json.loads(line)
                if row["partition"] in parts:
                    out.sha256.add(row["sha256"])
                    out.source_ids.add(row["source_id"])
                    counts[row["partition"]] = counts.get(row["partition"], 0) + 1
    for path in spec["snapshot_holdouts"]:
        for line in (root / path).read_text().splitlines():
            out.holdout_events.add(json.loads(line)["event_id"])
    out.description = {
        "splits": spec["splits"] if parts else None,
        "splits_sha256": hashlib.sha256(splits.read_bytes()).hexdigest() if parts else None,
        "partitions": dict(sorted(counts.items())),
        "snapshot_holdouts": list(spec["snapshot_holdouts"]),
        "holdout_events": len(out.holdout_events),
    }
    return out


def approved_reviewers(protocol: dict[str, Any]) -> list[str]:
    approval = protocol.get("approval") or {}
    return list(approval.get("reviewers") or [protocol["deployment"]["reviewer"]])


def review_label(event: dict[str, Any]) -> tuple[str | None, str]:
    """(label, kind) from the latest review: kind is supported, unsupported, or unresolved."""
    review = event["latest_review"]
    if review["outcome"] == "unresolved":
        return None, "unresolved"
    return review["confirmed_label"], "reviewed"


def _provenance(event: dict[str, Any], detail: dict[str, Any], use: str) -> dict[str, Any]:
    review = event["latest_review"]
    decision = detail.get("decision") or {}
    return {
        "event_id": event["id"],
        "batch_id": event["batch_id"],
        "camera_id": event["camera_id"],
        "start_at": event["start_at"],
        "use": use,
        "label": review["confirmed_label"],
        "review_id": review["id"],
        "reviewer": review["reviewer"],
        "reviewed_at": review["created_at"],
        "outcome": review["outcome"],
        "suggested_label": review["suggested_label"],
        "suggested_by_release": decision.get("model_release_id"),
        "policy_version": decision.get("policy_version"),
        "disposition": decision.get("disposition"),
        "audit_selected": decision.get("audit_selected"),
        "review_chain": [r["id"] for r in detail.get("reviews", [])],
        "frames": sorted(i["sha256"] for i in detail["images"]),
    }


def build(
    protocol_path: Path,
    api_url: str,
    out_root: Path,
    client: httpx.Client | None = None,
    root: Path = Path("."),
) -> Path:
    protocol = load_protocol(protocol_path)
    api = client or httpx.Client(base_url=api_url, timeout=120)
    deployed = protocol["deployed_release"]
    releases = {r["id"]: r for r in api.get("/releases").json()["releases"]}
    if deployed not in releases:
        raise RuntimeError(f"deployed release {deployed} is not registered at {api_url}")
    classes = set(releases[deployed]["class_names"])
    approved = approved_reviewers(protocol)
    protected = protected_records(protocol, root)
    train_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    not_approved = 0
    cutoffs: dict[str, str] = {}
    to_download: dict[str, str] = {}  # sha256 -> image id
    for cam in protocol["deployment"]["cameras"]:
        camera_id = f"cct-{cam}"
        events = []
        details: dict[str, dict[str, Any]] = {}
        for e in _events(api, camera_id):
            if e["latest_review"]["reviewer"] not in approved:
                not_approved += 1
                continue
            detail = api.get(f"/events/{e['id']}").json()
            why = protected.reason(e["id"], detail["images"])
            if why is not None:
                excluded.append(
                    {
                        "event_id": e["id"],
                        "camera_id": camera_id,
                        "reason": why,
                        "review_ids": [r["id"] for r in detail.get("reviews", [])],
                    }
                )
                provenance.append(_provenance(e, detail, f"excluded:{why}"))
                continue
            details[e["id"]] = detail
            events.append(e)
        before, after, cutoff = split_by_time(events)
        cutoffs[camera_id] = cutoff
        for group, rows in ((before, train_rows), (after, holdout_rows)):
            for e in group:
                detail = details[e["id"]]
                label, kind = review_label(e)
                frames = [
                    {"image_id": i["id"], "sha256": i["sha256"], "filename": i["filename"]}
                    for i in detail["images"]
                ]
                if rows is train_rows:
                    if label not in classes:
                        # unsupported species and unresolved events are not fit
                        provenance.append(_provenance(e, detail, "not_fit"))
                        continue
                    provenance.append(_provenance(e, detail, "train"))
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
                    provenance.append(_provenance(e, detail, "holdout"))
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
    labels_text = dumps(sorted(provenance, key=lambda r: (r["camera_id"], r["event_id"])))
    version = hashlib.sha256((train_text + holdout_text).encode()).hexdigest()[:12]
    out = out_root / f"{protocol.get('name', 'update1')}-{version}"
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
    (out / "labels.jsonl").write_text(labels_text)
    per_class: dict[str, int] = {}
    for r in train_rows:
        per_class[r["label"]] = per_class.get(r["label"], 0) + 1
    reasons: dict[str, int] = {}
    for x in excluded:
        reasons[x["reason"]] = reasons.get(x["reason"], 0) + 1
    summary = {
        "schema": SNAPSHOT_SCHEMA,
        "version": version,
        "protocol": str(protocol_path),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "deployed_release": deployed,
        "approved_reviewers": approved,
        "reviews_not_approved": not_approved,
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
        "protected": protected.description,
        "excluded": {"events": len(excluded), "by_reason": reasons, "records": excluded},
        "provenance": {
            "file": "labels.jsonl",
            "sha256": hashlib.sha256(labels_text.encode()).hexdigest(),
            "events": len(provenance),
        },
        "created_at": datetime.now(UTC).isoformat(),
    }
    (out / "snapshot.json").write_text(json.dumps(summary, indent=2) + "\n")
    assert all(r["label"] != "" for r in train_rows)
    assert EMPTY_CLASS in classes
    load_summary(out)  # what training will read back
    return out
