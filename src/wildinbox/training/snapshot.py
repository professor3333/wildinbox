"""`wildinbox snapshot build`: turn approved reviews into a versioned training
snapshot, following an update-cycle protocol (configs/experiments/update_cycle*.yaml).

Reads through the API (as an offline training machine would): reviewed events
on the protocol's cameras, their frames, and their originals.

- **Approved labels only**: an event counts when its latest review is by an
  approved reviewer. Automatic labels and unreviewed suggestions never do.
- **Protected evaluation records are excluded**, with every review on them
  (corrections included): frames of the protected dataset partitions (final
  test, calibration, seen-camera diagnostic by default), matched by SHA-256 or
  source file name, and earlier snapshots' holdouts, matched by event id or
  frame SHA-256. Exclusion happens before the time split, so protected events
  never shape a cutoff.
- Per camera, events before the median reviewed start time go to `train.jsonl`
  (frames of events whose reviewed label is a supported class); later events
  go to `holdout.jsonl` for the promotion gate and are never trained on.
- **Content separation** after the split, across cameras and by whole events:
  a holdout event holding a dataset training-partition frame is excluded, and
  so is a training event sharing any frame (by SHA-256) with a holdout event.
- **Only usable frames are ML inputs**: an event keeps every member (rejected,
  duplicate, or failed files included), but only frames that decoded and were
  scored without error enter `train.jsonl` or `holdout.jsonl`; downloaded
  originals must match their SHA-256 and decode. An event with no usable frame
  is excluded (`no_usable_frames`).
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
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
import yaml
from PIL import Image as PILImage

from wildinbox.api_client import api_client, request
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
    for manifest in ("train.jsonl", "holdout.jsonl"):
        if not (snapshot_dir / manifest).is_file():
            problems.append(manifest)
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
    mismatches = manifest_mismatches(snapshot_dir, summary)
    if mismatches:
        raise SnapshotError(
            f"{snapshot_dir}: the manifests do not match the recorded snapshot: "
            + "; ".join(mismatches)
        )
    return summary


def manifest_mismatches(snapshot_dir: Path, summary: dict[str, Any]) -> list[str]:
    """Where `train.jsonl` and `holdout.jsonl` disagree with what the snapshot
    recorded. The version is recomputed from the manifests' bytes, so a changed
    label, frame, or event anywhere changes it. The summary's counts are
    checked, and so is the provenance: each train/holdout event's reviewed label
    and frame set in `labels.jsonl` must match its manifest rows."""
    train_text = (snapshot_dir / "train.jsonl").read_text()
    hold_text = (snapshot_dir / "holdout.jsonl").read_text()
    out = []
    version = hashlib.sha256((train_text + hold_text).encode()).hexdigest()[:12]
    if version != summary["version"]:
        out.append(f"version recomputes to {version}, recorded {summary['version']}")
    train = [json.loads(x) for x in train_text.splitlines()]
    hold = [json.loads(x) for x in hold_text.splitlines()]
    counts = {
        "train.images": len(train),
        "train.events": len({r["event_id"] for r in train}),
        "train.per_class": dict(sorted(Counter(r["label"] for r in train).items())),
        "holdout.events": len(hold),
        "holdout.by_kind": {
            k: sum(1 for h in hold if h["kind"] == k)
            for k in ("supported", "unsupported", "unresolved")
        },
    }
    for key, value in counts.items():
        part, name = key.split(".")
        recorded = (summary.get(part) or {}).get(name)
        if recorded is not None and recorded != value:
            out.append(f"{key} is {value}, recorded {recorded}")
    prov = summary.get("provenance")
    if prov:
        rows = [json.loads(x) for x in (snapshot_dir / prov["file"]).read_text().splitlines()]
        if len(rows) != prov.get("events", len(rows)):
            out.append(f"provenance has {len(rows)} events, recorded {prov.get('events')}")
        manifest: dict[str, dict[str, tuple[Any, set[str]]]] = {"train": {}, "holdout": {}}
        for r in train:
            label, frames = manifest["train"].setdefault(r["event_id"], (r["label"], set()))
            frames.add(r["sha256"])
        for h in hold:
            manifest["holdout"][h["event_id"]] = (h["label"], {f["sha256"] for f in h["frames"]})
        for use, events in manifest.items():
            recorded_events = {r["event_id"]: r for r in rows if r["use"] == use}
            if set(recorded_events) != set(events):
                out.append(f"{use} events differ from the provenance")
                continue
            for event_id, (label, frames) in events.items():
                r = recorded_events[event_id]
                if r["label"] != label or set(r["frames"]) != frames:
                    out.append(f"{use} event {event_id} differs from its provenance")
    return out


def _get(api: httpx.Client, path: str, **params: Any) -> httpx.Response:
    return request(api, "GET", path, params=params or None)


def decode_problem(data: bytes) -> str | None:
    """Why these bytes cannot be an ML input, or None when they fully decode."""
    try:
        with PILImage.open(BytesIO(data)) as img:
            img.load()
    except Exception as e:
        return f"does not decode: {type(e).__name__}"
    return None


def snapshot_train_frames(snapshot_dir: Path) -> set[str]:
    """SHA-256s of the frames a snapshot's `train.jsonl` fits."""
    return {
        json.loads(line)["sha256"]
        for line in (snapshot_dir / "train.jsonl").read_text().splitlines()
    }


def model_training_frames(meta: dict[str, Any], model: str) -> set[str]:
    """Snapshot frames a model was fit on, beyond the dataset's training
    partition. Read from the lineage its `meta.json` records
    (`trained_on.snapshot.train_sha256`). For models trained before that was
    recorded, it comes from the snapshot directory, if that directory still
    holds the same version. Otherwise `SnapshotError`: an unknown lineage
    cannot be protected."""
    snap = (meta.get("trained_on") or {}).get("snapshot")
    if snap is None:
        return set()
    if "train_sha256" in snap:
        return set(snap["train_sha256"])
    d = Path(snap["path"])
    try:
        found = load_summary(d)["version"]
    except SnapshotError as e:
        raise SnapshotError(f"{model}: training lineage unknown ({e})") from e
    if found != snap["version"]:
        raise SnapshotError(
            f"{model}: trained on snapshot {snap['version']}, but {d} now holds {found}"
        )
    return snapshot_train_frames(d)


def deployed_training_frames(protocol: dict[str, Any], root: Path = Path(".")) -> set[str]:
    """Snapshot frames the protocol's deployed release was fit on (its exact
    artifacts, verified as the gate verifies them). The plumbing test release
    was fit on nothing."""
    from wildinbox.inference.plumbing import TEST_RELEASE_ID
    from wildinbox.training.gate import resolve_release

    release_id = protocol["deployed_release"]
    if release_id == TEST_RELEASE_ID:
        return set()
    deployed = resolve_release(
        release_id,
        root / protocol.get("models_root", "models"),
        root / protocol["deployed_policy"] if protocol.get("deployed_policy") else None,
    )
    meta = json.loads((deployed.model_dir / "meta.json").read_text())
    return model_training_frames(meta, release_id)


def load_protocol(path: Path) -> dict[str, Any]:
    raw: dict[str, Any] = yaml.safe_load(path.read_text())
    return raw


def _events(api: httpx.Client, camera: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    offset: int | None = 0
    while offset is not None:
        page = _get(
            api, "/events", camera_id=camera, reviewed=True, limit=500, offset=offset
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
    # Frame content of earlier holdouts: a re-upload gets a new event id but
    # keeps its bytes.
    holdout_sha256: set[str] = field(default_factory=set)
    # Frames the deployed model (and every candidate) already trained on; they
    # cannot measure either model, so they never enter a holdout.
    training_sha256: set[str] = field(default_factory=set)
    # Snapshot frames the deployed release was fit on (earlier update cycles):
    # it cannot be measured on them either.
    deployed_training_sha256: set[str] = field(default_factory=set)
    description: dict[str, Any] = field(default_factory=dict)

    def reason(self, event_id: str, frames: list[dict[str, Any]]) -> str | None:
        if event_id in self.holdout_events:
            return "earlier_snapshot_holdout"
        for f in frames:
            if f["sha256"] in self.sha256:
                return "protected_partition_sha256"
            if Path(f["filename"]).stem in self.source_ids:
                return "protected_partition_filename"
            if f["sha256"] in self.holdout_sha256:
                return "earlier_snapshot_holdout_sha256"
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
                elif row["partition"] == "train":
                    out.training_sha256.add(row["sha256"])
    for path in spec["snapshot_holdouts"]:
        for line in (root / path).read_text().splitlines():
            event = json.loads(line)
            out.holdout_events.add(event["event_id"])
            out.holdout_sha256.update(f["sha256"] for f in event.get("frames", []))
    out.description = {
        "splits": spec["splits"] if parts else None,
        "splits_sha256": hashlib.sha256(splits.read_bytes()).hexdigest() if parts else None,
        "partitions": dict(sorted(counts.items())),
        "snapshot_holdouts": list(spec["snapshot_holdouts"]),
        "holdout_events": len(out.holdout_events),
        "holdout_frames": len(out.holdout_sha256),
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


# Members that are ML inputs; `invalid` members carry no content (nothing
# decoded, so the reviewer saw nothing either) and are simply withheld. Any
# other status blocks the whole event: using the rest of its frames under the
# event's reviewed label could score (or fit) an empty frame as the animal the
# reviewer saw in the missing one.
ML_INPUT = ("usable", "duplicate_resolved")
NO_CONTENT = ("invalid",)


def frame_status(image: dict[str, Any]) -> tuple[str, str | None]:
    """(status, reason) of an event member:

    - `usable`: decoded and scored without error;
    - `duplicate_resolved`: the same bytes as an earlier upload whose
      prediction the API shows for it (its original decoded and was scored),
      so it is the same evidence the reviewer saw;
    - `invalid`: rejected at upload or on decoding (no content);
    - `duplicate_unresolved`, `processing_failed`, `unprocessed`: content
      that cannot be used as is, which excludes the event."""
    if image["validation_status"] == "invalid":
        return "invalid", image.get("validation_error")
    if image["validation_status"] == "duplicate":
        original = f"duplicate of image {image.get('duplicate_of')}"
        if image.get("sha256") and image.get("prediction") is not None:
            return "duplicate_resolved", original
        return "duplicate_unresolved", f"{original}, which has no prediction"
    if image.get("processing_error"):
        return "processing_failed", image["processing_error"]
    if image["validation_status"] != "valid" or not image.get("sha256"):
        return "unprocessed", f"validation status {image['validation_status']}"
    return "usable", None


def usable_frames(detail: dict[str, Any]) -> list[dict[str, Any]]:
    """The event's ML-input frames, once per distinct content, in upload order."""
    out: dict[str, dict[str, Any]] = {}
    for i in sorted(detail["images"], key=lambda i: i["position"]):
        if frame_status(i)[0] in ML_INPUT and i["sha256"] not in out:
            out[i["sha256"]] = {
                "image_id": i["id"],
                "sha256": i["sha256"],
                "filename": i["filename"],
            }
    return list(out.values())


def blocking_member(detail: dict[str, Any]) -> str | None:
    """The status of the first member that keeps this event out, if any."""
    for i in sorted(detail["images"], key=lambda i: i["position"]):
        status, _ = frame_status(i)
        if status not in ML_INPUT and status not in NO_CONTENT:
            return status
    return None


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
        # ML inputs only; `members` keeps every frame and why it was withheld.
        "frames": sorted(f["sha256"] for f in usable_frames(detail)),
        "members": [
            {
                "image_id": i["id"],
                "filename": i["filename"],
                "sha256": i["sha256"],
                "status": status,
                "reason": reason,
            }
            for i in sorted(detail["images"], key=lambda i: i["position"])
            for status, reason in [frame_status(i)]
        ],
    }


def build(
    protocol_path: Path,
    api_url: str,
    out_root: Path,
    client: httpx.Client | None = None,
    root: Path = Path("."),
) -> Path:
    protocol = load_protocol(protocol_path)
    # Reviewer approval trusts the reviewer names the API returns, so snapshots
    # come from an authenticated API.
    api = client or api_client(api_url)
    deployed = protocol["deployed_release"]
    releases = {r["id"]: r for r in _get(api, "/releases").json()["releases"]}
    if deployed not in releases:
        raise RuntimeError(f"deployed release {deployed} is not registered at {api_url}")
    classes = set(releases[deployed]["class_names"])
    approved = approved_reviewers(protocol)
    protected = protected_records(protocol, root)
    protected.deployed_training_sha256 = deployed_training_frames(protocol, root)
    train_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    not_approved = 0
    cutoffs: dict[str, str] = {}
    to_download: dict[str, str] = {}  # sha256 -> image id
    withheld: Counter[str] = Counter()  # members of train/holdout events kept out of ML inputs
    details: dict[str, dict[str, Any]] = {}
    undecodable: set[str] = set()
    before: list[dict[str, Any]] = []
    after: list[dict[str, Any]] = []

    def exclude(e: dict[str, Any], why: str) -> None:
        detail = details[e["id"]]
        excluded.append(
            {
                "event_id": e["id"],
                "camera_id": e["camera_id"],
                "reason": why,
                "review_ids": [r["id"] for r in detail.get("reviews", [])],
            }
        )
        provenance.append(_provenance(e, detail, f"excluded:{why}"))

    def shas(e: dict[str, Any]) -> set[str]:
        """Content of every member whose bytes decode, usable or not: a
        duplicate or a frame that failed inference still carries the image.
        Bytes of a rejected file (corrupt, or a duplicate of one) carry none,
        and would otherwise tie unrelated events together."""
        return {i["sha256"] for i in details[e["id"]]["images"] if i["sha256"]} - undecodable

    for cam in protocol["deployment"]["cameras"]:
        camera_id = f"cct-{cam}"
        events = []
        for e in _events(api, camera_id):
            if e["latest_review"]["reviewer"] not in approved:
                not_approved += 1
                continue
            details[e["id"]] = _get(api, f"/events/{e['id']}").json()
            why = protected.reason(e["id"], details[e["id"]]["images"])
            if why is not None:
                exclude(e, why)
                continue
            events.append(e)
        cam_before, cam_after, cutoffs[camera_id] = split_by_time(events)
        undecodable.update(
            i["sha256"]
            for e in events
            for i in details[e["id"]]["images"]
            if i["validation_status"] == "invalid" and i["sha256"]
        )
        before += cam_before
        after += cam_after
    # Content separation, whole events at a time, across cameras (the same
    # bytes can be uploaded again under another camera, time, or file name):
    # a holdout event may not contain frames any model trained on, and a
    # training event may not share a frame with any holdout event.
    holdout_shas = set().union(*map(shas, after))
    kept_after = []
    for e in after:
        if shas(e) & protected.training_sha256:
            exclude(e, "holdout_frame_in_training_partition")
        elif shas(e) & protected.deployed_training_sha256:
            exclude(e, "holdout_frame_in_deployed_training")
        else:
            kept_after.append(e)
    kept_before = []
    for e in before:
        if shas(e) & holdout_shas:
            exclude(e, "train_frame_in_holdout")
        else:
            kept_before.append(e)
    for group, rows in ((kept_before, train_rows), (kept_after, holdout_rows)):
        for e in group:
            detail = details[e["id"]]
            label, kind = review_label(e)
            frames = usable_frames(detail)
            if (blocked := blocking_member(detail)) is not None:
                exclude(e, f"member_{blocked}")
                continue
            if not frames:
                exclude(e, "no_usable_frames")
                continue
            withheld.update(s for s, _ in map(frame_status, detail["images"]) if s not in ML_INPUT)
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
                            "camera_id": e["camera_id"],
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
                        "camera_id": e["camera_id"],
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
        data = _get(api, f"/images/{image_id}/original").content
        if hashlib.sha256(data).hexdigest() != sha:
            raise RuntimeError(f"original for image {image_id} does not match its SHA-256")
        if (problem := decode_problem(data)) is not None:
            raise RuntimeError(f"image {image_id} was marked usable but {problem}")
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
        # Members of train/holdout events that are not ML inputs, by status;
        # each is listed with its reason under `members` in labels.jsonl.
        "withheld_frames": dict(sorted(withheld.items())),
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
