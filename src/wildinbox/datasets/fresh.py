"""Capture events for fresh cameras: locations outside CCT20 used by the v2
experiment (configs/experiments/fresh_cameras_v2.yaml).

Same events, ground truth, and roles as the CCT20 build (sequence_id/v1), but
the supported classes are the released model's, read from the CCT20 split
lock rather than selected here, and every camera goes to one partition. The
build also checks the fresh images against CCT20: no shared camera, no shared
file, no near-duplicate.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from wildinbox.config import ConfigError
from wildinbox.datasets.build import (
    BuildError,
    Check,
    _event_row,
    _image_rows,
    _write_jsonl_gz,
    group_events,
    load_frames,
)
from wildinbox.datasets.events import Event, apply_ground_truth, assign_role
from wildinbox.datasets.spec import Kind, Partition, Taxonomy
from wildinbox.ingestion.inventory import NEAR_DUPLICATE_MIN_SIMILARITY, manifest_version
from wildinbox.ingestion.validate import unpack_signatures


class FreshSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    source: str  # inventory name: data/inventory/<source>.sqlite
    inventory_manifest_version: str
    taxonomy: Path
    grouping_rule: Literal["sequence_id/v1"]
    partition: Literal[Partition.ADAPTATION_DEVELOPMENT, Partition.FRESH_TEST]
    selection: Path  # manifests/fresh-cameras-v2.json
    supported_from: Path  # the CCT20 split lock that fixed the supported classes
    reference_source: str  # CCT20's inventory, for the leakage checks


def load_fresh_spec(path: str | Path) -> FreshSpec:
    path = Path(path)
    try:
        return FreshSpec.model_validate(yaml.safe_load(path.read_text()))
    except FileNotFoundError:
        raise ConfigError(f"{path}: file not found") from None
    except ValidationError as e:
        raise ConfigError(f"{path}: invalid fresh-camera spec\n{e}") from None


@dataclass
class FreshResult:
    version: str
    events: list[Event]
    supported_classes: list[str]
    checks: list[Check]
    events_path: Path

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


def _signatures(conn: sqlite3.Connection) -> tuple[list[str], np.ndarray]:
    rows = conn.execute(
        "SELECT source_id, signature FROM records WHERE status = 'accepted' "
        "AND signature IS NOT NULL AND NOT low_information ORDER BY source_id"
    ).fetchall()
    return [r[0] for r in rows], unpack_signatures([r[1] for r in rows])


def leakage_checks(
    spec: FreshSpec, events: list[Event], fresh: sqlite3.Connection, ref: sqlite3.Connection
) -> list[Check]:
    selection = json.loads(spec.selection.read_text())
    group = {c["location"] for c in selection[spec.partition.value]}
    cams = {e.camera_id for e in events}
    ref_cams = {r[0] for r in ref.execute("SELECT DISTINCT camera_id FROM records")}
    shared_cams = sorted(cams & ref_cams)
    fresh_sha = {r[0] for r in fresh.execute("SELECT sha256 FROM records WHERE sha256 IS NOT NULL")}
    ref_sha = {r[0] for r in ref.execute("SELECT sha256 FROM records WHERE sha256 IS NOT NULL")}
    ids, sigs = _signatures(fresh)
    _, ref_sigs = _signatures(ref)
    near = 0
    for start in range(0, len(ids), 1024):
        sim = sigs[start : start + 1024] @ ref_sigs.T
        near += int((sim >= NEAR_DUPLICATE_MIN_SIMILARITY).any(axis=1).sum())
    return [
        Check(
            "cameras_are_the_selected_group",
            cams == group,
            f"{len(cams)} cameras; selection lists {len(group)}",
        ),
        Check("no_camera_shared_with_cct20", not shared_cams, f"shared: {shared_cams or 'none'}"),
        Check(
            "no_file_shared_with_cct20",
            not (fresh_sha & ref_sha),
            f"{len(fresh_sha & ref_sha)} identical files",
        ),
        Check(
            "no_near_duplicate_of_cct20",
            near == 0,
            f"{near} of {len(ids)} images at similarity >= {NEAR_DUPLICATE_MIN_SIMILARITY}",
        ),
    ]


def build_fresh(spec: FreshSpec, taxonomy: Taxonomy, data_dir: Path) -> FreshResult:
    fresh = sqlite3.connect(data_dir / "inventory" / f"{spec.source}.sqlite")
    fresh.row_factory = sqlite3.Row
    ref = sqlite3.connect(data_dir / "inventory" / f"{spec.reference_source}.sqlite")
    actual = manifest_version(fresh, spec.source)
    if actual != spec.inventory_manifest_version:
        raise BuildError(
            f"inventory is {actual}, spec pins {spec.inventory_manifest_version}; "
            "re-run `wildinbox data ingest`"
        )
    cats = [r[0] for r in fresh.execute("SELECT DISTINCT normalized FROM categories")]
    missing = sorted(set(cats) - set(taxonomy.categories))
    if missing:
        raise BuildError(f"categories missing from taxonomy {taxonomy.name!r}: {missing}")
    supported_classes: list[str] = json.loads(spec.supported_from.read_text())["supported_classes"]
    supported = frozenset(c for c in supported_classes if taxonomy.kind_of(c) is Kind.ANIMAL)

    events = group_events(load_frames(fresh), spec.source)
    for e in events:
        e.partition = spec.partition
        apply_ground_truth(e, taxonomy)
        assign_role(e, supported)
    checks = leakage_checks(spec, events, fresh, ref)
    fresh.close()
    ref.close()

    out_dir = data_dir / "splits" / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    events_path = out_dir / "events.jsonl.gz"
    _write_jsonl_gz(events_path, (_event_row(e, spec.grouping_rule) for e in events), digest)
    _write_jsonl_gz(
        out_dir / "images.jsonl.gz", (r for e in events for r in _image_rows(e)), digest
    )
    return FreshResult(
        version=f"{spec.name}-{digest.hexdigest()[:12]}",
        events=events,
        supported_classes=supported_classes,
        checks=checks,
        events_path=events_path,
    )


def lock_payload(spec: FreshSpec, result: FreshResult) -> dict[str, Any]:
    """Pinned counts. Per-role counts per camera only for development cameras;
    the fresh test's lock records totals, so its composition stays unread."""
    per_camera: dict[str, Counter[str]] = defaultdict(Counter)
    for e in result.events:
        if e.excluded_reason:
            per_camera[e.camera_id][f"excluded:{e.excluded_reason}"] += 1
        elif e.role:
            per_camera[e.camera_id][e.role.value] += 1
        per_camera[e.camera_id]["events"] += 1
        per_camera[e.camera_id]["images"] += len(e.images)
    payload: dict[str, Any] = {
        "split_version": result.version,
        "inventory_manifest_version": spec.inventory_manifest_version,
        "partition": spec.partition.value,
        "grouping_rule": spec.grouping_rule,
        "supported_classes": result.supported_classes,
        "checks": {c.name: c.passed for c in result.checks},
    }
    ordered = sorted(per_camera.items(), key=lambda kv: (len(kv[0]), kv[0]))  # "9" before "10"
    if spec.partition is Partition.FRESH_TEST:
        payload["cameras"] = {
            cam: {"events": c["events"], "images": c["images"]} for cam, c in ordered
        }
    else:
        payload["cameras"] = {cam: dict(sorted(c.items())) for cam, c in ordered}
    return payload
