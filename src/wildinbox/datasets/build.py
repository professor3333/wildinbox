"""Build capture events, supported species, partitions, and leakage checks."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wildinbox.class_map import EMPTY_CLASS
from wildinbox.datasets.events import (
    Event,
    Frame,
    Role,
    apply_ground_truth,
    assign_role,
    fit_exclusion,
    frame_labels,
)
from wildinbox.datasets.spec import Kind, Partition, SplitSpec, Taxonomy
from wildinbox.ingestion.inventory import Status, manifest_version

DEVELOPMENT = (Partition.CALIBRATION, Partition.POLICY_VALIDATION)


class BuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class SpeciesStats:
    species: str
    train_events: int
    train_cameras: int
    selected: bool
    reason: str


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class BuildResult:
    version: str
    events: list[Event]
    species: list[SpeciesStats]
    supported_classes: list[str]
    checks: list[Check]
    events_path: Path
    images_path: Path

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


# ------------------------------------------------------------------ loading


def load_frames(conn: sqlite3.Connection) -> list[Frame]:
    rows = conn.execute(
        "SELECT source_id, status, camera_id, sequence_id, source_file, frame_num, "
        "captured_at, storage_path, sha256, original_labels FROM records ORDER BY source_id"
    )
    frames = []
    for r in rows:
        if r[2] is None or r[3] is None:
            raise BuildError(f"record {r[0]} has no camera or sequence id; fix the inventory")
        frames.append(
            Frame(
                source_id=r[0],
                status=Status(r[1]),
                camera_id=r[2],
                sequence_id=r[3],
                source_file=r[4],
                frame_num=r[5],
                captured_at=r[6],
                storage_path=r[7],
                sha256=r[8],
                labels=frame_labels(json.loads(r[9])),
            )
        )
    return frames


def group_events(frames: Iterable[Frame], dataset: str) -> list[Event]:
    """sequence_id/v1: one event per sequence id."""
    by_seq: dict[str, list[Frame]] = defaultdict(list)
    for f in frames:
        by_seq[f.sequence_id].append(f)
    events = []
    for seq, fs in sorted(by_seq.items()):
        fs.sort(key=lambda f: (f.frame_num if f.frame_num is not None else -1, f.source_id))
        cams = {f.camera_id for f in fs}
        if len(cams) != 1:
            raise BuildError(f"sequence {seq} spans cameras {sorted(cams)}")
        events.append(
            Event(
                event_id=f"{dataset}:{seq}", camera_id=fs[0].camera_id, sequence_id=seq, frames=fs
            )
        )
    return events


# --------------------------------------------------------------- partitions


def assign_partitions(events: list[Event], spec: SplitSpec) -> None:
    by_camera = spec.cameras.partition_of()
    unassigned = sorted({e.camera_id for e in events} - set(by_camera))
    if unassigned:
        raise BuildError(f"cameras not assigned to any partition in the split spec: {unassigned}")
    diag = spec.seen_camera_diagnostic
    for e in events:
        part = by_camera[e.camera_id]
        if part is Partition.TRAIN and diag and diag.contains(e.sequence_id):
            part = Partition.SEEN_CAMERA_DIAGNOSTIC
        e.partition = part


def select_species(events: list[Event], spec: SplitSpec) -> list[SpeciesStats]:
    """Supported species from TRAINING-partition, single-species events only."""
    rule = spec.species_selection
    per_camera: dict[str, Counter[str]] = defaultdict(Counter)
    for e in events:
        if e.partition is Partition.TRAIN and not e.excluded_reason and len(e.animals) == 1:
            per_camera[e.animals[0]][e.camera_id] += 1
    stats = []
    for species, cams in per_camera.items():
        n = sum(cams.values())
        n_cams = sum(1 for c in cams.values() if c >= rule.min_events_per_camera)
        reasons = []
        if n < rule.min_train_events:
            reasons.append(f"{n} < {rule.min_train_events} training events")
        if n_cams < rule.min_train_cameras:
            reasons.append(f"{n_cams} < {rule.min_train_cameras} cameras")
        stats.append(
            SpeciesStats(species, n, n_cams, not reasons, "; ".join(reasons) or "meets rule")
        )
    return sorted(stats, key=lambda s: (-s.train_events, s.species))


# ------------------------------------------------------------------- checks


def leakage_checks(
    events: list[Event], frames: list[Frame], near_dups: list[tuple[str, str]], spec: SplitSpec
) -> list[Check]:
    kept = [e for e in events if not e.excluded_reason]
    img_part = {f.source_id: e.partition for e in kept for f in e.images}
    checks: list[Check] = []

    def check(name: str, bad: list[Any], ok: str) -> None:
        checks.append(
            Check(name, not bad, ok if not bad else f"{len(bad)} violation(s): {bad[:5]}")
        )

    seq_parts: dict[str, set[Partition | None]] = defaultdict(set)
    sha_parts: dict[str, set[Partition | None]] = defaultdict(set)
    for e in kept:
        for f in e.images:
            seq_parts[f.sequence_id].add(e.partition)
            if f.sha256:
                sha_parts[f.sha256].add(e.partition)
    check(
        "no sequence id in more than one partition",
        sorted(s for s, p in seq_parts.items() if len(p) > 1),
        f"{len(seq_parts)} sequences, each in exactly one partition",
    )
    check(
        "no exact image duplicate (SHA-256) across partitions",
        sorted(s[:12] for s, p in sha_parts.items() if len(p) > 1),
        f"{len(sha_parts)} distinct image hashes checked",
    )
    check(
        "no suspected near-duplicate pair across partitions",
        [
            (a, b)
            for a, b in near_dups
            if a in img_part and b in img_part and img_part[a] != img_part[b]
        ],
        f"{len(near_dups)} inventory near-duplicate pairs checked",
    )

    cams: dict[Partition | None, set[str]] = defaultdict(set)
    for e in kept:
        cams[e.partition].add(e.camera_id)
    test = cams[Partition.FINAL_TEST]
    others = set().union(*(c for p, c in cams.items() if p is not Partition.FINAL_TEST))
    check(
        "final-test cameras used in no other partition",
        sorted(test & others),
        f"{len(test)} final-test cameras, none elsewhere",
    )
    fit_cams = cams[Partition.TRAIN] | cams[Partition.SEEN_CAMERA_DIAGNOSTIC]
    dev = cams[Partition.CALIBRATION] | cams[Partition.POLICY_VALIDATION]
    check(
        "development cameras separate from training cameras",
        sorted(dev & fit_cams),
        f"{len(dev)} development cameras, none used for training",
    )
    check(
        "calibration and policy-validation cameras separate",
        sorted(cams[Partition.CALIBRATION] & cams[Partition.POLICY_VALIDATION]),
        "camera-disjoint, so their events are disjoint too",
    )
    check(
        "seen-camera diagnostic only uses training cameras",
        sorted(cams[Partition.SEEN_CAMERA_DIAGNOSTIC] - cams[Partition.TRAIN]),
        "diagnostic cameras are a subset of training cameras",
    )

    in_events = Counter(f.source_id for e in events for f in e.frames)
    check(
        "every inventory record belongs to exactly one event",
        sorted(sid for sid in {f.source_id for f in frames} if in_events[sid] != 1),
        f"{len(frames)} records, {len(events)} events",
    )
    check(
        "every event is in exactly one partition or excluded with a reason",
        [e.event_id for e in events if not e.excluded_reason and e.partition is None],
        f"{len(kept)} events kept, {len(events) - len(kept)} excluded with reasons",
    )
    bad_fit = []
    for e in kept:
        for f in e.images:
            if fit_exclusion(e, f) is not None:
                continue
            if (
                e.partition is not Partition.TRAIN
                or e.role not in (Role.SUPPORTED, Role.EMPTY)
                or f.labels != (e.label,)
                or (e.role is Role.EMPTY and e.animal_present)
            ):
                bad_fit.append(f"{e.event_id}:{f.source_id}")
    check(
        "fit examples are training-only, supported or all-empty, frame label = event label",
        bad_fit,
        "unsupported, mixed, and non-animal events never become fit examples",
    )
    return checks


# ------------------------------------------------------------------ outputs


def _event_row(e: Event, spec: SplitSpec) -> dict[str, Any]:
    return {
        "event_id": e.event_id,
        "partition": e.partition.value if e.partition and not e.excluded_reason else None,
        "excluded_reason": e.excluded_reason,
        "camera_id": e.camera_id,
        "sequence_id": e.sequence_id,
        "grouping_rule": spec.grouping_rule,
        "image_ids": [f.source_id for f in e.images],
        "all_frame_ids": [f.source_id for f in e.frames],
        "frame_labels": {f.source_id: list(f.labels) for f in e.frames},
        "animal_present": e.animal_present if not e.excluded_reason else None,
        "animals": list(e.animals),
        "label": e.label,
        "role": e.role.value if e.role else None,
        "start": min((f.captured_at for f in e.frames if f.captured_at), default=None),
        "end": max((f.captured_at for f in e.frames if f.captured_at), default=None),
        "source_files": sorted({f.source_file for f in e.frames}),
        "notes": e.notes,
    }


def _image_rows(e: Event) -> Iterable[dict[str, Any]]:
    if e.excluded_reason:
        return
    for f in e.images:
        excl = fit_exclusion(e, f)
        yield {
            "source_id": f.source_id,
            "event_id": e.event_id,
            "partition": e.partition.value if e.partition else None,
            "camera_id": f.camera_id,
            "storage_path": f.storage_path,
            "sha256": f.sha256,
            "image_label": f.labels[0] if len(f.labels) == 1 else None,
            "event_label": e.label,
            "event_role": e.role.value if e.role else None,
            "use_for_fit": excl is None,
            "fit_exclusion": excl.value if excl else None,
        }


def _write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]], digest: Any) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as f:
        for row in rows:
            line = json.dumps(row, sort_keys=True).encode() + b"\n"
            digest.update(line)
            f.write(line)
    os.replace(tmp, path)


def build(spec: SplitSpec, taxonomy: Taxonomy, inventory_db: Path, out_root: Path) -> BuildResult:
    conn = sqlite3.connect(inventory_db)
    conn.row_factory = sqlite3.Row
    dataset = spec.inventory_manifest_version.split("-")[0]
    actual = manifest_version(conn, dataset)
    if actual != spec.inventory_manifest_version:
        raise BuildError(
            f"inventory is {actual}, split spec pins "
            f"{spec.inventory_manifest_version}; re-run `wildinbox data ingest`"
        )
    cats = [r[0] for r in conn.execute("SELECT DISTINCT normalized FROM categories")]
    missing = sorted(set(cats) - set(taxonomy.categories))
    if missing:
        raise BuildError(f"categories missing from taxonomy {taxonomy.name!r}: {missing}")
    near = [(a, b) for a, b in conn.execute("SELECT a, b FROM near_duplicates")]
    frames = load_frames(conn)
    conn.close()

    events = group_events(frames, dataset)
    assign_partitions(events, spec)
    for e in events:
        apply_ground_truth(e, taxonomy)
    species = select_species(events, spec)
    supported = frozenset(s.species for s in species if s.selected)
    if not supported:
        raise BuildError("species-selection rule selected no species")
    for e in events:
        assign_role(e, supported)
    assert all(taxonomy.kind_of(s) is Kind.ANIMAL for s in supported)

    checks = leakage_checks(events, frames, near, spec)
    out_dir = out_root / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    events_path, images_path = out_dir / "events.jsonl.gz", out_dir / "images.jsonl.gz"
    _write_jsonl_gz(events_path, (_event_row(e, spec) for e in events), digest)
    _write_jsonl_gz(images_path, (r for e in events for r in _image_rows(e)), digest)
    return BuildResult(
        version=f"{spec.name}-{digest.hexdigest()[:12]}",
        events=events,
        species=species,
        supported_classes=[EMPTY_CLASS, *sorted(supported)],
        checks=checks,
        events_path=events_path,
        images_path=images_path,
    )


def partition_counts(events: list[Event]) -> dict[str, dict[str, int]]:
    out: dict[str, Counter[str]] = defaultdict(Counter)
    for e in events:
        if e.excluded_reason:
            out["excluded"][e.excluded_reason] += 1
        else:
            assert e.partition is not None and e.role is not None
            out[e.partition.value][e.role.value] += 1
            out[e.partition.value]["events"] += 1
            out[e.partition.value]["images"] += len(e.images)
    return {k: dict(sorted(v.items())) for k, v in sorted(out.items())}


def lock_payload(spec: SplitSpec, result: BuildResult) -> dict[str, Any]:
    return {
        "split_version": result.version,
        "inventory_manifest_version": spec.inventory_manifest_version,
        "grouping_rule": spec.grouping_rule,
        "supported_classes": result.supported_classes,
        "partitions": partition_counts(result.events),
        "checks_passed": result.passed,
    }
