"""Build the versioned inventory: every source record ends up accepted,
quarantined, or intentionally excluded, with reasons.

Original annotations are stored untouched; normalized labels are stored in
separate columns together with the rule that produced them. A label is only
ever assigned to an *accepted* record that has annotations, so a failed
download or unreadable image can never become an `empty` example.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import re
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from wildinbox.class_map import EMPTY_CLASS
from wildinbox.ingestion.sources import SourceConfig
from wildinbox.ingestion.validate import CHECK_VERSION, FileCheck, check_file, unpack_signatures

log = logging.getLogger(__name__)

LABEL_RULE = "lowercase_snake(category name)"
ASPECT_TOLERANCE = 0.02
# Cosine similarity of high-pass signatures. Measured on CCT20: re-encoded,
# resized, brightened, or cropped copies score >= 0.89; unrelated
# cross-camera pairs score <= 0.48.
NEAR_DUPLICATE_MIN_SIMILARITY = 0.8
_SEARCH_CHUNK = 2048

# Bump when the table layout changes; the inventory is derived data and is
# rebuilt from the sources.
SCHEMA_VERSION = 2


class Status(StrEnum):
    ACCEPTED = "accepted"
    QUARANTINED = "quarantined"
    EXCLUDED = "excluded"


class Reason(StrEnum):
    """Why a record is quarantined or excluded."""

    MISSING_FILE = "missing_file"
    UNREADABLE_IMAGE = "unreadable_image"
    NO_ANNOTATION = "no_annotation"
    UNKNOWN_CATEGORY = "unknown_category"
    CONFLICTING_ANNOTATIONS = "conflicting_annotations"
    MISSING_CAMERA = "missing_camera"
    MISSING_SEQUENCE = "missing_sequence"
    EXACT_DUPLICATE = "exact_duplicate"
    CONFIGURED_EXCLUSION = "configured_exclusion"


class Warning_(StrEnum):
    """Noted in the report, but the record stays accepted."""

    MISSING_TIMESTAMP = "missing_timestamp"
    BAD_TIMESTAMP = "bad_timestamp"
    ASPECT_MISMATCH = "aspect_mismatch"
    MULTI_SPECIES = "multi_species"
    SEQUENCE_INCOMPLETE = "sequence_incomplete"
    SEQUENCE_MULTIPLE_CAMERAS = "sequence_multiple_cameras"
    NEAR_DUPLICATE_OTHER_CAMERA = "near_duplicate_other_camera"


EXCLUDING = {Reason.EXACT_DUPLICATE, Reason.CONFIGURED_EXCLUSION}

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    source_id        TEXT PRIMARY KEY,
    source_file      TEXT NOT NULL,
    raw_image        TEXT NOT NULL,
    raw_annotations  TEXT NOT NULL,
    file_name        TEXT,
    camera_id        TEXT,
    sequence_id      TEXT,
    frame_num        INTEGER,
    seq_num_frames   INTEGER,
    captured_at      TEXT,
    storage_path     TEXT,
    file_size        INTEGER,
    file_mtime_ns    INTEGER,
    sha256           TEXT,
    width            INTEGER,
    height           INTEGER,
    signature        BLOB,
    low_information  INTEGER,
    check_version    INTEGER,
    decode_error     TEXT,
    original_labels  TEXT NOT NULL,
    normalized_label TEXT,
    label_rule       TEXT,
    status           TEXT NOT NULL CHECK (status IN ('accepted', 'quarantined', 'excluded')),
    reasons          TEXT NOT NULL,
    warnings         TEXT NOT NULL,
    CHECK (status = 'accepted' OR normalized_label IS NULL),
    CHECK (normalized_label IS NULL OR raw_annotations != '[]')
);
CREATE TABLE IF NOT EXISTS categories (
    source_file  TEXT NOT NULL,
    category_id  INTEGER NOT NULL,
    name         TEXT NOT NULL,
    normalized   TEXT NOT NULL,
    rule         TEXT NOT NULL,
    PRIMARY KEY (source_file, category_id)
);
CREATE TABLE IF NOT EXISTS orphan_annotations (
    source_file TEXT NOT NULL,
    annotation  TEXT NOT NULL,
    reason      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS orphan_files (
    storage_path TEXT PRIMARY KEY,
    size         INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS near_duplicates (
    a        TEXT NOT NULL,
    b        TEXT NOT NULL,
    similarity REAL NOT NULL,
    PRIMARY KEY (a, b)
);
CREATE TABLE IF NOT EXISTS runs (
    run_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at       TEXT NOT NULL,
    finished_at      TEXT NOT NULL,
    manifest_version TEXT NOT NULL,
    counts           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS records_status ON records (status);
CREATE INDEX IF NOT EXISTS records_sha ON records (sha256);
"""


def normalize_label(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class _Record:
    source_id: str
    source_file: str
    image: dict[str, Any]
    annotations: list[dict[str, Any]] = field(default_factory=list)
    original_labels: list[str] = field(default_factory=list)
    storage_path: str | None = None
    check: FileCheck | None = None
    reasons: list[dict[str, str]] = field(default_factory=list)
    warnings: list[dict[str, str]] = field(default_factory=list)
    normalized_label: str | None = None

    def reason(self, code: Reason, detail: str = "") -> None:
        self.reasons.append({"code": code.value, "detail": detail})

    def warn(self, code: Warning_, detail: str = "") -> None:
        self.warnings.append({"code": code.value, "detail": detail})

    @property
    def status(self) -> Status:
        codes = {r["code"] for r in self.reasons}
        if not codes:
            return Status.ACCEPTED
        if codes <= {r.value for r in EXCLUDING}:
            return Status.EXCLUDED
        return Status.QUARANTINED

    @property
    def camera_id(self) -> str | None:
        loc = self.image.get("location")
        return None if loc is None or str(loc).strip() == "" else str(loc)

    @property
    def sequence_id(self) -> str | None:
        seq = self.image.get("seq_id")
        return None if seq is None or str(seq).strip() == "" else str(seq)


@dataclass
class Paths:
    root: Path
    name: str

    @property
    def downloads(self) -> Path:
        return self.root / "raw" / self.name / "downloads"

    @property
    def images(self) -> Path:
        return self.root / "raw" / self.name / "images"

    @property
    def annotations(self) -> Path:
        return self.root / "raw" / self.name / "annotations"

    @property
    def database(self) -> Path:
        return self.root / "inventory" / f"{self.name}.sqlite"

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    if conn.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
        existing = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        if existing:
            log.warning("inventory schema changed; rebuilding %s from sources", db_path.name)
        for table in existing:
            if table != "sqlite_sequence":
                conn.execute(f"DROP TABLE {table}")
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.executescript(SCHEMA)
    return conn


# --------------------------------------------------------------------------- load


def _load_sources(
    source: SourceConfig, annotations_dir: Path
) -> tuple[dict[str, _Record], list[tuple[str, dict[str, Any], str]], list[tuple[Any, ...]]]:
    records: dict[str, _Record] = {}
    orphans: list[tuple[str, dict[str, Any], str]] = []
    categories: list[tuple[Any, ...]] = []
    for rel in source.annotation_files:
        data = json.loads((annotations_dir / rel).read_text())
        cats = {c["id"]: c["name"] for c in data.get("categories", [])}
        categories += [(rel, cid, n, normalize_label(n), LABEL_RULE) for cid, n in cats.items()]
        by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for ann in data.get("annotations", []):
            by_image[str(ann.get("image_id"))].append(ann)
        image_ids = set()
        for img in data.get("images", []):
            sid = str(img["id"])
            image_ids.add(sid)
            anns = by_image.get(sid, [])
            if sid in records:
                prev = records[sid]
                same = prev.image == img and prev.annotations == anns
                if not same:
                    prev.reason(
                        Reason.CONFLICTING_ANNOTATIONS, f"also in {rel} with different data"
                    )
                continue
            rec = _Record(source_id=sid, source_file=rel, image=img, annotations=anns)
            unknown = sorted(
                {a.get("category_id") for a in anns if a.get("category_id") not in cats}, key=str
            )
            if unknown:
                rec.reason(Reason.UNKNOWN_CATEGORY, f"category ids {unknown}")
            rec.original_labels = sorted(
                {cats[a["category_id"]] for a in anns if a.get("category_id") in cats}
            )
            records[sid] = rec
        orphans += [
            (rel, a, "annotation refers to an image not in this file")
            for iid, anns in by_image.items()
            if iid not in image_ids
            for a in anns
        ]
    return records, orphans, categories


def _index_files(images_dir: Path) -> dict[str, str]:
    """Map basename -> path relative to images_dir."""
    index: dict[str, str] = {}
    for dirpath, _, files in os.walk(images_dir):
        for f in files:
            if f.endswith(".tmp") or f.startswith("."):
                continue
            rel = os.path.relpath(os.path.join(dirpath, f), images_dir)
            index.setdefault(f, rel)
    return index


# ---------------------------------------------------------------------- checking


def _cached_checks(conn: sqlite3.Connection) -> dict[str, FileCheck]:
    rows = conn.execute(
        "SELECT storage_path, file_size, file_mtime_ns, sha256, width, height, signature, "
        "low_information, decode_error FROM records WHERE storage_path IS NOT NULL "
        "AND sha256 IS NOT NULL AND check_version = ?",
        (CHECK_VERSION,),
    )
    return {
        r["storage_path"]: FileCheck(
            path=r["storage_path"],
            size=r["file_size"],
            mtime_ns=r["file_mtime_ns"],
            sha256=r["sha256"],
            width=r["width"],
            height=r["height"],
            signature=r["signature"],
            low_information=bool(r["low_information"]),
            error=r["decode_error"],
        )
        for r in rows
    }


def _run_checks(
    images_dir: Path, rel_paths: list[str], cache: dict[str, FileCheck], workers: int
) -> dict[str, FileCheck]:
    results: dict[str, FileCheck] = {}
    todo: list[str] = []
    for rel in rel_paths:
        st = (images_dir / rel).stat()
        cached = cache.get(rel)
        if cached and cached.size == st.st_size and cached.mtime_ns == st.st_mtime_ns:
            results[rel] = cached
        else:
            todo.append(rel)
    log.info("checking %d files (%d unchanged, reused)", len(todo), len(results))
    abs_paths = [str(images_dir / rel) for rel in todo]
    it: Iterable[FileCheck]
    if workers > 1 and len(todo) > 100:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            it = list(pool.map(check_file, abs_paths, chunksize=64))
    else:
        it = map(check_file, abs_paths)
    for rel, chk in zip(todo, it, strict=True):
        results[rel] = FileCheck(**{**chk.__dict__, "path": rel})
    return results


def _validate_record(rec: _Record, exclusions: dict[str, str]) -> None:
    img = rec.image
    if rec.source_id in exclusions:
        rec.reason(Reason.CONFIGURED_EXCLUSION, exclusions[rec.source_id])
    if not rec.annotations:
        rec.reason(Reason.NO_ANNOTATION, "no annotation records; never treated as empty")
    labels = set(rec.original_labels)
    if EMPTY_CLASS in {normalize_label(x) for x in labels} and len(labels) > 1:
        rec.reason(Reason.CONFLICTING_ANNOTATIONS, f"empty together with {sorted(labels)}")
    elif len(labels) > 1:
        rec.warn(Warning_.MULTI_SPECIES, ", ".join(sorted(labels)))
    if rec.camera_id is None:
        rec.reason(Reason.MISSING_CAMERA, "no location")
    if rec.sequence_id is None:
        rec.reason(Reason.MISSING_SEQUENCE, "no seq_id")

    raw_ts = img.get("date_captured")
    if raw_ts in (None, ""):
        rec.warn(Warning_.MISSING_TIMESTAMP)
    else:
        try:
            datetime.fromisoformat(str(raw_ts))
        except ValueError:
            rec.warn(Warning_.BAD_TIMESTAMP, str(raw_ts))

    if rec.storage_path is None:
        rec.reason(Reason.MISSING_FILE, f"{img.get('file_name')} not found")
        return
    chk = rec.check
    assert chk is not None
    if chk.error:
        rec.reason(Reason.UNREADABLE_IMAGE, chk.error)
        return
    mw, mh = img.get("width"), img.get("height")
    if mw and mh and chk.width and chk.height:
        meta_ratio, real_ratio = mw / mh, chk.width / chk.height
        if abs(meta_ratio - real_ratio) / meta_ratio > ASPECT_TOLERANCE:
            rec.warn(Warning_.ASPECT_MISMATCH, f"metadata {mw}x{mh}, file {chk.width}x{chk.height}")


def _check_sequences(records: Iterable[_Record]) -> None:
    by_seq: dict[str, list[_Record]] = defaultdict(list)
    for r in records:
        if r.sequence_id:
            by_seq[r.sequence_id].append(r)
    for seq, recs in by_seq.items():
        cams = {r.camera_id for r in recs}
        if len(cams) > 1:
            for r in recs:
                r.warn(Warning_.SEQUENCE_MULTIPLE_CAMERAS, f"{seq}: {sorted(map(str, cams))}")
        expected = {r.image.get("seq_num_frames") for r in recs}
        if len(expected) == 1 and (n := expected.pop()) is not None and n != len(recs):
            for r in recs:
                r.warn(Warning_.SEQUENCE_INCOMPLETE, f"{len(recs)} of {n} frames present")


def _label_of(rec: _Record) -> tuple[str, ...]:
    return tuple(sorted({normalize_label(x) for x in rec.original_labels}))


def _resolve_exact_duplicates(records: Iterable[_Record]) -> int:
    groups: dict[str, list[_Record]] = defaultdict(list)
    for r in records:
        # Only records that are otherwise acceptable compete to be the kept copy.
        if r.status is Status.ACCEPTED and r.check and r.check.sha256:
            groups[r.check.sha256].append(r)
    n = 0
    for sha, recs in groups.items():
        if len(recs) < 2:
            continue
        n += 1
        recs.sort(key=lambda r: r.source_id)
        ids = [r.source_id for r in recs]
        if len({_label_of(r) for r in recs}) > 1:
            for r in recs:
                r.reason(
                    Reason.CONFLICTING_ANNOTATIONS,
                    f"identical file {sha[:12]} labeled differently across {ids}",
                )
        else:
            for r in recs[1:]:
                r.reason(Reason.EXACT_DUPLICATE, f"same bytes as {recs[0].source_id}")
    return n


def _near_duplicates(records: Iterable[_Record]) -> list[tuple[str, str, float]]:
    """Suspected near-duplicates across *different* cameras (possible leakage).

    Exhaustive: every accepted image is compared with every image from another
    camera. Frames within one camera are expected to look alike, and splits
    hold out whole cameras, so only cross-camera matches are suspicious.
    """
    items = [
        r
        for r in records
        if r.status is Status.ACCEPTED
        and r.check
        and r.check.signature
        and not r.check.low_information
    ]
    if len(items) < 2:
        return []
    sigs = unpack_signatures([r.check.signature for r in items if r.check and r.check.signature])
    cams = np.array([r.camera_id or "" for r in items])
    pairs: list[tuple[str, str, float]] = []
    for start in range(0, len(items), _SEARCH_CHUNK):
        stop = min(start + _SEARCH_CHUNK, len(items))
        sim = sigs[start:stop] @ sigs.T
        sim[cams[start:stop, None] == cams[None, :]] = -1  # same camera: not suspicious
        rows, cols = np.nonzero(sim >= NEAR_DUPLICATE_MIN_SIMILARITY)
        for i, j in zip(rows + start, cols, strict=True):
            if i < j:
                a, b = sorted((items[i].source_id, items[j].source_id))
                pairs.append((a, b, round(float(sim[i - start, j]), 4)))
    return sorted(pairs)


# -------------------------------------------------------------------- manifest

_MANIFEST_FIELDS = (
    "source_id",
    "source_file",
    "file_name",
    "camera_id",
    "sequence_id",
    "frame_num",
    "seq_num_frames",
    "captured_at",
    "storage_path",
    "sha256",
    "width",
    "height",
    "original_labels",
    "normalized_label",
    "label_rule",
    "status",
    "reasons",
    "warnings",
)


def iter_manifest(conn: sqlite3.Connection) -> Iterator[dict[str, Any]]:
    cols = ", ".join(_MANIFEST_FIELDS)
    for row in conn.execute(f"SELECT {cols} FROM records ORDER BY source_id"):
        d = dict(row)
        for k in ("original_labels", "reasons", "warnings"):
            d[k] = json.loads(d[k])
        yield d


def manifest_version(conn: sqlite3.Connection, name: str) -> str:
    h = hashlib.sha256()
    for row in iter_manifest(conn):
        h.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
        h.update(b"\n")
    return f"{name}-{h.hexdigest()[:12]}"


def export_manifest(conn: sqlite3.Connection, out_dir: Path, version: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{version}.jsonl.gz"
    tmp = path.with_suffix(".tmp")
    # mtime=0 keeps the gzip bytes identical across runs
    with open(tmp, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as f:
        for row in iter_manifest(conn):
            f.write(json.dumps(row, sort_keys=True).encode() + b"\n")
    os.replace(tmp, path)
    return path


def counts(conn: sqlite3.Connection) -> dict[str, Any]:
    by_status = dict(
        conn.execute("SELECT status, COUNT(*) FROM records GROUP BY status").fetchall()
    )
    reasons: Counter[str] = Counter()
    for (raw,) in conn.execute("SELECT reasons FROM records WHERE status != 'accepted'"):
        reasons.update({r["code"] for r in json.loads(raw)})
    total = conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    return {
        "source_records": total,
        "by_status": {s.value: by_status.get(s.value, 0) for s in Status},
        "by_reason": dict(sorted(reasons.items())),
        "orphan_annotations": conn.execute("SELECT COUNT(*) FROM orphan_annotations").fetchone()[0],
        "orphan_files": conn.execute("SELECT COUNT(*) FROM orphan_files").fetchone()[0],
        "near_duplicate_pairs": conn.execute("SELECT COUNT(*) FROM near_duplicates").fetchone()[0],
    }


# ------------------------------------------------------------------------ ingest


@dataclass(frozen=True)
class IngestResult:
    version: str
    manifest_path: Path
    counts: dict[str, Any]


def ingest(source: SourceConfig, paths: Paths, *, workers: int = 1) -> IngestResult:
    """Validate every source record and (re)write the inventory. Idempotent."""
    started = _now()
    conn = connect(paths.database)
    records, orphan_anns, categories = _load_sources(source, paths.annotations)
    exclusions = {e.source_id: e.reason for e in source.exclusions}

    files = _index_files(paths.images) if paths.images.exists() else {}
    for rec in records.values():
        rec.storage_path = files.get(Path(str(rec.image.get("file_name", ""))).name)
    referenced = {r.storage_path for r in records.values() if r.storage_path}
    checks = _run_checks(paths.images, sorted(referenced), _cached_checks(conn), workers)
    for rec in records.values():
        if rec.storage_path:
            rec.check = checks[rec.storage_path]

    for rec in records.values():
        _validate_record(rec, exclusions)
    _check_sequences(records.values())
    _resolve_exact_duplicates(records.values())
    near = _near_duplicates(records.values())
    near_ids: dict[str, list[str]] = defaultdict(list)
    for a, b, _ in near:
        near_ids[a].append(b)
        near_ids[b].append(a)
    for sid, others in near_ids.items():
        records[sid].warn(Warning_.NEAR_DUPLICATE_OTHER_CAMERA, ", ".join(sorted(others)))

    # Labels last: only accepted, annotated, single-label records get one.
    for rec in records.values():
        labels = _label_of(rec)
        if rec.status is Status.ACCEPTED and len(labels) == 1:
            rec.normalized_label = labels[0]

    orphan_files = sorted(set(files.values()) - referenced)
    with conn:
        conn.execute(
            "DELETE FROM records WHERE source_id NOT IN (SELECT value FROM json_each(?))",
            (json.dumps(list(records)),),
        )
        conn.executemany(
            f"""INSERT INTO records ({", ".join(_RECORD_COLUMNS)})
               VALUES ({", ".join("?" * len(_RECORD_COLUMNS))})
               ON CONFLICT(source_id) DO UPDATE SET
               {", ".join(f"{c}=excluded.{c}" for c in _RECORD_COLUMNS[1:])}""",
            [_row(r) for r in records.values()],
        )
        conn.execute("DELETE FROM categories")
        conn.executemany("INSERT INTO categories VALUES (?,?,?,?,?)", categories)
        conn.execute("DELETE FROM orphan_annotations")
        conn.executemany(
            "INSERT INTO orphan_annotations VALUES (?,?,?)",
            [(f, json.dumps(a, sort_keys=True), why) for f, a, why in orphan_anns],
        )
        conn.execute("DELETE FROM orphan_files")
        conn.executemany(
            "INSERT INTO orphan_files VALUES (?,?)",
            [(p, (paths.images / p).stat().st_size) for p in orphan_files],
        )
        conn.execute("DELETE FROM near_duplicates")
        conn.executemany("INSERT INTO near_duplicates VALUES (?,?,?)", near)

    version = manifest_version(conn, source.name)
    manifest = export_manifest(conn, paths.manifests, version)
    summary = counts(conn)
    with conn:
        conn.execute(
            "INSERT INTO runs (started_at, finished_at, manifest_version, counts) VALUES (?,?,?,?)",
            (started, _now(), version, json.dumps(summary)),
        )
    conn.close()
    return IngestResult(version=version, manifest_path=manifest, counts=summary)


_RECORD_COLUMNS = (
    "source_id",
    "source_file",
    "raw_image",
    "raw_annotations",
    "file_name",
    "camera_id",
    "sequence_id",
    "frame_num",
    "seq_num_frames",
    "captured_at",
    "storage_path",
    "file_size",
    "file_mtime_ns",
    "sha256",
    "width",
    "height",
    "signature",
    "low_information",
    "check_version",
    "decode_error",
    "original_labels",
    "normalized_label",
    "label_rule",
    "status",
    "reasons",
    "warnings",
)


def _row(r: _Record) -> tuple[Any, ...]:
    c = r.check
    raw_ts = r.image.get("date_captured")
    try:
        captured = datetime.fromisoformat(str(raw_ts)).isoformat() if raw_ts else None
    except ValueError:
        captured = None
    return (
        r.source_id,
        r.source_file,
        json.dumps(r.image, sort_keys=True),
        json.dumps(r.annotations, sort_keys=True),
        r.image.get("file_name"),
        r.camera_id,
        r.sequence_id,
        r.image.get("frame_num"),
        r.image.get("seq_num_frames"),
        captured,
        r.storage_path,
        c.size if c else None,
        c.mtime_ns if c else None,
        c.sha256 if c else None,
        c.width if c else None,
        c.height if c else None,
        c.signature if c else None,
        int(c.low_information) if c else None,
        c.check_version if c else None,
        c.error if c else None,
        json.dumps(r.original_labels),
        r.normalized_label,
        LABEL_RULE if r.normalized_label else None,
        r.status.value,
        json.dumps(r.reasons),
        json.dumps(r.warnings),
    )
