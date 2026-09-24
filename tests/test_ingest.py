from __future__ import annotations

import gzip
import io
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageEnhance

from wildinbox.ingestion.inventory import Paths, Status, ingest
from wildinbox.ingestion.pipeline import LockMismatchError, check_or_write_lock, lock_payload
from wildinbox.ingestion.report import write_report
from wildinbox.ingestion.sources import Archive, Exclusion, SourceConfig

CATS = [{"id": 30, "name": "empty"}, {"id": 3, "name": "raccoon"}, {"id": 9, "name": "coyote"}]
CAT_ID = {"empty": 30, "raccoon": 3, "coyote": 9}


def _jpeg(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    grad = np.linspace(0, 200, 128, dtype=np.float64)[None, :, None].repeat(96, 0).repeat(3, 2)
    arr = np.clip(grad + rng.normal(0, 40, grad.shape), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=90)
    return buf.getvalue()


def _brighter_png(jpeg: bytes) -> bytes:
    img = ImageEnhance.Brightness(Image.open(io.BytesIO(jpeg))).enhance(1.03)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _img(
    sid: str,
    cam: Any = "A",
    seq: Any = None,
    date: Any = "2012-05-09 07:33:45",
    frames: int = 1,
    frame: int = 1,
) -> dict[str, Any]:
    return {
        "id": sid,
        "file_name": f"{sid}.jpg",
        "location": cam,
        "seq_id": seq if seq is not None else f"seq-{sid}",
        "seq_num_frames": frames,
        "frame_num": frame,
        "date_captured": date,
        "width": 128,
        "height": 96,
    }


def _ann(sid: str, *labels: str, cat_id: int | None = None) -> list[dict[str, Any]]:
    if cat_id is not None:
        return [{"id": f"a-{sid}", "image_id": sid, "category_id": cat_id}]
    return [
        {"id": f"a-{sid}-{i}", "image_id": sid, "category_id": CAT_ID[lab]}
        for i, lab in enumerate(labels)
    ]


@pytest.fixture
def dataset(tmp_path: Path) -> tuple[SourceConfig, Paths]:
    paths = Paths(root=tmp_path / "data", name="toy")
    paths.images.mkdir(parents=True)
    paths.annotations.mkdir(parents=True)
    raccoon = _jpeg(1)
    dup = _jpeg(7)
    dupx = _jpeg(8)
    files = {
        "ok_raccoon": raccoon,
        "ok_empty": _jpeg(2),
        "corrupt": b"\xff\xd8 not really a jpeg",
        "truncated": _jpeg(3)[:400],
        "no_ann": _jpeg(4),
        "unknown_cat": _jpeg(5),
        "conflict": _jpeg(6),
        "no_cam": _jpeg(9),
        "no_seq": _jpeg(10),
        "dup1": dup,
        "dup2": dup,
        "dupx1": dupx,
        "dupx2": dupx,
        "multi": _jpeg(11),
        "near_other_cam": _brighter_png(raccoon),
        "excluded_me": _jpeg(12),
        "bad_ts": _jpeg(13),
        "twice": _jpeg(14),
        "frame_a": _jpeg(15),
    }
    for sid, data in files.items():
        (paths.images / f"{sid}.jpg").write_bytes(data)
    (paths.images / "orphan_file.jpg").write_bytes(_jpeg(99))

    train_imgs = [
        _img("ok_raccoon"),
        _img("ok_empty"),
        _img("missing"),
        _img("corrupt"),
        _img("truncated"),
        _img("no_ann"),
        _img("unknown_cat"),
        _img("conflict"),
        _img("no_cam", cam=None),
        _img("no_seq", seq=""),
        _img("dup1"),
        _img("dup2", cam="B"),
        _img("dupx1"),
        _img("dupx2"),
        _img("multi"),
        _img("near_other_cam", cam="B"),
        _img("excluded_me"),
        _img("bad_ts", date="yesterday-ish"),
        _img("twice"),
        _img("frame_a", seq="seq-3", frames=3),
    ]
    train_anns = (
        _ann("ok_raccoon", "raccoon")
        + _ann("ok_empty", "empty")
        + _ann("missing", "empty")
        + _ann("corrupt", "empty")
        + _ann("truncated", "empty")
        + _ann("unknown_cat", cat_id=999)
        + _ann("conflict", "empty", "raccoon")
        + _ann("no_cam", "coyote")
        + _ann("no_seq", "coyote")
        + _ann("dup1", "coyote")
        + _ann("dup2", "coyote")
        + _ann("dupx1", "coyote")
        + _ann("dupx2", "raccoon")
        + _ann("multi", "raccoon", "coyote")
        + _ann("near_other_cam", "raccoon")
        + _ann("excluded_me", "empty")
        + _ann("bad_ts", "raccoon")
        + _ann("twice", "raccoon")
        + _ann("frame_a", "raccoon")
        + [{"id": "orphan-ann", "image_id": "ghost", "category_id": 3}]
    )
    test_imgs = [_img("twice")]
    test_anns = _ann("twice", "coyote")  # same id, different label: conflict
    for name, imgs, anns in [("train", train_imgs, train_anns), ("test", test_imgs, test_anns)]:
        (paths.annotations / f"{name}.json").write_text(
            json.dumps({"images": imgs, "annotations": anns, "categories": CATS})
        )
    arch = Archive(url="https://example.invalid/x.tgz", filename="x.tgz", size=1, md5="0" * 32)
    source = SourceConfig(
        name="toy",
        images_archive=arch,
        annotations_archive=arch,
        annotation_files=["train.json", "test.json"],
        exclusions=[Exclusion(source_id="excluded_me", reason="photographer test shot")],
    )
    return source, paths


def _rows(paths: Paths) -> dict[str, dict[str, Any]]:
    conn = sqlite3.connect(paths.database)
    conn.row_factory = sqlite3.Row
    out = {}
    for r in conn.execute("SELECT * FROM records"):
        d = dict(r)
        d["reason_codes"] = sorted({x["code"] for x in json.loads(d["reasons"])})
        d["warning_codes"] = sorted({x["code"] for x in json.loads(d["warnings"])})
        out[d["source_id"]] = d
    conn.close()
    return out


EXPECTED = {
    "ok_raccoon": ("accepted", [], "raccoon"),
    "ok_empty": ("accepted", [], "empty"),
    "missing": ("quarantined", ["missing_file"], None),
    "corrupt": ("quarantined", ["unreadable_image"], None),
    "truncated": ("quarantined", ["unreadable_image"], None),
    "no_ann": ("quarantined", ["no_annotation"], None),
    "unknown_cat": ("quarantined", ["unknown_category"], None),
    "conflict": ("quarantined", ["conflicting_annotations"], None),
    "no_cam": ("quarantined", ["missing_camera"], None),
    "no_seq": ("quarantined", ["missing_sequence"], None),
    "dup1": ("accepted", [], "coyote"),
    "dup2": ("excluded", ["exact_duplicate"], None),
    "dupx1": ("quarantined", ["conflicting_annotations"], None),
    "dupx2": ("quarantined", ["conflicting_annotations"], None),
    "multi": ("accepted", [], None),
    "near_other_cam": ("accepted", [], "raccoon"),
    "excluded_me": ("excluded", ["configured_exclusion"], None),
    "bad_ts": ("accepted", [], "raccoon"),
    "twice": ("quarantined", ["conflicting_annotations"], None),
    "frame_a": ("accepted", [], "raccoon"),
}


def test_every_record_accounted_for(dataset: tuple[SourceConfig, Paths]) -> None:
    source, paths = dataset
    result = ingest(source, paths)
    rows = _rows(paths)
    assert set(rows) == set(EXPECTED)
    for sid, (status, reasons, label) in EXPECTED.items():
        r = rows[sid]
        assert (r["status"], r["reason_codes"], r["normalized_label"]) == (
            status,
            reasons,
            label,
        ), sid
    c = result.counts
    assert c["source_records"] == sum(c["by_status"].values()) == len(EXPECTED)
    assert c["orphan_annotations"] == 1
    assert c["orphan_files"] == 1


def test_failed_or_unlabeled_images_never_become_empty(
    dataset: tuple[SourceConfig, Paths],
) -> None:
    source, paths = dataset
    ingest(source, paths)
    rows = _rows(paths)
    for sid in ("missing", "corrupt", "truncated", "no_ann"):
        assert rows[sid]["normalized_label"] is None, sid
        assert rows[sid]["status"] == Status.QUARANTINED
    empties = {sid for sid, r in rows.items() if r["normalized_label"] == "empty"}
    assert empties == {"ok_empty"}


def test_database_refuses_labels_on_rejected_records(
    dataset: tuple[SourceConfig, Paths],
) -> None:
    source, paths = dataset
    ingest(source, paths)
    conn = sqlite3.connect(paths.database)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE records SET normalized_label='empty' WHERE source_id='corrupt'")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "UPDATE records SET status='accepted', reasons='[]', "
            "normalized_label='empty' WHERE source_id='no_ann'"
        )


def test_originals_kept_and_labels_explained(dataset: tuple[SourceConfig, Paths]) -> None:
    source, paths = dataset
    ingest(source, paths)
    r = _rows(paths)["ok_raccoon"]
    assert json.loads(r["raw_annotations"])[0]["category_id"] == 3
    assert json.loads(r["raw_image"])["location"] == "A"
    assert json.loads(r["original_labels"]) == ["raccoon"]
    assert r["label_rule"] == "lowercase_snake(category name)"


def test_warnings(dataset: tuple[SourceConfig, Paths]) -> None:
    source, paths = dataset
    ingest(source, paths)
    rows = _rows(paths)
    assert "multi_species" in rows["multi"]["warning_codes"]
    assert "bad_timestamp" in rows["bad_ts"]["warning_codes"]
    assert "sequence_incomplete" in rows["frame_a"]["warning_codes"]
    assert "near_duplicate_other_camera" in rows["ok_raccoon"]["warning_codes"]
    assert "near_duplicate_other_camera" in rows["near_other_cam"]["warning_codes"]
    assert rows["ok_raccoon"]["sha256"] != rows["near_other_cam"]["sha256"]
    flagged = {
        sid for sid, r in rows.items() if "near_duplicate_other_camera" in r["warning_codes"]
    }
    assert flagged == {"ok_raccoon", "near_other_cam"}  # unrelated images are not flagged


def test_rerun_is_idempotent(dataset: tuple[SourceConfig, Paths]) -> None:
    source, paths = dataset
    first = ingest(source, paths)
    manifest_bytes = first.manifest_path.read_bytes()
    second = ingest(source, paths)
    assert second.version == first.version
    assert second.counts == first.counts
    assert second.manifest_path.read_bytes() == manifest_bytes
    conn = sqlite3.connect(paths.database)
    assert conn.execute("SELECT COUNT(*) FROM records").fetchone()[0] == len(EXPECTED)
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 2
    with gzip.open(first.manifest_path) as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == len(EXPECTED)


def test_changed_file_is_rechecked(dataset: tuple[SourceConfig, Paths]) -> None:
    source, paths = dataset
    first = ingest(source, paths)
    (paths.images / "ok_empty.jpg").write_bytes(b"now corrupt")
    second = ingest(source, paths)
    assert _rows(paths)["ok_empty"]["status"] == Status.QUARANTINED
    assert second.version != first.version


def test_lock_detects_changed_inventory(
    dataset: tuple[SourceConfig, Paths], tmp_path: Path
) -> None:
    source, paths = dataset
    lock = tmp_path / "toy.lock.json"
    first = ingest(source, paths)
    assert check_or_write_lock(lock, lock_payload(source, first), update=False) == "created"
    assert check_or_write_lock(lock, lock_payload(source, first), update=False) == "unchanged"
    (paths.images / "ok_empty.jpg").write_bytes(b"now corrupt")
    second = ingest(source, paths)
    with pytest.raises(LockMismatchError, match="--update-lock"):
        check_or_write_lock(lock, lock_payload(source, second), update=False)
    assert check_or_write_lock(lock, lock_payload(source, second), update=True) == "updated"


def test_report(dataset: tuple[SourceConfig, Paths], tmp_path: Path) -> None:
    source, paths = dataset
    result = ingest(source, paths)
    out = write_report(paths, result.version, tmp_path / "report")
    text = out.read_text()
    for heading in (
        "Accounting",
        "Class counts",
        "Cameras",
        "Sequence sizes",
        "Missing fields",
        "Duplicates",
        "Quarantined",
        "Visual samples",
        "Suspicious records",
    ):
        assert f"## {heading}" in text
    assert (tmp_path / "report" / "images" / "class_raccoon.jpg").exists()
    assert (tmp_path / "report" / "images" / "suspicious.jpg").exists()


def test_schema_change_rebuilds_inventory(dataset: tuple[SourceConfig, Paths]) -> None:
    source, paths = dataset
    first = ingest(source, paths)
    conn = sqlite3.connect(paths.database)
    conn.execute("PRAGMA user_version = 1")  # pretend an older layout
    conn.commit()
    conn.close()
    assert ingest(source, paths).version == first.version
