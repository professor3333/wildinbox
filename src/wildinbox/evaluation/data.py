"""Rows from the split manifests, restricted to development partitions.

The locked final test is refused here, so no development code path can read
it by accident.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wildinbox.datasets.spec import Partition


class FinalTestAccessError(RuntimeError):
    pass


def assert_development(partitions: Iterable[Partition | str]) -> list[Partition]:
    parts = [Partition(p) for p in partitions]
    if Partition.FINAL_TEST in parts:
        raise FinalTestAccessError(
            "the locked final test is not available during development; it is opened once, "
            "after models, calibration, and thresholds are frozen"
        )
    return parts


@dataclass(frozen=True)
class ImageRow:
    source_id: str
    event_id: str
    partition: Partition
    camera_id: str
    storage_path: str
    image_label: str | None
    event_label: str | None
    event_role: str
    use_for_fit: bool
    max_box_area: float | None = None


@dataclass(frozen=True)
class EventRow:
    event_id: str
    partition: Partition
    camera_id: str
    role: str
    label: str | None
    animal_present: bool
    image_ids: tuple[str, ...]


def _opt(value: Any) -> str | None:
    return None if value is None else str(value)


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt") as f:
        for line in f:
            yield json.loads(line)


def load_rows(
    split_dir: Path,
    partitions: Iterable[Partition | str],
    boxes: dict[str, float | None] | None = None,
) -> tuple[list[ImageRow], dict[str, EventRow]]:
    wanted = set(assert_development(partitions))
    boxes = boxes or {}
    images = [
        ImageRow(
            source_id=str(r["source_id"]),
            event_id=str(r["event_id"]),
            partition=Partition(str(r["partition"])),
            camera_id=str(r["camera_id"]),
            storage_path=str(r["storage_path"]),
            image_label=_opt(r["image_label"]),
            event_label=_opt(r["event_label"]),
            event_role=str(r["event_role"]),
            use_for_fit=bool(r["use_for_fit"]),
            max_box_area=boxes.get(str(r["source_id"])),
        )
        for r in _jsonl(split_dir / "images.jsonl.gz")
        if r["partition"] in wanted
    ]
    events = {
        str(r["event_id"]): EventRow(
            event_id=str(r["event_id"]),
            partition=Partition(str(r["partition"])),
            camera_id=str(r["camera_id"]),
            role=str(r["role"]),
            label=_opt(r["label"]),
            animal_present=bool(r["animal_present"]),
            image_ids=tuple(r["image_ids"]),
        )
        for r in _jsonl(split_dir / "events.jsonl.gz")
        if r["partition"] in wanted
    }
    return images, events


def box_areas(inventory_db: Path) -> dict[str, float | None]:
    """Largest annotated bounding box per image, as a fraction of the frame
    (annotation coordinates are in the original, full-resolution frame)."""
    out: dict[str, float | None] = {}
    conn = sqlite3.connect(inventory_db)
    for sid, raw_image, raw_anns in conn.execute(
        "SELECT source_id, raw_image, raw_annotations FROM records"
    ):
        img, anns = json.loads(raw_image), json.loads(raw_anns)
        w, h = img.get("width"), img.get("height")
        areas = [a["bbox"][2] * a["bbox"][3] / (w * h) for a in anns if a.get("bbox") and w and h]
        out[sid] = max(areas) if areas else None
    conn.close()
    return out


def box_lists(inventory_db: Path) -> dict[str, list[tuple[float, float, float, float]]]:
    """Annotated boxes per image as (x, y, w, h) fractions of the original frame."""
    out: dict[str, list[tuple[float, float, float, float]]] = {}
    conn = sqlite3.connect(inventory_db)
    for sid, raw_image, raw_anns in conn.execute(
        "SELECT source_id, raw_image, raw_annotations FROM records"
    ):
        img, anns = json.loads(raw_image), json.loads(raw_anns)
        w, h = img.get("width"), img.get("height")
        if not (w and h):
            continue
        boxes = [
            (a["bbox"][0] / w, a["bbox"][1] / h, a["bbox"][2] / w, a["bbox"][3] / h)
            for a in anns
            if a.get("bbox")
        ]
        if boxes:
            out[sid] = boxes
    conn.close()
    return out
