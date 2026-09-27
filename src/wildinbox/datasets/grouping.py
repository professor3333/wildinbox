"""Versioned rules that group images into capture events.

- `sequence_id/v1`: images sharing a supplied sequence id form one event
  (public datasets, where sequence ids are globally unique; used by the
  dataset build).
- `sequence_id/v2`: uploads. A sequence id only means something within one
  camera, since camera counters restart and overlap. Images form one event when
  they share the camera AND the sequence id, and consecutive capture times are
  at most `split_seconds` apart (a counter reused hours later starts a new
  event). Images without a capture time stay with their sequence's first run.
  The worker groups each batch separately, so ids never join across uploads.
- `time_gap/v1`: for uploads without sequence ids. Images from the same camera,
  ordered by timestamp, stay in one event while consecutive gaps are at most
  `gap_seconds`. Images without a timestamp or camera each form their own
  event. Images from different cameras are never grouped.

The default gap comes from CCT20: every within-sequence gap is <= 3 s, so 5 s
never splits a trigger burst. Back-to-back re-triggers (19% of consecutive
sequences start <= 5 s apart) are merged, so a time-grouped event can span
several triggers of one visit.

An upload may set its own gap (`gap_seconds`, 0 to `MAX_GAP_SECONDS`). The
effective gap is recorded in the batch's manifest when the batch is created,
and the worker groups from that record on every attempt, so neither a retry
nor a later change to the default regroups a batch.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise

SEQUENCE_RULE = "sequence_id/v2"
# Far above any gap inside a trigger burst (<= 3 s in CCT20), far below the
# time between two uses of a restarted counter.
SEQUENCE_SPLIT_SECONDS = 300.0
DEFAULT_GAP_SECONDS = 5.0
# One hour: longer gaps would merge unrelated visits into one event.
MAX_GAP_SECONDS = 3600.0


def time_gap_rule_id(gap_seconds: float) -> str:
    return f"time_gap/v1(gap_s={gap_seconds:g})"


def batch_grouping(gap_seconds: float | None) -> dict[str, object]:
    """The grouping a batch is created with, recorded in its manifest: the
    requested gap, or the default when the upload named none."""
    gap = DEFAULT_GAP_SECONDS if gap_seconds is None else float(gap_seconds)
    return {
        "gap_seconds": gap,
        "gap_source": "default" if gap_seconds is None else "request",
        "time_gap_rule": time_gap_rule_id(gap),
        "sequence_rule": SEQUENCE_RULE,
    }


def recorded_gap(manifest: dict[str, object]) -> float:
    """The gap a batch was created with. Every batch records one (earlier batches
    were backfilled with the default of their time), so a missing record is an
    error rather than a reason to fall back to today's default."""
    grouping = manifest.get("grouping")
    if not isinstance(grouping, dict) or "gap_seconds" not in grouping:
        raise ValueError("batch manifest has no recorded grouping gap")
    return float(grouping["gap_seconds"])


@dataclass(frozen=True)
class GroupableImage:
    image_id: str
    camera_id: str | None
    captured_at: datetime | None
    sequence_id: str | None = None


def group_by_sequence(
    images: Sequence[GroupableImage], split_seconds: float = SEQUENCE_SPLIT_SECONDS
) -> list[list[str]]:
    """`sequence_id/v2`: group by (camera, sequence id), split where consecutive
    capture times are more than `split_seconds` apart. Images without a
    sequence id become single-image events."""
    groups: dict[tuple[str, str], list[GroupableImage]] = defaultdict(list)
    singles: list[list[str]] = []
    for img in images:
        if img.sequence_id:
            groups[(img.camera_id or "", img.sequence_id)].append(img)
        else:
            singles.append([img.image_id])
    events: list[list[str]] = []
    for members in groups.values():
        timed = sorted(
            (m for m in members if m.captured_at is not None),
            key=lambda m: (m.captured_at, m.image_id),
        )
        runs: list[list[GroupableImage]] = [[]]
        for prev, img in zip([None, *timed], timed, strict=False):
            if prev is not None:
                assert prev.captured_at is not None and img.captured_at is not None
                if (img.captured_at - prev.captured_at).total_seconds() > split_seconds:
                    runs.append([])
            runs[-1].append(img)
        runs[0].extend(m for m in members if m.captured_at is None)
        events.extend(sorted(m.image_id for m in run) for run in runs if run)
    return sorted([*events, *singles])


def group_by_time_gap(
    images: Sequence[GroupableImage], gap_seconds: float = DEFAULT_GAP_SECONDS
) -> list[list[str]]:
    """Group same-camera images whose consecutive timestamps are <= gap_seconds apart."""
    if gap_seconds < 0:
        raise ValueError(f"gap_seconds must be >= 0, got {gap_seconds}")
    by_camera: dict[str, list[GroupableImage]] = defaultdict(list)
    events: list[list[str]] = []
    for img in images:
        if img.camera_id is None or img.captured_at is None:
            events.append([img.image_id])
        else:
            by_camera[img.camera_id].append(img)
    for imgs in by_camera.values():
        imgs.sort(key=lambda i: (i.captured_at, i.image_id))
        current = [imgs[0]]
        for prev, img in pairwise(imgs):
            assert prev.captured_at is not None and img.captured_at is not None
            if (img.captured_at - prev.captured_at).total_seconds() <= gap_seconds:
                current.append(img)
            else:
                events.append([i.image_id for i in current])
                current = [img]
        events.append([i.image_id for i in current])
    return sorted(events)
