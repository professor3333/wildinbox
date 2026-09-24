"""Versioned rules that group images into capture events.

- `sequence_id/v1`: images sharing a supplied sequence id form one event
  (public datasets).
- `time_gap/v1`: for uploads without sequence ids. Images from the same camera,
  ordered by timestamp, stay in one event while consecutive gaps are at most
  `gap_seconds`. Images without a timestamp or camera each form their own
  event. Images from different cameras are never grouped.

The default gap comes from CCT20: every within-sequence gap is <= 3 s, so 5 s
never splits a trigger burst. Back-to-back re-triggers (19% of consecutive
sequences start <= 5 s apart) are merged, so a time-grouped event can span
several triggers of one visit.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise

SEQUENCE_RULE = "sequence_id/v1"
DEFAULT_GAP_SECONDS = 5.0


def time_gap_rule_id(gap_seconds: float) -> str:
    return f"time_gap/v1(gap_s={gap_seconds:g})"


@dataclass(frozen=True)
class GroupableImage:
    image_id: str
    camera_id: str | None
    captured_at: datetime | None
    sequence_id: str | None = None


def group_by_sequence(images: Sequence[GroupableImage]) -> list[list[str]]:
    """Group by sequence id; images without one become single-image events."""
    groups: dict[str, list[str]] = defaultdict(list)
    singles: list[list[str]] = []
    for img in sorted(images, key=lambda i: i.image_id):
        if img.sequence_id:
            groups[img.sequence_id].append(img.image_id)
        else:
            singles.append([img.image_id])
    return sorted([*groups.values(), *singles])


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
