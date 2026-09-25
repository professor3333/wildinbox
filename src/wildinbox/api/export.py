"""Observation export: one row per capture event with its current label and the
provenance needed to reproduce or audit it.

The current label follows the latest human review when there is one, else the
automatic decision (only when automation actually decided it), else it is
pending review. Capture events are not individual animals or counts.
"""

from __future__ import annotations

import csv
import io
from datetime import UTC, datetime
from typing import Any

from wildinbox.class_map import EMPTY_CLASS
from wildinbox.storage.models import Decision, Event, ModelRelease

COLUMNS = (
    "event_id",
    "batch_id",
    "camera_id",
    "start_at",
    "end_at",
    "frames",
    "filenames",
    "observation",
    "label_source",
    "review_outcome",
    "reviewer",
    "reviewed_at",
    "review_id",
    "disposition",
    "suggested_label",
    "confidence",
    "reasons",
    "audit_selected",
    "audit_rule",
    "model_release_id",
    "weights_sha256",
    "preprocessing_version",
    "calibration_version",
    "policy_version",
    "exported_at",
)


def observation(event: Event, decision: Decision | None) -> tuple[str | None, str]:
    """(label, source): source is review, automatic, pending_review, or unresolved."""
    review = event.reviews[-1] if event.reviews else None
    if review is not None:
        if review.outcome == "unresolved":
            return None, "unresolved"
        return review.confirmed_label, "review"
    if decision is not None and decision.disposition == "likely_empty":
        return EMPTY_CLASS, "automatic"
    if decision is not None and decision.disposition == "species_identified":
        return decision.suggested_label, "automatic"
    return None, "pending_review"


def rows(
    events: list[Event], latest: dict[Any, Decision], releases: dict[str, ModelRelease]
) -> list[dict[str, Any]]:
    now = datetime.now(UTC).isoformat()
    out = []
    for e in events:
        d = latest.get(e.id)
        r = releases.get(d.model_release_id) if d else None
        label, source = observation(e, d)
        review = e.reviews[-1] if e.reviews else None
        out.append(
            {
                "event_id": str(e.id),
                "batch_id": str(e.batch_id),
                "camera_id": e.camera_id,
                "start_at": e.start_at.isoformat() if e.start_at else None,
                "end_at": e.end_at.isoformat() if e.end_at else None,
                "frames": len(e.images),
                "filenames": [i.original_filename for i in e.images],
                "observation": label,
                "label_source": source,
                "review_outcome": review.outcome if review else None,
                "reviewer": review.reviewer if review else None,
                "reviewed_at": review.created_at.isoformat() if review else None,
                "review_id": str(review.id) if review else None,
                "disposition": d.disposition if d else None,
                "suggested_label": d.suggested_label if d else None,
                "confidence": d.confidence if d else None,
                "reasons": list(d.reasons) if d else [],
                "audit_selected": d.audit_selected if d else False,
                "audit_rule": d.audit_rule if d else None,
                "model_release_id": d.model_release_id if d else None,
                "weights_sha256": r.weights_sha256 if r else None,
                "preprocessing_version": r.preprocessing_version if r else None,
                "calibration_version": (r.calibration or {}).get("version") if r else None,
                "policy_version": d.policy_version if d else None,
                "exported_at": now,
            }
        )
    return out


def to_csv(data: list[dict[str, Any]]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS)
    w.writeheader()
    for row in data:
        w.writerow(
            {
                **row,
                "filenames": ";".join(row["filenames"]),
                "reasons": ";".join(row["reasons"]),
            }
        )
    return buf.getvalue()
