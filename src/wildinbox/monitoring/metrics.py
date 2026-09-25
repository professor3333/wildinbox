"""Monitoring computed from PostgreSQL, the source of truth.

Three separate views, because they answer different questions:

- operations: is processing healthy (queue, leases, failures, throughput)?
- signals: did a camera's incoming data or the model's behaviour change? No
  labels needed, so this can flag trouble but cannot measure accuracy.
- accuracy: how often reviewers corrected the suggestions. Needs reviews, and
  covers only reviewed events.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from wildinbox.evaluation.metrics import wilson
from wildinbox.storage.models import Batch, Event, Image, Job, Prediction

CONFIDENCE_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001)


def load_config(path: Path) -> dict[str, Any]:
    cfg: dict[str, Any] = yaml.safe_load(path.read_text())
    return cfg


def psi(expected: Counter[str], actual: Counter[str], eps: float = 1e-4) -> float:
    """Population stability index between two categorical distributions."""
    keys = set(expected) | set(actual)
    ne, na = sum(expected.values()) or 1, sum(actual.values()) or 1
    total = 0.0
    for k in keys:
        p, q = max(expected[k] / ne, eps), max(actual[k] / na, eps)
        total += (q - p) * math.log(q / p)
    return total


def _bin(conf: float | None) -> str:
    if conf is None:
        return "none"
    for lo, hi in pairwise(CONFIDENCE_BINS):
        if lo <= conf < hi:
            return f"{lo:.1f}-{min(hi, 1.0):.1f}"
    return "none"


# ------------------------------------------------------------------ operations


def operations(
    session: Session, lease_seconds: int, cfg: dict[str, Any], now: datetime
) -> dict[str, Any]:
    window = now - timedelta(days=cfg["window_days"])
    jobs = session.scalars(select(Job)).all()
    by_status = Counter(j.status for j in jobs)
    waiting = [
        j
        for j in jobs
        if j.status == "queued" and (j.next_attempt_at is None or j.next_attempt_at <= now)
    ]
    oldest = max(
        ((now - (j.next_attempt_at or j.created_at)).total_seconds() for j in waiting), default=0.0
    )
    stale = [
        j
        for j in jobs
        if j.status == "running"
        and j.heartbeat_at is not None
        and (now - j.heartbeat_at).total_seconds() > lease_seconds
    ]
    retrying = [j for j in jobs if j.status == "queued" and j.error]
    failed = [j for j in jobs if j.status == "failed" and j.finished_at and j.finished_at >= window]

    uploaded = (
        session.scalar(select(func.count()).select_from(Image).where(Image.created_at >= window))
        or 0
    )
    unreadable = (
        session.scalar(
            select(func.count())
            .select_from(Image)
            .where(Image.created_at >= window, Image.validation_status == "invalid")
        )
        or 0
    )
    scored = (
        session.scalar(
            select(func.count()).select_from(Prediction).where(Prediction.created_at >= window)
        )
        or 0
    )
    frame_errors = (
        session.scalar(
            select(func.count())
            .select_from(Image)
            .where(Image.created_at >= window, Image.processing_error.is_not(None))
        )
        or 0
    )
    last_day = (
        session.scalar(
            select(func.count())
            .select_from(Prediction)
            .where(Prediction.created_at >= now - timedelta(days=1))
        )
        or 0
    )

    # Cost: wall seconds per 1,000 images for jobs that succeeded in the window.
    per_release: dict[str, dict[str, float]] = defaultdict(
        lambda: {"seconds": 0.0, "images": 0.0, "jobs": 0.0}
    )
    for j in jobs:
        if j.status == "succeeded" and j.finished_at and j.started_at and j.finished_at >= window:
            n = (
                session.scalar(
                    select(func.count())
                    .select_from(Prediction)
                    .join(Image)
                    .where(
                        Image.batch_id == j.batch_id,
                        Prediction.model_release_id == j.model_release_id,
                    )
                )
                or 0
            )
            r = per_release[j.model_release_id]
            r["seconds"] += (j.finished_at - j.started_at).total_seconds()
            r["images"] += n
            r["jobs"] += 1
    cost = {
        rid: {
            "jobs": int(v["jobs"]),
            "images": int(v["images"]),
            "seconds_per_1000_images": round(1000 * v["seconds"] / v["images"], 1)
            if v["images"]
            else None,
        }
        for rid, v in per_release.items()
    }
    return {
        "window_days": cfg["window_days"],
        "jobs": dict(sorted(by_status.items())),
        "queued_waiting": len(waiting),
        "oldest_queued_seconds": round(oldest, 1),
        "retrying": len(retrying),
        "stale_leases": len(stale),
        "failed_jobs_in_window": [
            {"job_id": str(j.id), "batch_id": str(j.batch_id), "error": j.error} for j in failed
        ],
        "files_uploaded": uploaded,
        "files_unreadable": unreadable,
        "unreadable_rate": unreadable / uploaded if uploaded else 0.0,
        "images_scored": scored,
        "frames_failed": frame_errors,
        "processing_error_rate": frame_errors / scored if scored else 0.0,
        "images_scored_last_24h": last_day,
        "cost_by_release": cost,
    }


# ------------------------------------------------------------------ per-camera data


@dataclass
class EventView:
    camera: str
    start_at: datetime | None
    release: str | None
    disposition: str | None
    label: str | None
    confidence: float | None
    reasons: list[str]
    review_outcome: str | None
    reviewer: str | None
    night: list[bool]
    blur: list[float]


def event_views(session: Session) -> list[EventView]:
    events = session.scalars(
        select(Event)
        .join(Batch)
        .options(
            selectinload(Event.decisions), selectinload(Event.reviews), selectinload(Event.images)
        )
    ).all()
    out = []
    for e in events:
        d = max(e.decisions, key=lambda d: d.created_at, default=None)
        r = e.reviews[-1] if e.reviews else None
        q = [i.quality for i in e.images if i.quality and "night" in i.quality]
        out.append(
            EventView(
                camera=e.camera_id or "unknown",
                start_at=e.start_at,
                release=d.model_release_id if d else None,
                disposition=d.disposition if d else None,
                label=d.suggested_label if d else None,
                confidence=d.confidence if d else None,
                reasons=list(d.reasons) if d else [],
                review_outcome=r.outcome if r else None,
                reviewer=r.reviewer if r else None,
                night=[bool(x["night"]) for x in q],
                blur=[float(x["blur"]) for x in q],
            )
        )
    return out


def _window(events: list[EventView]) -> dict[str, Any]:
    n = len(events)
    night = [x for e in events for x in e.night]
    blur = [x for e in events for x in e.blur]
    return {
        "events": n,
        "needs_review_share": sum(e.disposition == "needs_review" for e in events) / n
        if n
        else None,
        "low_confidence_share": sum("low_confidence" in e.reasons for e in events) / n
        if n
        else None,
        "labels": Counter(e.label or "none" for e in events),
        "confidence": Counter(_bin(e.confidence) for e in events),
        "night_share": sum(night) / len(night) if night else None,
        "median_blur": statistics.median(blur) if blur else None,
        "frames_with_quality": len(night),
        "releases": sorted({e.release for e in events if e.release}),
        "from": min((e.start_at for e in events if e.start_at), default=None),
        "to": max((e.start_at for e in events if e.start_at), default=None),
    }


def _jsonable_window(w: dict[str, Any]) -> dict[str, Any]:
    return {
        **w,
        "labels": dict(w["labels"].most_common()),
        "confidence": dict(sorted(w["confidence"].items())),
        "from": w["from"].isoformat() if w["from"] else None,
        "to": w["to"].isoformat() if w["to"] else None,
    }


def signals(views: list[EventView], cfg: dict[str, Any]) -> dict[str, Any]:
    by_camera: dict[str, list[EventView]] = defaultdict(list)
    for v in views:
        by_camera[v.camera].append(v)
    out = {}
    for cam, evs in sorted(by_camera.items()):
        timed = sorted((e for e in evs if e.start_at), key=lambda e: e.start_at)  # type: ignore[arg-type, return-value]
        recent, earlier = timed[-cfg["recent_events"] :], timed[: -cfg["recent_events"]]
        overall = _window(evs)
        entry: dict[str, Any] = {
            "events": len(evs),
            "undated_events": len(evs) - len(timed),
            "needs_review_share": overall["needs_review_share"],
            "low_confidence_share": overall["low_confidence_share"],
            "night_share": overall["night_share"],
            "comparison": None,
        }
        if len(recent) >= cfg["min_events"] and len(earlier) >= cfg["min_events"]:
            a, b = _window(earlier), _window(recent)
            blur_change = (
                (b["median_blur"] - a["median_blur"]) / a["median_blur"]
                if a["median_blur"] and b["median_blur"] is not None
                else None
            )
            entry["comparison"] = {
                "earlier": _jsonable_window(a),
                "recent": _jsonable_window(b),
                "label_psi": psi(a["labels"], b["labels"]),
                "confidence_psi": psi(a["confidence"], b["confidence"]),
                "uncertainty_change": b["low_confidence_share"] - a["low_confidence_share"],
                "night_share_change": (
                    b["night_share"] - a["night_share"]
                    if a["night_share"] is not None and b["night_share"] is not None
                    else None
                ),
                "blur_change": blur_change,
                "release_changed": a["releases"] != b["releases"],
            }
        out[cam] = entry
    return out


def accuracy(views: list[EventView], cfg: dict[str, Any]) -> dict[str, Any]:
    """Correction rates from reviews, per camera and per release."""

    def block(evs: list[EventView]) -> dict[str, Any]:
        reviewed = [e for e in evs if e.review_outcome]
        judged = [e for e in reviewed if e.review_outcome in ("confirmed", "corrected")]
        corrected = sum(e.review_outcome == "corrected" for e in judged)
        automated = [e for e in reviewed if e.disposition in ("likely_empty", "species_identified")]
        enough = len(judged) >= cfg["min_reviewed"]
        return {
            "events": len(evs),
            "reviewed": len(reviewed),
            "review_coverage": len(reviewed) / len(evs) if evs else None,
            "unresolved": sum(e.review_outcome == "unresolved" for e in reviewed),
            "judged": len(judged),
            "correction_rate": corrected / len(judged) if enough else None,
            "correction_rate_ci95": wilson(corrected, len(judged)) if enough else None,
            "automated_events_reviewed": len(automated),
            "automated_events_corrected": sum(e.review_outcome == "corrected" for e in automated),
            "reviewers": dict(Counter(e.reviewer for e in reviewed if e.reviewer).most_common()),
        }

    cams: dict[str, list[EventView]] = defaultdict(list)
    rels: dict[str, list[EventView]] = defaultdict(list)
    for v in views:
        cams[v.camera].append(v)
        rels[v.release or "none"].append(v)
    return {
        "by_camera": {k: block(v) for k, v in sorted(cams.items())},
        "by_release": {k: block(v) for k, v in sorted(rels.items())},
    }


# ------------------------------------------------------------------ alerts


def alerts(
    ops: dict[str, Any], sig: dict[str, Any], acc: dict[str, Any], cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def add(
        level: str, area: str, message: str, value: Any, limit: Any, camera: str | None = None
    ) -> None:
        out.append(
            {
                "level": level,
                "area": area,
                "camera": camera,
                "message": message,
                "value": value,
                "limit": limit,
            }
        )

    o, d, a = cfg["operations"], cfg["drift"], cfg["accuracy"]
    if ops["oldest_queued_seconds"] > o["max_queue_age_seconds"]:
        add(
            "warning",
            "operations",
            "a job has waited too long to run",
            ops["oldest_queued_seconds"],
            o["max_queue_age_seconds"],
        )
    if ops["stale_leases"] > o["max_stale_leases"]:
        add(
            "critical",
            "operations",
            "a worker stopped renewing its lease",
            ops["stale_leases"],
            o["max_stale_leases"],
        )
    if len(ops["failed_jobs_in_window"]) > o["max_failed_jobs"]:
        add(
            "critical",
            "operations",
            "jobs failed terminally",
            len(ops["failed_jobs_in_window"]),
            o["max_failed_jobs"],
        )
    if ops["processing_error_rate"] > o["max_processing_error_rate"]:
        add(
            "warning",
            "operations",
            "frames failed inference",
            ops["processing_error_rate"],
            o["max_processing_error_rate"],
        )
    if ops["unreadable_rate"] > o["max_unreadable_rate"]:
        add(
            "warning",
            "operations",
            "many uploaded files were unusable",
            ops["unreadable_rate"],
            o["max_unreadable_rate"],
        )

    for cam, s in sig.items():
        c = s["comparison"]
        if c is None:
            continue
        why = " (the release also changed between windows)" if c["release_changed"] else ""
        checks = [
            ("label_psi", "suggested-label mix shifted", d["max_label_psi"]),
            ("confidence_psi", "confidence distribution shifted", d["max_confidence_psi"]),
            (
                "uncertainty_change",
                "share of low-confidence events changed",
                d["max_uncertainty_change"],
            ),
            ("night_share_change", "share of night frames changed", d["max_night_share_change"]),
            ("blur_change", "image sharpness changed", d["max_blur_change"]),
        ]
        for key, message, limit in checks:
            value = c[key]
            if value is not None and abs(value) > limit:
                add("info", "signal", message + why, round(value, 3), limit, cam)

    for cam, b in acc["by_camera"].items():
        ci = b["correction_rate_ci95"]
        if ci is not None and ci[0] > a["max_correction_rate"]:
            add(
                "warning",
                "accuracy",
                "reviewers correct most suggestions",
                round(b["correction_rate"], 3),
                a["max_correction_rate"],
                cam,
            )
    return out


def summary(
    session: Session, lease_seconds: int, cfg: dict[str, Any], now: datetime | None = None
) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    ops = operations(session, lease_seconds, cfg["operations"], now)
    views = event_views(session)
    sig = signals(views, cfg["drift"])
    acc = accuracy(views, cfg["accuracy"])
    return {
        "generated_at": now.isoformat(),
        "operations": ops,
        "signals": sig,
        "accuracy": acc,
        "alerts": alerts(ops, sig, acc, cfg),
        "notes": {
            "signals": (
                "Label-free: can show that inputs or model behaviour changed, "
                "not whether accuracy did."
            ),
            "accuracy": "From human reviews only; unreviewed events have unknown accuracy.",
        },
    }


def backfill_quality(session: Session, store: Any, chunk: int = 100) -> int:
    """Record quality for valid images scored before workers recorded it."""
    from wildinbox.preprocessing import load_image
    from wildinbox.quality import quality

    done = 0
    while True:
        images = session.scalars(
            select(Image)
            .where(
                Image.validation_status == "valid",
                Image.quality.is_(None),
                Image.storage_key.is_not(None),
            )
            .limit(chunk)
        ).all()
        if not images:
            return done
        for img in images:
            try:
                img.quality = quality(load_image(store.get(img.storage_key)))
            except Exception:
                img.quality = {"error": "could not compute"}
            done += 1
        session.commit()
