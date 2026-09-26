"""Monitoring computed from PostgreSQL, the source of truth.

Two dashboards, several questions:

Operational health
- operations: is processing healthy (queue, leases, failures, throughput,
  latency, workers, API errors, storage, batch cost)?

Model behavior
- behavior: what the model decided (filtered, auto-labeled, sent to review,
  labels, confidence), globally and per camera, for the latest batch against
  the camera's earlier batches, and across time periods with the effect of a
  changing camera mix separated from change within cameras.
- signals: did a camera's incoming photos change over capture time? No labels
  needed, so this can flag trouble but cannot measure accuracy.
- audits: errors found in the random audit sample of automatic decisions
  (false empties, wrong species). The only unbiased error measure of
  automation; a camera with no audit labels has unknown observed quality.
- accuracy: how often reviewers corrected the suggestions. Covers only
  reviewed events, which were mostly chosen because they were uncertain.
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

from wildinbox.class_map import EMPTY_CLASS
from wildinbox.evaluation.metrics import wilson
from wildinbox.storage.models import Batch, Event, Image, Job, Prediction, WorkerProcess

CONFIDENCE_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001)
AUTOMATIC = ("likely_empty", "species_identified")
# Routes serving files or heavy reports are left out of the metadata latency target.
_NOT_METADATA = ("/images/", "/export", "/monitoring", "/metrics", "/view")


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


def _quantiles(values: list[float]) -> dict[str, float | None]:
    v = sorted(values)
    if not v:
        return {"p50": None, "p95": None}
    return {
        "p50": round(v[len(v) // 2], 1),
        "p95": round(v[min(len(v) - 1, int(0.95 * len(v)))], 1),
    }


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
    # Inference outcomes over one population: frames uploaded in the window
    # that decoded and so reached the model. Each one is scored (it has a
    # prediction) or failed (processing_error); frames still waiting count as
    # neither. The rate is failed / attempted, so an outage where nothing
    # scores is 100%, and a window with no attempts has no rate (None), not 0%.
    attempted_frame = (Image.created_at >= window) & (Image.validation_status == "valid")
    frame_errors = (
        session.scalar(
            select(func.count())
            .select_from(Image)
            .where(attempted_frame, Image.processing_error.is_not(None))
        )
        or 0
    )
    frames_scored = (
        session.scalar(
            select(func.count())
            .select_from(Image)
            .where(
                attempted_frame,
                Image.processing_error.is_(None),
                select(Prediction.id).where(Prediction.image_id == Image.id).exists(),
            )
        )
        or 0
    )
    frames_attempted = frames_scored + frame_errors
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
    done = [
        j
        for j in jobs
        if j.status == "succeeded" and j.finished_at and j.started_at and j.finished_at >= window
    ]
    latency = {
        "jobs": len(done),
        "queue_wait_seconds": _quantiles(
            [(j.started_at - j.created_at).total_seconds() for j in done]  # type: ignore[operator]
        ),
        "run_seconds": _quantiles(
            [(j.finished_at - j.started_at).total_seconds() for j in done]  # type: ignore[operator]
        ),
        "upload_to_done_seconds": _quantiles(
            [(j.finished_at - j.created_at).total_seconds() for j in done]  # type: ignore[operator]
        ),
    }
    return {
        "window_days": cfg["window_days"],
        "jobs": dict(sorted(by_status.items())),
        "job_latency": latency,
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
        "frames_attempted": frames_attempted,
        "frames_scored": frames_scored,
        "frames_failed": frame_errors,
        "processing_error_rate": frame_errors / frames_attempted if frames_attempted else None,
        "images_scored_last_24h": last_day,
        "cost_by_release": cost,
    }


def workers(
    session: Session, lease_seconds: int, cfg: dict[str, Any], now: datetime
) -> dict[str, Any]:
    """Worker processes: live ones and their memory, starts, and deaths.

    A worker that has not reported within the lease period and never recorded
    a clean stop is counted as died (killed, crashed, out of memory)."""
    since = now - timedelta(hours=cfg["restart_window_hours"])
    stale = now - timedelta(seconds=lease_seconds)
    rows = session.scalars(
        select(WorkerProcess).where(
            (WorkerProcess.last_seen_at >= since) | (WorkerProcess.started_at >= since)
        )
    ).all()
    live = [w for w in rows if w.stopped_at is None and w.last_seen_at >= stale]
    died = [w for w in rows if w.stopped_at is None and w.last_seen_at < stale]

    def row(w: WorkerProcess) -> dict[str, Any]:
        return {
            "id": w.id,
            "started_at": w.started_at.isoformat(),
            "last_seen_at": w.last_seen_at.isoformat(),
            "rss_mb": round(w.rss_bytes / 2**20, 1) if w.rss_bytes else None,
            "peak_rss_mb": round(w.peak_rss_bytes / 2**20, 1) if w.peak_rss_bytes else None,
        }

    return {
        "window_hours": cfg["restart_window_hours"],
        "live": [row(w) for w in sorted(live, key=lambda w: w.started_at)],
        "starts_in_window": sum(w.started_at >= since for w in rows),
        "stopped_cleanly_in_window": sum(
            w.stopped_at is not None and w.stopped_at >= since for w in rows
        ),
        "died_in_window": [row(w) for w in sorted(died, key=lambda w: w.last_seen_at)],
    }


def storage(session: Session, cfg: dict[str, Any], now: datetime) -> dict[str, Any]:
    """Original photos stored (thumbnails are small and not counted), and daily growth."""
    total_bytes, total_images = session.execute(
        select(func.coalesce(func.sum(Image.size_bytes), 0), func.count()).select_from(Image)
    ).one()
    day = func.date_trunc("day", Image.created_at)
    since = now - timedelta(days=cfg["window_days"])
    daily = session.execute(
        select(day, func.sum(Image.size_bytes), func.count())
        .where(Image.created_at >= since)
        .group_by(day)
        .order_by(day)
    ).all()
    return {
        "original_bytes": int(total_bytes),
        "original_gb": round(int(total_bytes) / 1e9, 3),
        "images": int(total_images),
        "daily": [
            {"day": d.date().isoformat(), "bytes": int(b or 0), "images": int(n)}
            for d, b, n in daily
        ],
    }


def batch_costs(session: Session, limit: int = 20) -> list[dict[str, Any]]:
    """The most recent batches: size, events, processing time, and cost per 1,000 images."""
    batches = session.scalars(
        select(Batch)
        .options(selectinload(Batch.jobs))
        .order_by(Batch.created_at.desc())
        .limit(limit)
    ).all()
    out = []
    for b in batches:
        images = (
            session.scalar(select(func.count()).select_from(Image).where(Image.batch_id == b.id))
            or 0
        )
        events = (
            session.scalar(select(func.count()).select_from(Event).where(Event.batch_id == b.id))
            or 0
        )
        job = max(b.jobs, key=lambda j: j.created_at, default=None)
        run = (
            (job.finished_at - job.started_at).total_seconds()
            if job and job.finished_at and job.started_at
            else None
        )
        out.append(
            {
                "batch_id": str(b.id),
                "created_at": b.created_at.isoformat(),
                "status": b.status,
                "release": job.model_release_id if job else None,
                "images": images,
                "events": events,
                "attempts": job.attempts if job else 0,
                "run_seconds": round(run, 1) if run is not None else None,
                "seconds_per_1000_images": round(1000 * run / images, 1)
                if run is not None and images
                else None,
            }
        )
    return out


def api_health(api: dict[str, dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    """API responses and errors since the API process started (in memory)."""
    responses = sum(v.get("responses", 0) for v in api.values())
    server = sum(v.get("server_errors", 0) for v in api.values())
    client = sum(v.get("client_errors", 0) for v in api.values())
    metadata = {
        k: v
        for k, v in api.items()
        if k.startswith("GET ")
        and not any(x in k for x in _NOT_METADATA)
        and v.get("requests", 0) >= cfg["min_api_requests"]
    }
    slowest = max(metadata.items(), key=lambda kv: kv[1]["p95_ms"], default=None)
    return {
        "responses": responses,
        "server_errors": server,
        "client_errors": client,
        "server_error_rate": server / responses if responses else 0.0,
        "slowest_metadata_route": {"route": slowest[0], "p95_ms": slowest[1]["p95_ms"]}
        if slowest
        else None,
        "routes": api,
    }


# ------------------------------------------------------------------ per-camera data


_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


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
    batch_id: str = ""
    batch_created_at: datetime = _EPOCH
    decided_at: datetime | None = None
    audit_selected: bool = False
    reviewed_label: str | None = None


def event_views(session: Session) -> list[EventView]:
    rows = session.execute(
        select(Event, Batch.created_at)
        .join(Batch)
        .options(
            selectinload(Event.decisions), selectinload(Event.reviews), selectinload(Event.images)
        )
    ).all()
    out = []
    for e, batch_created_at in rows:
        d = max(e.decisions, key=lambda d: d.created_at, default=None)
        r = e.current_review
        q = [i.quality for i in e.images if i.quality and "night" in i.quality]
        out.append(
            EventView(
                camera=e.camera_id or "unknown",
                start_at=e.start_at,
                batch_id=str(e.batch_id),
                batch_created_at=batch_created_at,
                decided_at=d.created_at if d else None,
                audit_selected=bool(d.audit_selected) if d else False,
                reviewed_label=r.confirmed_label if r else None,
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


# ------------------------------------------------------------------ model behavior


def _mix(events: list[EventView]) -> dict[str, Any]:
    """What the model decided for these events (no labels needed)."""
    n = len(events)
    conf = [e.confidence for e in events if e.confidence is not None]

    def share(pred: Any) -> float | None:
        return sum(1 for e in events if pred(e)) / n if n else None

    return {
        "events": n,
        "filtered_share": share(lambda e: e.disposition == "likely_empty"),
        "auto_labeled_share": share(lambda e: e.disposition == "species_identified"),
        "needs_review_share": share(lambda e: e.disposition == "needs_review"),
        "low_confidence_share": share(lambda e: "low_confidence" in e.reasons),
        "unfamiliar_share": share(lambda e: "possible_unknown" in e.reasons),
        "mean_confidence": statistics.fmean(conf) if conf else None,
        "labels": Counter(e.label or "none" for e in events),
        "confidence": Counter(_bin(e.confidence) for e in events),
    }


def _jsonable_mix(m: dict[str, Any]) -> dict[str, Any]:
    return {
        **m,
        "labels": dict(m["labels"].most_common()),
        "confidence": dict(sorted(m["confidence"].items())),
    }


_SHARES = (
    "filtered_share",
    "auto_labeled_share",
    "needs_review_share",
    "low_confidence_share",
    "unfamiliar_share",
    "mean_confidence",
)


def _compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """b against a: PSI of labels and confidence, and the change of each share."""
    out: dict[str, Any] = {
        "label_psi": psi(a["labels"], b["labels"]),
        "confidence_psi": psi(a["confidence"], b["confidence"]),
    }
    for k in _SHARES:
        out[f"{k}_change"] = b[k] - a[k] if a[k] is not None and b[k] is not None else None
    return out


def _weighted(parts: dict[str, dict[str, Any]], weights: Counter[str], key: str) -> Counter[str]:
    """A distribution mixing each camera's `key` distribution by `weights`."""
    total = sum(weights.values()) or 1
    out: Counter[str] = Counter()
    for cam, w in weights.items():
        dist = parts[cam][key]
        n = sum(dist.values()) or 1
        for k, v in dist.items():
            out[k] += (w / total) * (v / n) * 1_000_000  # Counter of pseudo-counts
    return out


def periods(views: list[EventView], cfg: dict[str, Any], now: datetime) -> dict[str, Any]:
    """Recent period against the one before it, globally, with the camera mix separated.

    A global shift can come from a changed camera mix alone. `mix` compares the
    earlier period with the earlier per-camera behaviour reweighted to the recent
    camera mix; `within` compares that with what recent events actually got. A
    large `total` with a small `within` means the cameras changed, not the model's
    behaviour on them. Cameras with no earlier events are new and count as mix."""
    days = cfg["period_days"]
    recent_from, earlier_from = now - timedelta(days=days), now - timedelta(days=2 * days)
    decided = [e for e in views if e.decided_at is not None]
    recent = [e for e in decided if e.decided_at >= recent_from]  # type: ignore[operator]
    earlier = [e for e in decided if earlier_from <= e.decided_at < recent_from]  # type: ignore[operator]
    out: dict[str, Any] = {
        "period_days": days,
        "recent_events": len(recent),
        "earlier_events": len(earlier),
        "cameras": {},
        "comparison": None,
    }
    by_cam_recent: dict[str, list[EventView]] = defaultdict(list)
    by_cam_earlier: dict[str, list[EventView]] = defaultdict(list)
    for e in recent:
        by_cam_recent[e.camera].append(e)
    for e in earlier:
        by_cam_earlier[e.camera].append(e)
    out["cameras"] = {
        cam: {
            "earlier": len(by_cam_earlier.get(cam, [])),
            "recent": len(by_cam_recent.get(cam, [])),
        }
        for cam in sorted(set(by_cam_recent) | set(by_cam_earlier))
    }
    if len(recent) < cfg["min_events"] or len(earlier) < cfg["min_events"]:
        return out
    a, b = _mix(earlier), _mix(recent)
    recent_mix = Counter({cam: len(v) for cam, v in by_cam_recent.items()})
    new = sorted(c for c in recent_mix if c not in by_cam_earlier)
    parts = {c: _mix(by_cam_earlier[c]) for c in by_cam_earlier} | {
        c: _mix(by_cam_recent[c]) for c in new
    }
    expected = {k: _weighted(parts, recent_mix, k) for k in ("labels", "confidence")}
    out["comparison"] = {
        "earlier": _jsonable_mix(a),
        "recent": _jsonable_mix(b),
        "new_cameras": new,
        "label_psi": {
            "total": psi(a["labels"], b["labels"]),
            "mix": psi(a["labels"], expected["labels"]),
            "within": psi(expected["labels"], b["labels"]),
        },
        "confidence_psi": {
            "total": psi(a["confidence"], b["confidence"]),
            "mix": psi(a["confidence"], expected["confidence"]),
            "within": psi(expected["confidence"], b["confidence"]),
        },
    }
    return out


def behavior(views: list[EventView], cfg: dict[str, Any], now: datetime) -> dict[str, Any]:
    """What the model decided, globally and per camera; each camera's latest batch
    against its earlier batches; and time periods with the camera mix separated."""
    by_camera: dict[str, list[EventView]] = defaultdict(list)
    for v in views:
        by_camera[v.camera].append(v)
    cameras = {}
    for cam, evs in sorted(by_camera.items()):
        entry: dict[str, Any] = _jsonable_mix(_mix(evs))
        batches = sorted({(e.batch_created_at, e.batch_id) for e in evs})
        entry["batches"] = len(batches)
        entry["latest_batch"] = None
        if batches:
            _, latest_id = batches[-1]
            latest = [e for e in evs if e.batch_id == latest_id]
            before = [e for e in evs if e.batch_id != latest_id]
            q_latest = _window(latest)
            q_before = _window(before) if before else None
            block: dict[str, Any] = {
                "batch_id": latest_id,
                "latest": _jsonable_mix(_mix(latest)),
                "comparison": None,
            }
            if len(latest) >= cfg["min_batch_events"] and len(before) >= cfg["min_events"]:
                cmp = _compare(_mix(before), _mix(latest))
                assert q_before is not None
                cmp["night_share_change"] = (
                    q_latest["night_share"] - q_before["night_share"]
                    if q_latest["night_share"] is not None and q_before["night_share"] is not None
                    else None
                )
                cmp["blur_change"] = (
                    (q_latest["median_blur"] - q_before["median_blur"]) / q_before["median_blur"]
                    if q_before["median_blur"] and q_latest["median_blur"] is not None
                    else None
                )
                cmp["release_changed"] = q_latest["releases"] != q_before["releases"]
                cmp["earlier_events"] = len(before)
                block["comparison"] = cmp
            entry["latest_batch"] = block
        cameras[cam] = entry
    return {
        "global": _jsonable_mix(_mix(views)),
        "cameras": cameras,
        "periods": periods(views, cfg, now),
    }


def audits(views: list[EventView], cfg: dict[str, Any]) -> dict[str, Any]:
    """Errors of automatic decisions found in the random audit sample.

    Only audit-sampled events count: other reviews of automatic events were
    chosen by people (browsing, recovering) and would bias the rate."""

    def block(evs: list[EventView]) -> dict[str, Any]:
        automatic = [e for e in evs if e.disposition in AUTOMATIC]
        sampled = [e for e in automatic if e.audit_selected]
        reviewed = [e for e in sampled if e.review_outcome]
        judged = [e for e in reviewed if e.review_outcome != "unresolved"]
        empties = [e for e in judged if e.disposition == "likely_empty"]
        species = [e for e in judged if e.disposition == "species_identified"]
        false_empty = sum(e.reviewed_label != EMPTY_CLASS for e in empties)
        wrong_species = sum(e.reviewed_label != e.label for e in species)
        if not automatic:
            quality = "not applicable: no automatic decisions"
        elif not judged:
            quality = "unknown: no audit labels"
        else:
            quality = "measured"
        return {
            "automatic_events": len(automatic),
            "audit_sampled": len(sampled),
            "audit_reviewed": len(reviewed),
            "audit_pending": len(sampled) - len(reviewed),
            "audit_unresolved": len(reviewed) - len(judged),
            "filtered_audited": len(empties),
            "false_empty": false_empty,
            "false_empty_rate": false_empty / len(empties) if empties else None,
            "false_empty_ci95": wilson(false_empty, len(empties)) if empties else None,
            "auto_labeled_audited": len(species),
            "species_errors": wrong_species,
            "species_error_rate": wrong_species / len(species) if species else None,
            "species_error_ci95": wilson(wrong_species, len(species)) if species else None,
            "observed_quality": quality,
        }

    cams: dict[str, list[EventView]] = defaultdict(list)
    for v in views:
        cams[v.camera].append(v)
    return {
        "global": block(views),
        "by_camera": {k: block(v) for k, v in sorted(cams.items())},
    }


# ------------------------------------------------------------------ alerts


def alerts(
    ops: dict[str, Any],
    sig: dict[str, Any],
    acc: dict[str, Any],
    cfg: dict[str, Any],
    *,
    work: dict[str, Any] | None = None,
    api: dict[str, Any] | None = None,
    store: dict[str, Any] | None = None,
    beh: dict[str, Any] | None = None,
    aud: dict[str, Any] | None = None,
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
    rate = ops["processing_error_rate"]
    if ops["frames_attempted"] and ops["frames_scored"] == 0:
        add(
            "critical",
            "operations",
            "inference failed for every frame",
            rate,
            o["max_processing_error_rate"],
        )
    elif rate is not None and rate > o["max_processing_error_rate"]:
        add(
            "warning",
            "operations",
            "frames failed inference",
            rate,
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

    if work is not None:
        busy = ops["queued_waiting"] + ops["jobs"].get("running", 0)
        if busy and not work["live"]:
            add("critical", "operations", "no worker is running while jobs wait", busy, 0)
        if len(work["died_in_window"]) > o["max_worker_deaths"]:
            add(
                "warning",
                "operations",
                f"a worker died without shutting down (last {work['window_hours']} h)",
                len(work["died_in_window"]),
                o["max_worker_deaths"],
            )
        if work["starts_in_window"] > o["max_worker_starts"]:
            add(
                "warning",
                "operations",
                f"workers restarted often (last {work['window_hours']} h)",
                work["starts_in_window"],
                o["max_worker_starts"],
            )
        for w in work["live"]:
            mb = w["rss_mb"] or w["peak_rss_mb"]
            if mb is not None and mb > o["max_worker_rss_mb"]:
                add(
                    "warning",
                    "operations",
                    f"worker {w['id']} uses a lot of memory (MB)",
                    mb,
                    o["max_worker_rss_mb"],
                )
    if api is not None:
        if api["server_errors"] and api["server_error_rate"] > o["max_api_server_error_rate"]:
            add(
                "warning",
                "operations",
                "the API returned server errors (share of responses)",
                round(api["server_error_rate"], 4),
                o["max_api_server_error_rate"],
            )
        slow = api["slowest_metadata_route"]
        if slow and slow["p95_ms"] > o["max_api_p95_ms"]:
            add(
                "warning",
                "operations",
                f"metadata API is slow: {slow['route']} p95 (ms)",
                slow["p95_ms"],
                o["max_api_p95_ms"],
            )
    if store is not None and store["original_gb"] > o["max_storage_gb"]:
        add(
            "warning",
            "operations",
            "stored originals are near the disk budget (GB)",
            store["original_gb"],
            o["max_storage_gb"],
        )

    if beh is not None:
        b = cfg["behavior"]
        checks = [
            ("label_psi", "suggested-label mix", b["max_label_psi"]),
            ("confidence_psi", "confidence distribution", b["max_confidence_psi"]),
            ("filtered_share_change", "share filtered as empty", b["max_share_change"]),
            ("auto_labeled_share_change", "share labeled automatically", b["max_share_change"]),
            ("needs_review_share_change", "share sent to review", b["max_share_change"]),
            ("unfamiliar_share_change", "share flagged unfamiliar", b["max_share_change"]),
            ("mean_confidence_change", "mean confidence", b["max_mean_confidence_change"]),
            ("night_share_change", "share of night frames", b["max_night_share_change"]),
            ("blur_change", "image sharpness", b["max_blur_change"]),
        ]
        for cam, entry in beh["cameras"].items():
            latest = entry["latest_batch"]
            c = latest["comparison"] if latest else None
            if c is None:
                continue
            why = " (the release also changed)" if c["release_changed"] else ""
            for key, what, limit in checks:
                value = c[key]
                if value is not None and abs(value) > limit:
                    add(
                        "info",
                        "behavior",
                        f"latest batch differs from earlier batches: {what}{why}",
                        round(value, 3),
                        limit,
                        cam,
                    )
        cmp = beh["periods"]["comparison"]
        if cmp is not None:
            for key, what, limit in (
                ("label_psi", "suggested-label mix", b["max_label_psi"]),
                ("confidence_psi", "confidence distribution", b["max_confidence_psi"]),
            ):
                v = cmp[key]
                if v["total"] <= limit:
                    continue
                if v["within"] > limit:
                    msg = f"{what} shifted within cameras (recent period vs the one before)"
                    add("info", "behavior", msg, round(v["within"], 3), limit)
                else:
                    msg = (
                        f"{what} shifted globally only because the camera mix changed; "
                        "no shift within cameras"
                    )
                    add("info", "behavior", msg, round(v["total"], 3), limit)

    if aud is not None:
        u = cfg["audits"]
        for cam, blk in aud["by_camera"].items():
            if blk["false_empty"]:
                add(
                    "warning",
                    "audit",
                    "an audit found an animal in an automatically filtered event",
                    blk["false_empty"],
                    0,
                    cam,
                )
            ci = blk["false_empty_ci95"]
            if ci is not None and ci[0] > u["max_false_empty_rate"]:
                add(
                    "critical",
                    "audit",
                    "audited false-empty rate is above the limit (lower 95% bound)",
                    round(ci[0], 3),
                    u["max_false_empty_rate"],
                    cam,
                )
            ci = blk["species_error_ci95"]
            if ci is not None and ci[0] > u["max_species_error_rate"]:
                add(
                    "warning",
                    "audit",
                    "audited species-error rate is above the limit (lower 95% bound)",
                    round(ci[0], 3),
                    u["max_species_error_rate"],
                    cam,
                )
            if blk["observed_quality"].startswith("unknown") and blk["automatic_events"]:
                add(
                    "info",
                    "audit",
                    "automation runs here with no audit labels yet: quality unknown",
                    blk["automatic_events"],
                    0,
                    cam,
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
    session: Session,
    lease_seconds: int,
    cfg: dict[str, Any],
    now: datetime | None = None,
    api: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    ops = operations(session, lease_seconds, cfg["operations"], now)
    ops["workers"] = work = workers(session, lease_seconds, cfg["operations"], now)
    ops["storage"] = store = storage(session, cfg["operations"], now)
    ops["batches"] = batch_costs(session)
    ops["api"] = api_data = api_health(api or {}, cfg["operations"])
    views = event_views(session)
    sig = signals(views, cfg["drift"])
    acc = accuracy(views, cfg["accuracy"])
    beh = behavior(views, cfg["behavior"], now)
    aud = audits(views, cfg["audits"])
    return {
        "generated_at": now.isoformat(),
        "operations": ops,
        "behavior": beh,
        "signals": sig,
        "audits": aud,
        "accuracy": acc,
        "alerts": alerts(
            ops, sig, acc, cfg, work=work, api=api_data, store=store, beh=beh, aud=aud
        ),
        "notes": {
            "signals": (
                "Label-free: can show that inputs or model behaviour changed, "
                "not whether accuracy did."
            ),
            "accuracy": "From human reviews only; unreviewed events have unknown accuracy.",
            "audits": (
                "Random audit sample of automatic decisions: the unbiased error measure. "
                "No audit labels means unknown observed quality, not zero errors."
            ),
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
