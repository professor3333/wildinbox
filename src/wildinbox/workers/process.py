"""Batch processing: validate images -> predict -> group events -> decide.

Reliability rules:

- A job is **claimed** under a lease (a token plus a heartbeat). Every commit of
  progress renews the heartbeat in the same transaction and only succeeds while
  the worker still holds the lease, so a worker that was presumed dead cannot
  write after another has taken over.
- Images are scored in **bounded chunks**; each chunk is committed, so a
  restarted job skips what is already stored.
- Every write is keyed by a unique constraint (image+release for predictions,
  batch+group key for events, event+release+policy for decisions), so repeating
  work updates rows instead of duplicating them.
- Events and decisions are written in **one final transaction**, after every
  image has a prediction or a recorded failure, so no event is finalized early.
- A failed attempt is retried with exponential backoff up to the job's
  `max_attempts`, then fails terminally. A worker that dies mid-job leaves a
  stale lease that `recover_stale` turns back into a queued job.
- A job is pinned to the release it was created with; retries never switch.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import socket
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from io import BytesIO
from typing import Protocol

from PIL import Image as PILImage
from sqlalchemy import or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, sessionmaker

from wildinbox.datasets.grouping import (
    DEFAULT_GAP_SECONDS,
    SEQUENCE_RULE,
    GroupableImage,
    group_by_sequence,
    group_by_time_gap,
    time_gap_rule_id,
)
from wildinbox.inference.serving import calibrated, scorer_for
from wildinbox.policy import review_all
from wildinbox.policy.conservative import Frame, PolicyConfig, decide
from wildinbox.schemas import Disposition, FrameStatus, ReviewReason
from wildinbox.settings import Settings
from wildinbox.storage.models import Batch, Decision, Event, Image, Job, ModelRelease, Prediction
from wildinbox.storage.objects import ObjectStore

log = logging.getLogger(__name__)

_EXIF_IFD = 0x8769
_DATETIME_ORIGINAL = 0x9003
_DATETIME = 0x0132


class LeaseLost(RuntimeError):
    """Another worker owns the job now; stop without writing."""


class Dispatcher(Protocol):
    def enqueue(self, job_id: uuid.UUID, attempt: int = 1) -> None: ...


# Unique per process start: a restarted container keeps its hostname and often
# its PID (1), and RQ refuses a name that a killed worker still holds in Redis.
_WORKER_ID = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


def worker_id() -> str:
    return _WORKER_ID


def _now() -> datetime:
    return datetime.now(UTC)


# ------------------------------------------------------------------ validation


@dataclass(frozen=True)
class Inspection:
    width: int | None = None
    height: int | None = None
    captured_at: datetime | None = None
    error: str | None = None


def inspect(data: bytes) -> Inspection:
    """Fully decode an image and read its EXIF capture time."""
    try:
        with PILImage.open(BytesIO(data)) as img:
            img.load()
            if img.format not in ("JPEG", "PNG"):
                return Inspection(error=f"unsupported format {img.format}")
            exif = img.getexif()
            raw = exif.get_ifd(_EXIF_IFD).get(_DATETIME_ORIGINAL) or exif.get(_DATETIME)
            captured = None
            if isinstance(raw, str):
                try:
                    captured = datetime.strptime(raw.strip(), "%Y:%m:%d %H:%M:%S")
                except ValueError:
                    captured = None
            return Inspection(img.width, img.height, captured)
    except Exception as e:
        message = re.sub(r"<[^<>]* at 0x[0-9a-f]+>", "<file>", str(e))
        return Inspection(error=f"unreadable image: {type(e).__name__}: {message}")


# ----------------------------------------------------------------------- lease


def claim(
    factory: sessionmaker[Session], job_id: uuid.UUID, worker: str, settings: Settings
) -> uuid.UUID | None:
    """Take the job's lease if it is due and nobody live holds it. Returns the
    lease token, or None if the job is finished, owned, not yet due, or out of
    attempts (in which case it is failed terminally here)."""
    with factory() as s:
        job = s.get(Job, job_id, with_for_update=True)
        if job is None:
            raise LookupError(f"job {job_id} not found")
        now = _now()
        if job.status in ("succeeded", "failed"):
            return None
        live = timedelta(seconds=settings.lease_seconds)
        if job.status == "running" and job.heartbeat_at and now - job.heartbeat_at < live:
            return None  # a live worker holds it
        if job.status == "queued" and job.next_attempt_at is not None and job.next_attempt_at > now:
            return None  # waiting for its retry time
        if job.attempts >= job.max_attempts:
            _fail_terminally(job, f"gave up after {job.attempts} attempts: {job.error or ''}")
            s.commit()
            return None
        token = uuid.uuid4()
        job.status, job.attempts = "running", job.attempts + 1
        job.lease_token, job.worker_id, job.heartbeat_at = token, worker, now
        job.started_at, job.next_attempt_at = now, None
        job.batch.status = "processing"
        s.commit()
        return token


def renew(session: Session, job_id: uuid.UUID, token: uuid.UUID) -> None:
    """Renew the lease inside the current transaction, or raise LeaseLost."""
    result = session.execute(
        update(Job)
        .where(Job.id == job_id, Job.lease_token == token, Job.status == "running")
        .values(heartbeat_at=_now())
    )
    if result.rowcount != 1:  # type: ignore[attr-defined]
        raise LeaseLost(f"job {job_id} lease {token} is no longer held")


def _fail_terminally(job: Job, error: str) -> None:
    job.status, job.error = "failed", error
    job.finished_at, job.lease_token, job.next_attempt_at = _now(), None, None
    job.batch.status, job.batch.error = "failed", error


def backoff(settings: Settings, attempts: int) -> timedelta:
    seconds = settings.retry_backoff_seconds * 2 ** max(0, attempts - 1)
    return timedelta(seconds=min(seconds, settings.retry_backoff_max_seconds))


# --------------------------------------------------------------------- scoring


def _chunks(items: Sequence[Image], n: int) -> list[Sequence[Image]]:
    return [items[i : i + n] for i in range(0, len(items), n)]


def _score_images(
    session: Session,
    store: ObjectStore,
    batch: Batch,
    release: ModelRelease,
    job_id: uuid.UUID,
    token: uuid.UUID,
    settings: Settings,
) -> None:
    scorer = scorer_for(release, store, settings.inference_device)
    done = set(
        session.scalars(
            select(Prediction.image_id)
            .join(Image)
            .where(Image.batch_id == batch.id, Prediction.model_release_id == release.id)
        )
    )
    todo = [
        i
        for i in batch.images
        if i.validation_status in ("pending", "valid")
        and i.id not in done
        and i.processing_error is None
    ]
    for chunk in _chunks(todo, settings.inference_chunk):
        ready: list[tuple[Image, bytes]] = []
        for img in chunk:
            assert img.storage_key and img.sha256
            data = store.get(img.storage_key)
            found = inspect(data)
            if found.error:
                # Explicit per-file error; the rest of the batch continues.
                img.validation_status, img.validation_error = "invalid", found.error
                continue
            img.validation_status = "valid"
            img.width, img.height = found.width, found.height
            if img.captured_at is None and found.captured_at is not None:
                img.captured_at = found.captured_at
            ready.append((img, data))
        for img, raw in _predict(scorer, ready):
            if raw is None:
                continue
            label = max(raw, key=raw.__getitem__)
            session.execute(
                insert(Prediction)
                .values(
                    id=uuid.uuid4(),
                    image_id=img.id,
                    model_release_id=release.id,
                    class_probabilities=raw,
                    calibrated_probabilities=calibrated(release, raw),
                    suggested_label=label,
                    confidence=raw[label],
                )
                .on_conflict_do_nothing(index_elements=["image_id", "model_release_id"])
            )
        renew(session, job_id, token)
        session.commit()  # progress survives a worker restart


def _predict(
    scorer: object, ready: list[tuple[Image, bytes]]
) -> list[tuple[Image, dict[str, float] | None]]:
    """Score a chunk together; if that fails, score images one by one so a
    single bad input becomes a failed frame instead of failing the batch."""
    if not ready:
        return []
    imgs, datas = [i for i, _ in ready], [d for _, d in ready]
    try:
        raws = scorer.score(datas, [i.sha256 or "" for i in imgs])  # type: ignore[attr-defined]
        return list(zip(imgs, raws, strict=True))
    except Exception:
        log.warning("chunk scoring failed; retrying images one at a time", exc_info=True)
    out: list[tuple[Image, dict[str, float] | None]] = []
    for img, data in ready:
        try:
            out.append((img, scorer.score([data], [img.sha256 or ""])[0]))  # type: ignore[attr-defined]
        except Exception as e:
            img.processing_error = f"inference failed: {type(e).__name__}: {e}"
            out.append((img, None))
    return out


# ---------------------------------------------------------------------- events


def _group_key(rule: str, image_ids: list[str]) -> str:
    return hashlib.sha256(f"{rule}|{','.join(sorted(image_ids))}".encode()).hexdigest()[:32]


def _group_events(session: Session, batch: Batch) -> list[Event]:
    valid = [i for i in batch.images if i.validation_status == "valid"]
    by_id = {str(i.id): i for i in valid}
    with_seq = [
        GroupableImage(str(i.id), i.camera_id, i.captured_at, i.sequence_id)
        for i in valid
        if i.sequence_id
    ]
    without = [
        GroupableImage(str(i.id), i.camera_id, i.captured_at) for i in valid if not i.sequence_id
    ]
    groups = [(SEQUENCE_RULE, g) for g in group_by_sequence(with_seq)]
    groups += [
        (time_gap_rule_id(DEFAULT_GAP_SECONDS), g)
        for g in group_by_time_gap(without, DEFAULT_GAP_SECONDS)
    ]

    keys = []
    for rule, ids in groups:
        members = [by_id[i] for i in ids]
        times = [m.captured_at for m in members if m.captured_at]
        key = _group_key(rule, ids)
        keys.append(key)
        session.execute(
            insert(Event)
            .values(
                id=uuid.uuid4(),
                batch_id=batch.id,
                camera_id=members[0].camera_id,
                grouping_rule=rule,
                group_key=key,
                start_at=min(times, default=None),
                end_at=max(times, default=None),
            )
            .on_conflict_do_nothing(index_elements=["batch_id", "group_key"])
        )
    events = {
        e.group_key: e for e in session.scalars(select(Event).where(Event.batch_id == batch.id))
    }
    for stale in set(events) - set(keys):  # from an earlier attempt with different inputs
        session.delete(events.pop(stale))
    for (_, ids), key in zip(groups, keys, strict=True):
        for i in ids:
            by_id[i].event_id = events[key].id
    session.flush()
    return list(events.values())


def _frames(event: Event, preds: dict[uuid.UUID, Prediction], release: ModelRelease) -> list[Frame]:
    frames = []
    for img in event.images:
        p = preds.get(img.id)
        if p is not None:
            probs = p.calibrated_probabilities or calibrated(release, p.class_probabilities)
            frames.append(Frame(probs, FrameStatus.COMPLETED))
        elif img.processing_error:
            frames.append(Frame(None, FrameStatus.FAILED))
        else:
            frames.append(Frame(None, FrameStatus.PENDING))
    return frames


def _decide(session: Session, events: list[Event], release: ModelRelease) -> None:
    for event in events:
        rows = session.scalars(
            select(Prediction)
            .join(Image, Image.id == Prediction.image_id)
            .where(Image.event_id == event.id, Prediction.model_release_id == release.id)
        ).all()
        preds = {p.image_id: p for p in rows}
        for p in rows:
            p.event_id = event.id
        values, policy_version = _decision(event, preds, release)
        session.execute(
            insert(Decision)
            .values(
                id=uuid.uuid4(),
                event_id=event.id,
                model_release_id=release.id,
                policy_version=policy_version,
                **values,
            )
            .on_conflict_do_update(
                index_elements=["event_id", "model_release_id", "policy_version"], set_=values
            )
        )


def _decision(
    event: Event, preds: dict[uuid.UUID, Prediction], release: ModelRelease
) -> tuple[dict[str, object], str]:
    frames = _frames(event, preds, release)
    if release.kind == "test_predictor":
        done = [p for p in (preds.get(i.id) for i in event.images) if p is not None]
        if not done:
            return _values(
                Disposition.NEEDS_REVIEW, None, None, [ReviewReason.PROCESSING_FAILURE]
            ), (review_all.POLICY_VERSION)
        d = review_all.decide(
            event.id,
            [p.image_id for p in done],
            [p.class_probabilities for p in done],
            model_version=release.id,
            preprocessing_version=release.preprocessing_version,
        )
        reasons = list(d.reasons)
        if len(done) < len(frames):
            reasons.insert(0, ReviewReason.PROCESSING_FAILURE)
        return _values(d.disposition, d.suggested_label, d.confidence, reasons), d.policy_version
    c = release.policy["config"]
    cfg = PolicyConfig(
        c["empty_threshold"],
        c["species_threshold"],
        c["auto_filter_enabled"],
        c["auto_accept_enabled"],
        tuple(c["accept_species"]) if c.get("accept_species") is not None else None,
    )
    o = decide(frames, cfg)
    return _values(o.disposition, o.label, o.confidence, o.reasons), release.policy_version


def _values(
    disposition: Disposition,
    label: str | None,
    confidence: float | None,
    reasons: Sequence[ReviewReason],
) -> dict[str, object]:
    return {
        "disposition": disposition.value,
        "suggested_label": label,
        "confidence": None if confidence is None else round(confidence, 6),
        "reasons": [r.value for r in reasons],
    }


# ------------------------------------------------------------------------ jobs


def process_batch(
    factory: sessionmaker[Session],
    store: ObjectStore,
    job_id: uuid.UUID,
    settings: Settings | None = None,
    worker: str | None = None,
) -> str:
    """Run one attempt of a job. Returns the job's status afterwards (or
    "skipped" when it was not claimable)."""
    settings = settings or Settings()
    token = claim(factory, job_id, worker or worker_id(), settings)
    if token is None:
        log.info("job %s not claimable (finished, owned, or not yet due)", job_id)
        return "skipped"
    with factory() as session:
        try:
            job = session.get(Job, job_id)
            assert job is not None
            batch, release = job.batch, job.release  # the release pinned at creation
            _score_images(session, store, batch, release, job_id, token, settings)
            events = _group_events(session, batch)
            _decide(session, events, release)
            # Check (and lock) the lease BEFORE touching the job row, so a worker
            # that lost its lease cannot mark someone else's job finished.
            renew(session, job_id, token)
            has_errors = any(
                i.validation_status == "invalid" or i.processing_error for i in batch.images
            )
            batch.status = "completed_with_errors" if has_errors else "completed"
            batch.completed_at = job.finished_at = _now()
            job.status, job.error, job.lease_token = "succeeded", None, None
            session.commit()  # events, decisions, and completion land together
            return "succeeded"
        except LeaseLost:
            session.rollback()
            log.warning("job %s: lease lost; another worker owns it", job_id)
            return "skipped"
        except Exception as e:
            session.rollback()
            return _record_failure(session, job_id, token, e, settings)


def _record_failure(
    session: Session, job_id: uuid.UUID, token: uuid.UUID, e: Exception, settings: Settings
) -> str:
    job = session.get(Job, job_id, with_for_update=True)
    assert job is not None
    if job.lease_token != token:
        session.rollback()
        return "skipped"
    error = f"{type(e).__name__}: {e}"
    log.error("job %s attempt %d failed: %s", job_id, job.attempts, error)
    if job.attempts >= job.max_attempts:
        _fail_terminally(job, error)
    else:
        job.status, job.error, job.lease_token = "queued", error, None
        job.next_attempt_at = _now() + backoff(settings, job.attempts)
    session.commit()
    return job.status


def recover_stale(
    factory: sessionmaker[Session], dispatcher: Dispatcher, settings: Settings
) -> list[uuid.UUID]:
    """Requeue jobs whose worker stopped renewing its lease, and dispatch every
    queued job that is due. Safe to run from several workers at once: claiming
    decides who processes a job."""
    now = _now()
    stale_before = now - timedelta(seconds=settings.lease_seconds)
    due: list[tuple[uuid.UUID, int]] = []
    with factory() as s:
        stale = s.scalars(
            select(Job)
            .where(Job.status == "running", Job.heartbeat_at < stale_before)
            .with_for_update(skip_locked=True)
        ).all()
        for job in stale:
            log.warning("job %s: worker %s stopped responding; recovering", job.id, job.worker_id)
            error = f"worker {job.worker_id} stopped responding (lease expired)"
            if job.attempts >= job.max_attempts:
                _fail_terminally(job, f"gave up after {job.attempts} attempts: {error}")
            else:
                job.status, job.error, job.lease_token = "queued", error, None
                job.next_attempt_at = now
        s.commit()
        for job in s.scalars(
            select(Job).where(
                Job.status == "queued",
                or_(Job.next_attempt_at.is_(None), Job.next_attempt_at <= now),
            )
        ):
            due.append((job.id, job.attempts + 1))
    for job_id, attempt in due:
        try:
            dispatcher.enqueue(job_id, attempt)
        except Exception:
            log.exception("could not dispatch job %s; will retry", job_id)
    return [j for j, _ in due]
