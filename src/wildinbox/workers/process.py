"""Batch processing: validate images -> predict -> group events -> decide.

Every write is keyed by a unique constraint (image+release for predictions,
batch+group key for events, event+release+policy for decisions), so running a
job again updates rows instead of duplicating them. Progress is committed in
chunks, so a restarted job skips work that is already stored.
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO

from PIL import Image as PILImage
from sqlalchemy import select
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
from wildinbox.inference.plumbing import PlumbingPredictor
from wildinbox.policy import review_all
from wildinbox.storage.models import Batch, Decision, Event, Image, Job, ModelRelease, Prediction
from wildinbox.storage.objects import ObjectStore

log = logging.getLogger(__name__)

CHUNK = 50
_EXIF_IFD = 0x8769
_DATETIME_ORIGINAL = 0x9003
_DATETIME = 0x0132


@dataclass(frozen=True)
class Inspection:
    width: int | None = None
    height: int | None = None
    captured_at: datetime | None = None
    image: PILImage.Image | None = None
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
            return Inspection(img.width, img.height, captured, img.convert("RGB"))
    except Exception as e:
        message = re.sub(r"<[^<>]* at 0x[0-9a-f]+>", "<file>", str(e))
        return Inspection(error=f"unreadable image: {type(e).__name__}: {message}")


def _predictor(release: ModelRelease) -> PlumbingPredictor:
    if release.kind == "test_predictor":
        return PlumbingPredictor(list(release.class_names))
    raise NotImplementedError(f"release kind {release.kind!r} is not supported yet")


def _score_images(
    session: Session, store: ObjectStore, batch: Batch, release: ModelRelease
) -> None:
    predictor = _predictor(release)
    done = set(
        session.scalars(
            select(Prediction.image_id)
            .join(Image)
            .where(Image.batch_id == batch.id, Prediction.model_release_id == release.id)
        )
    )
    todo = [
        i for i in batch.images if i.validation_status in ("pending", "valid") and i.id not in done
    ]
    for n, img in enumerate(todo, start=1):
        assert img.storage_key and img.sha256
        found = inspect(store.get(img.storage_key))
        if found.error or found.image is None:
            img.validation_status, img.validation_error = "invalid", found.error
        else:
            img.validation_status = "valid"
            img.width, img.height = found.width, found.height
            if img.captured_at is None and found.captured_at is not None:
                img.captured_at = found.captured_at
            probs = predictor.predict(found.image, img.sha256)
            label = max(probs, key=probs.__getitem__)
            session.execute(
                insert(Prediction)
                .values(
                    id=uuid.uuid4(),
                    image_id=img.id,
                    model_release_id=release.id,
                    class_probabilities=probs,
                    suggested_label=label,
                    confidence=probs[label],
                )
                .on_conflict_do_nothing(index_elements=["image_id", "model_release_id"])
            )
        if n % CHUNK == 0:
            session.commit()  # progress survives a worker restart
    session.commit()


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


def _decide(session: Session, events: list[Event], release: ModelRelease) -> None:
    for event in events:
        preds = session.scalars(
            select(Prediction)
            .join(Image, Image.id == Prediction.image_id)
            .where(Image.event_id == event.id, Prediction.model_release_id == release.id)
            .order_by(Image.position)
        ).all()
        for p in preds:
            p.event_id = event.id
        decision = review_all.decide(
            event.id,
            [p.image_id for p in preds],
            [p.class_probabilities for p in preds],
            model_version=release.id,
            preprocessing_version=release.preprocessing_version,
        )
        values = {
            "disposition": decision.disposition.value,
            "suggested_label": decision.suggested_label,
            "confidence": decision.confidence,
            "reasons": [r.value for r in decision.reasons],
        }
        session.execute(
            insert(Decision)
            .values(
                id=uuid.uuid4(),
                event_id=event.id,
                model_release_id=release.id,
                policy_version=decision.policy_version,
                **values,
            )
            .on_conflict_do_update(
                index_elements=["event_id", "model_release_id", "policy_version"], set_=values
            )
        )


def process_batch(factory: sessionmaker[Session], store: ObjectStore, job_id: uuid.UUID) -> None:
    with factory() as session:
        job = session.get(Job, job_id, with_for_update=True)
        if job is None:
            raise LookupError(f"job {job_id} not found")
        if job.status == "succeeded":
            log.info("job %s already succeeded; nothing to do", job_id)
            return
        job.status, job.attempts, job.error = "running", job.attempts + 1, None
        job.started_at = datetime.now(UTC)
        job.batch.status = "processing"
        session.commit()

        try:
            batch, release = job.batch, job.release
            _score_images(session, store, batch, release)
            events = _group_events(session, batch)
            _decide(session, events, release)
            has_errors = any(i.validation_status == "invalid" for i in batch.images)
            batch.status = "completed_with_errors" if has_errors else "completed"
            batch.completed_at = job.finished_at = datetime.now(UTC)
            job.status = "succeeded"
            session.commit()
        except Exception as e:
            session.rollback()
            job = session.get(Job, job_id)
            assert job is not None
            job.status, job.error = "failed", f"{type(e).__name__}: {e}"
            job.finished_at = datetime.now(UTC)
            job.batch.status, job.batch.error = "failed", job.error
            session.commit()
            raise
