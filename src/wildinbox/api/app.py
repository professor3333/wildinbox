"""HTTP API: upload batches, follow jobs, read events, record reviews."""

from __future__ import annotations

import logging
import uuid
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from wildinbox.api import views
from wildinbox.api.uploads import (
    UploadedFile,
    UploadError,
    check_file,
    manifest,
    parse_metadata,
    request_fingerprint,
)
from wildinbox.config import load_config
from wildinbox.inference.releases import ensure_test_release, release_notice
from wildinbox.schemas import Review as ReviewContract
from wildinbox.schemas import ReviewOutcome
from wildinbox.settings import Settings
from wildinbox.storage.db import session_factory
from wildinbox.storage.models import Batch, Decision, Event, Image, Job, ModelRelease, Review
from wildinbox.storage.objects import (
    ObjectNotFoundError,
    ObjectStore,
    original_key,
    store_from_settings,
)
from wildinbox.workers.dispatch import Dispatcher, dispatcher_from_settings

log = logging.getLogger(__name__)
MULTIPART_OVERHEAD = 1024 * 1024


class ApiError(Exception):
    def __init__(self, status: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status, self.code, self.detail = status, code, detail


def _error(status: int, code: str, detail: str) -> JSONResponse:
    return JSONResponse({"error": code, "detail": detail}, status_code=status)


# ------------------------------------------------------------------ serializers


def _release(release: ModelRelease) -> dict[str, Any]:
    return {
        "id": release.id,
        "kind": release.kind,
        "is_test": release.is_test,
        "policy_version": release.policy_version,
        "notice": release_notice(release),
    }


def batch_summary(session: Session, batch: Batch) -> dict[str, Any]:
    counts = Counter(i.validation_status for i in batch.images)
    events = session.scalars(select(Event.id).where(Event.batch_id == batch.id)).all()
    job = batch.jobs[0] if batch.jobs else None
    return {
        "id": str(batch.id),
        "workspace": batch.workspace,
        "status": batch.status,
        "error": batch.error,
        "created_at": batch.created_at.isoformat(),
        "completed_at": batch.completed_at.isoformat() if batch.completed_at else None,
        "counts": {
            "images": len(batch.images),
            **{s: counts.get(s, 0) for s in ("pending", "valid", "invalid", "duplicate")},
            "events": len(events),
        },
        "job": _job(job) if job else None,
        "release": _release(job.release) if job else None,
        "links": {
            "self": f"/batches/{batch.id}",
            "images": f"/batches/{batch.id}/images",
            "events": f"/events?batch_id={batch.id}",
            "view": f"/batches/{batch.id}/view",
        },
    }


def _job(job: Job) -> dict[str, Any]:
    return {
        "id": str(job.id),
        "batch_id": str(job.batch_id),
        "kind": job.kind,
        "status": job.status,
        "attempts": job.attempts,
        "max_attempts": job.max_attempts,
        "error": job.error,
        "model_release_id": job.model_release_id,
        "created_at": job.created_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
    }


def image_row(img: Image) -> dict[str, Any]:
    return {
        "id": str(img.id),
        "position": img.position,
        "filename": img.original_filename,
        "size_bytes": img.size_bytes,
        "sha256": img.sha256,
        "content_type": img.content_type,
        "width": img.width,
        "height": img.height,
        "camera_id": img.camera_id,
        "captured_at": img.captured_at.isoformat() if img.captured_at else None,
        "sequence_id": img.sequence_id,
        "validation_status": img.validation_status,
        "validation_error": img.validation_error,
        "duplicate_of": str(img.duplicate_of) if img.duplicate_of else None,
        "event_id": str(img.event_id) if img.event_id else None,
        "original_url": f"/images/{img.id}/original" if img.storage_key else None,
    }


def event_row(session: Session, event: Event, detail: bool = False) -> dict[str, Any]:
    decision = max(event.decisions, key=lambda d: d.created_at, default=None)
    release = session.get(ModelRelease, decision.model_release_id) if decision else None
    latest = event.reviews[-1] if event.reviews else None
    row: dict[str, Any] = {
        "id": str(event.id),
        "batch_id": str(event.batch_id),
        "camera_id": event.camera_id,
        "grouping_rule": event.grouping_rule,
        "start_at": event.start_at.isoformat() if event.start_at else None,
        "end_at": event.end_at.isoformat() if event.end_at else None,
        "image_ids": [str(i.id) for i in event.images],
        "decision": None
        if decision is None
        else {
            "disposition": decision.disposition,
            "suggested_label": decision.suggested_label,
            "confidence": decision.confidence,
            "reasons": decision.reasons,
            "policy_version": decision.policy_version,
            "model_release_id": decision.model_release_id,
        },
        "release": _release(release) if release else None,
        "latest_review": _review(latest) if latest else None,
    }
    if detail:
        from wildinbox.storage.models import Prediction

        preds = {
            p.image_id: p
            for p in session.scalars(select(Prediction).where(Prediction.event_id == event.id))
        }
        row["images"] = [
            {
                **image_row(i),
                "prediction": None
                if i.id not in preds
                else {
                    "suggested_label": preds[i.id].suggested_label,
                    "confidence": preds[i.id].confidence,
                    "class_probabilities": preds[i.id].class_probabilities,
                    "model_release_id": preds[i.id].model_release_id,
                },
            }
            for i in event.images
        ]
        row["reviews"] = [_review(r) for r in event.reviews]
    return row


def _review(r: Review) -> dict[str, Any]:
    return {
        "id": str(r.id),
        "reviewer": r.reviewer,
        "outcome": r.outcome,
        "suggested_label": r.suggested_label,
        "confirmed_label": r.confirmed_label,
        "note": r.note,
        "previous_review_id": str(r.previous_review_id) if r.previous_review_id else None,
        "created_at": r.created_at.isoformat(),
    }


class ReviewIn(BaseModel):
    reviewer: str = Field(min_length=1, max_length=200)
    outcome: ReviewOutcome
    confirmed_label: str | None = Field(default=None, max_length=100)
    note: str | None = Field(default=None, max_length=2000)


# ------------------------------------------------------------------------ app


def create_app(
    settings: Settings | None = None,
    store: ObjectStore | None = None,
    dispatcher: Dispatcher | None = None,
) -> FastAPI:
    settings = settings or Settings()
    cfg = load_config(settings.config_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.store = store or store_from_settings(settings)
        app.state.dispatcher = dispatcher or dispatcher_from_settings(settings, app.state.store)
        app.state.sessions = session_factory(settings.database_url)
        with app.state.sessions() as s:
            ensure_test_release(s, cfg)
            s.commit()
        yield

    app = FastAPI(title="WildInbox", version="0.1.0", lifespan=lifespan)

    def sessions() -> sessionmaker[Session]:
        factory: sessionmaker[Session] = app.state.sessions
        return factory

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, e: ApiError) -> JSONResponse:
        return _error(e.status, e.code, e.detail)

    @app.exception_handler(UploadError)
    async def _upload_error(_: Request, e: UploadError) -> JSONResponse:
        return _error(e.status, e.code, e.detail)

    @app.get("/health")
    def health() -> dict[str, str]:
        with sessions()() as s:
            s.execute(select(1))
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return views.upload_page(settings)

    # ----------------------------------------------------------------- batches

    @app.post("/batches", status_code=202)
    async def create_batch(
        request: Request, idempotency_key: str | None = Header(default=None, max_length=200)
    ) -> JSONResponse:
        length = request.headers.get("content-length")
        if length and int(length) > settings.max_batch_bytes + MULTIPART_OVERHEAD:
            raise UploadError(
                413,
                "batch_too_large",
                f"request is {int(length)} bytes; the batch limit is "
                f"{settings.max_batch_bytes} bytes",
            )
        try:
            form = await request.form(max_files=settings.max_files_per_batch, max_fields=20)
        except StarletteHTTPException as e:
            if "Too many files" in str(e.detail):
                raise UploadError(
                    413,
                    "too_many_files",
                    f"a batch may contain at most {settings.max_files_per_batch} files",
                ) from None
            raise UploadError(400, "invalid_upload", str(e.detail)) from None
        uploads = [f for f in form.getlist("files") if isinstance(f, UploadFile)]
        if not uploads:
            raise UploadError(400, "no_files", "send one or more files in the 'files' field")
        raw_meta = form.get("metadata")
        metadata = parse_metadata(raw_meta if isinstance(raw_meta, str) else None)

        files: list[UploadedFile] = []
        total = 0
        for pos, up in enumerate(uploads):
            data = await up.read(settings.max_file_bytes + 1)
            total += len(data)
            if total > settings.max_batch_bytes:
                raise UploadError(
                    413,
                    "batch_too_large",
                    f"batch exceeds the {settings.max_batch_bytes} byte limit",
                )
            name = (up.filename or f"file-{pos}")[:500]
            files.append(
                check_file(
                    pos,
                    name,
                    data,
                    len(data) > settings.max_file_bytes,
                    settings,
                    metadata.for_file(name),
                )
            )
        await form.close()
        return await run_in_threadpool(_store_batch, files, metadata, idempotency_key)

    def _store_batch(
        files: list[UploadedFile], metadata: Any, idempotency_key: str | None
    ) -> JSONResponse:
        fingerprint = request_fingerprint(files, metadata)
        request_key = f"key:{idempotency_key}" if idempotency_key else f"content:{fingerprint}"
        with sessions()() as s:
            existing = _existing(s, request_key, fingerprint)
            if existing is not None:
                return existing
            for f in files:
                if f.accepted and f.data is not None and f.sha256 and f.content_type:
                    key = original_key(f.sha256)
                    if not app.state.store.exists(key):
                        app.state.store.put(key, f.data, f.content_type)

            batch = Batch(
                id=uuid.uuid4(),
                workspace=settings.workspace,
                request_key=request_key,
                status="queued",
                manifest=manifest(files, metadata, fingerprint, settings),
            )
            s.add(batch)
            prior = _workspace_images(s, {f.sha256 for f in files if f.sha256})
            seen: dict[str, uuid.UUID] = {}
            for f in files:
                img = Image(
                    id=uuid.uuid4(),
                    batch_id=batch.id,
                    position=f.position,
                    original_filename=f.filename,
                    size_bytes=f.size,
                    sha256=f.sha256,
                    content_type=f.content_type,
                    camera_id=f.metadata.camera_id,
                    captured_at=_naive(f.metadata.captured_at),
                    sequence_id=f.metadata.sequence_id,
                    user_metadata=f.metadata.model_dump(mode="json", exclude_none=True),
                )
                if not f.accepted:
                    img.validation_status = "invalid"
                    img.validation_error = f"{f.error_code}: {f.error}"
                elif f.sha256 in seen or f.sha256 in prior:
                    img.validation_status = "duplicate"
                    img.duplicate_of = seen.get(f.sha256) or prior[f.sha256 or ""]
                    img.storage_key = original_key(f.sha256 or "")
                else:
                    img.validation_status = "pending"
                    img.storage_key = original_key(f.sha256 or "")
                    seen[f.sha256 or ""] = img.id
                s.add(img)
            job = Job(
                id=uuid.uuid4(),
                batch_id=batch.id,
                kind="process_batch",
                status="queued",
                attempts=0,
                max_attempts=3,
                model_release_id=settings.active_release,
            )
            s.add(job)
            try:
                s.commit()
            except IntegrityError:
                s.rollback()  # a concurrent identical request won the race
                existing = _existing(s, request_key, fingerprint)
                if existing is not None:
                    return existing
                raise
            try:
                app.state.dispatcher.enqueue(job.id)
            except Exception:
                log.exception("could not dispatch job %s; it stays queued", job.id)
            s.expire_all()
            batch_row = s.get(Batch, batch.id)
            assert batch_row is not None
            return JSONResponse(batch_summary(s, batch_row), status_code=202)

    def _existing(s: Session, request_key: str, fingerprint: str) -> JSONResponse | None:
        batch = s.scalar(
            select(Batch).where(
                Batch.workspace == settings.workspace, Batch.request_key == request_key
            )
        )
        if batch is None:
            return None
        if batch.manifest.get("fingerprint") != fingerprint:
            raise ApiError(
                409,
                "idempotency_conflict",
                "this Idempotency-Key was already used for a different upload",
            )
        return JSONResponse({**batch_summary(s, batch), "duplicate_request": True}, status_code=200)

    def _workspace_images(s: Session, shas: set[str]) -> dict[str, uuid.UUID]:
        if not shas:
            return {}
        rows = s.execute(
            select(Image.sha256, Image.id)
            .join(Batch)
            .where(
                Batch.workspace == settings.workspace,
                Image.sha256.in_(shas),
                Image.validation_status.in_(("pending", "valid")),
            )
            .order_by(Image.created_at)
        ).all()
        out: dict[str, uuid.UUID] = {}
        for sha, iid in rows:
            out.setdefault(sha, iid)
        return out

    def _batch(s: Session, batch_id: uuid.UUID) -> Batch:
        batch = s.get(Batch, batch_id)
        if batch is None or batch.workspace != settings.workspace:
            raise ApiError(404, "not_found", f"batch {batch_id} not found")
        return batch

    @app.get("/batches/{batch_id}")
    def get_batch(batch_id: uuid.UUID) -> dict[str, Any]:
        with sessions()() as s:
            return batch_summary(s, _batch(s, batch_id))

    @app.get("/batches/{batch_id}/images")
    def get_batch_images(batch_id: uuid.UUID) -> dict[str, Any]:
        with sessions()() as s:
            return {"images": [image_row(i) for i in _batch(s, batch_id).images]}

    @app.get("/batches/{batch_id}/view", response_class=HTMLResponse)
    def view_batch(batch_id: uuid.UUID) -> str:
        with sessions()() as s:
            batch = _batch(s, batch_id)
            events = s.scalars(
                select(Event).where(Event.batch_id == batch.id).order_by(Event.start_at, Event.id)
            ).all()
            return views.batch_page(
                batch_summary(s, batch),
                [image_row(i) for i in batch.images],
                [event_row(s, e) for e in events],
            )

    @app.get("/jobs/{job_id}")
    def get_job(job_id: uuid.UUID) -> dict[str, Any]:
        with sessions()() as s:
            job = s.get(Job, job_id)
            if job is None or job.batch.workspace != settings.workspace:
                raise ApiError(404, "not_found", f"job {job_id} not found")
            return _job(job)

    # ------------------------------------------------------------------ events

    @app.get("/events")
    def list_events(
        batch_id: uuid.UUID | None = None,
        camera_id: str | None = None,
        disposition: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        limit = max(1, min(limit, 500))
        with sessions()() as s:
            q = select(Event).join(Batch).where(Batch.workspace == settings.workspace)
            if batch_id:
                q = q.where(Event.batch_id == batch_id)
            if camera_id:
                q = q.where(Event.camera_id == camera_id)
            if disposition:
                q = q.where(Event.decisions.any(Decision.disposition == disposition))
            events = s.scalars(q.order_by(Event.start_at, Event.id).limit(limit).offset(offset))
            return {"events": [event_row(s, e) for e in events], "limit": limit, "offset": offset}

    def _event(s: Session, event_id: uuid.UUID) -> Event:
        event = s.get(Event, event_id)
        if event is None or s.get(Batch, event.batch_id).workspace != settings.workspace:  # type: ignore[union-attr]
            raise ApiError(404, "not_found", f"event {event_id} not found")
        return event

    @app.get("/events/{event_id}")
    def get_event(event_id: uuid.UUID) -> dict[str, Any]:
        with sessions()() as s:
            return event_row(s, _event(s, event_id), detail=True)

    @app.post("/events/{event_id}/reviews", status_code=201)
    def create_review(event_id: uuid.UUID, body: ReviewIn) -> dict[str, Any]:
        with sessions()() as s:
            event = _event(s, event_id)
            decision = max(event.decisions, key=lambda d: d.created_at, default=None)
            suggested = decision.suggested_label if decision else None
            previous = event.reviews[-1] if event.reviews else None
            review_id = uuid.uuid4()
            try:
                ReviewContract(
                    review_id=review_id,
                    event_id=event.id,
                    outcome=body.outcome,
                    suggested_label=suggested,
                    confirmed_label=body.confirmed_label,
                    reviewer=body.reviewer,
                    reviewed_at=_utcnow(),
                    note=body.note,
                )
            except ValidationError as e:
                raise ApiError(
                    422, "invalid_review", "; ".join(err["msg"] for err in e.errors())
                ) from None
            review = Review(
                id=review_id,
                event_id=event.id,
                reviewer=body.reviewer,
                outcome=body.outcome.value,
                suggested_label=suggested,
                confirmed_label=body.confirmed_label,
                note=body.note,
                previous_review_id=previous.id if previous else None,
            )
            s.add(review)
            try:
                s.commit()
            except IntegrityError:
                raise ApiError(
                    409,
                    "review_conflict",
                    "another review was recorded concurrently; reload and retry",
                ) from None
            s.refresh(review)
            return _review(review)

    # ------------------------------------------------------------------ images

    @app.get("/images/{image_id}/original")
    def get_original(image_id: uuid.UUID) -> Response:
        with sessions()() as s:
            img = s.get(Image, image_id)
            if img is None or img.batch.workspace != settings.workspace or not img.storage_key:
                raise ApiError(404, "not_found", f"image {image_id} has no stored original")
            try:
                data = app.state.store.get(img.storage_key)
            except ObjectNotFoundError:
                raise ApiError(404, "not_found", "original missing from object storage") from None
            return Response(data, media_type=img.content_type or "application/octet-stream")

    return app


def _naive(ts: Any) -> Any:
    return ts.replace(tzinfo=None) if ts is not None and ts.tzinfo else ts


def _utcnow() -> Any:
    from datetime import UTC, datetime

    return datetime.now(UTC)
