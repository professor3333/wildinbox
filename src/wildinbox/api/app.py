"""HTTP API: upload batches, follow jobs, read events, record reviews."""

from __future__ import annotations

import logging
import time
import uuid
from collections import Counter, defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from wildinbox.api import views
from wildinbox.api.auth import PUBLIC_PATHS, principal
from wildinbox.api.uploads import (
    UploadedFile,
    UploadError,
    check_file,
    manifest,
    parse_metadata,
    request_fingerprint,
)
from wildinbox.config import load_config
from wildinbox.inference.releases import active_release_id, ensure_test_release, release_notice
from wildinbox.schemas import Review as ReviewContract
from wildinbox.schemas import ReviewOutcome
from wildinbox.settings import Settings
from wildinbox.storage.db import session_factory
from wildinbox.storage.models import (
    Batch,
    Decision,
    Event,
    Image,
    Job,
    ModelRelease,
    ReleaseActivation,
    Review,
)
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
        "weights_sha256": release.weights_sha256,
        "class_names": release.class_names,
        "class_map_fingerprint": release.class_map_fingerprint,
        "preprocessing_version": release.preprocessing_version,
        "calibration_version": (release.calibration or {}).get("version"),
        "policy_version": release.policy_version,
        "notice": release_notice(release),
    }


def _frame_status(img: Image, has_prediction: bool) -> str:
    if img.validation_status == "invalid":
        return "invalid"
    if has_prediction:
        return "completed"
    return "failed" if img.processing_error else "pending"


def batch_summary(session: Session, batch: Batch) -> dict[str, Any]:
    from wildinbox.storage.models import Prediction

    counts = Counter(i.validation_status for i in batch.images)
    events = session.scalars(select(Event.id).where(Event.batch_id == batch.id)).all()
    job = batch.jobs[0] if batch.jobs else None
    scored = (
        session.scalar(
            select(func.count())
            .select_from(Prediction)
            .join(Image)
            .where(Image.batch_id == batch.id, Prediction.model_release_id == job.model_release_id)
        )
        if job
        else 0
    )
    to_score = sum(1 for i in batch.images if i.validation_status in ("pending", "valid"))
    failures = [
        {
            "image_id": str(i.id),
            "filename": i.original_filename,
            "stage": "validation" if i.validation_status == "invalid" else "inference",
            "error": i.validation_error or i.processing_error,
        }
        for i in batch.images
        if i.validation_status == "invalid" or i.processing_error
    ]
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
            "processing_failed": sum(1 for i in batch.images if i.processing_error),
        },
        "progress": {
            "images_to_score": to_score,
            "images_scored": int(scored or 0),
            "images_failed": sum(1 for i in batch.images if i.processing_error),
            "finished": batch.status in ("completed", "completed_with_errors", "failed"),
        },
        "failures": failures,
        "job": _job(job) if job else None,
        "release": _release(job.release) if job else None,
        "links": {
            "self": f"/batches/{batch.id}",
            "images": f"/batches/{batch.id}/images",
            "events": f"/events?batch_id={batch.id}",
            "export": f"/batches/{batch.id}/export",
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
        "worker_id": job.worker_id,
        "heartbeat_at": job.heartbeat_at.isoformat() if job.heartbeat_at else None,
        "next_attempt_at": job.next_attempt_at.isoformat() if job.next_attempt_at else None,
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
        "processing_error": img.processing_error,
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
            "audit_selected": decision.audit_selected,
            "audit_rule": decision.audit_rule,
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
                "frame_status": _frame_status(i, i.id in preds),
                "prediction": None
                if i.id not in preds
                else {
                    "suggested_label": preds[i.id].suggested_label,
                    "confidence": preds[i.id].confidence,
                    "class_probabilities": preds[i.id].class_probabilities,
                    "calibrated_probabilities": preds[i.id].calibrated_probabilities,
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
    if settings.auth == "tokens" and not settings.api_tokens:
        raise RuntimeError(
            "WILDINBOX_AUTH=tokens but WILDINBOX_API_TOKENS is empty: create tokens with "
            "`wildinbox token new NAME`, or set WILDINBOX_AUTH=disabled for local development"
        )

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
    latency: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=2000))
    # Response status classes per route since the API started (2xx, 4xx, 5xx).
    statuses: dict[str, Counter[str]] = defaultdict(Counter)

    @app.middleware("http")
    async def _timing(request: Request, call_next: Any) -> Any:
        start = time.perf_counter()
        status = 500
        request_id = request.headers.get("x-request-id", "")[:100] or uuid.uuid4().hex
        who = None
        try:
            if settings.auth == "tokens":
                who = principal(request.headers.get("authorization"), settings.api_tokens)
                if who is None and request.url.path not in PUBLIC_PATHS:
                    status = 401
                    response = _error(401, "unauthorized", "send Authorization: Bearer <token>")
                    response.headers["WWW-Authenticate"] = "Bearer"
                    response.headers["X-Request-ID"] = request_id
                    return response
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            elapsed = time.perf_counter() - start
            route = request.scope.get("route")
            path = getattr(route, "path", None)
            if path:  # route templates only, so ids do not create unbounded series
                key = f"{request.method} {path}"
                latency[key].append(elapsed)
                statuses[key][f"{status // 100}xx"] += 1
            if request.url.path not in ("/health", "/ready") or status >= 400:
                log.info(
                    "request",
                    extra={
                        "fields": {
                            "request_id": request_id,
                            "method": request.method,
                            "route": path or request.url.path,
                            "status": status,
                            "duration_ms": round(1000 * elapsed, 1),
                            "principal": who,
                            "bytes_in": request.headers.get("content-length"),
                        }
                    },
                )

    def latency_summary() -> dict[str, dict[str, float]]:
        out = {}
        for key, values in sorted(latency.items()):
            v = sorted(values)
            codes = statuses[key]
            out[key] = {
                "requests": len(v),
                "p50_ms": round(1000 * v[len(v) // 2], 1),
                "p95_ms": round(1000 * v[min(len(v) - 1, int(0.95 * len(v)))], 1),
                "responses": sum(codes.values()),
                "client_errors": codes["4xx"],
                "server_errors": codes["5xx"],
            }
        return out

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

    @app.get("/ready")
    def ready() -> JSONResponse:
        """Ready to serve: database, queue, and object store reachable, and the
        expected model release active and loaded (weights verified)."""
        from wildinbox.api.readiness import check

        with sessions()() as s:
            ok, body = check(s, settings, app.state.store)
        return JSONResponse(body, status_code=200 if ok else 503)

    @app.get("/whoami")
    def whoami(request: Request) -> dict[str, Any]:
        who = principal(request.headers.get("authorization"), settings.api_tokens)
        return {"principal": who, "auth": settings.auth}

    @app.get("/version")
    def version() -> dict[str, Any]:
        """What new batches run: checked after every deploy."""
        with sessions()() as s:
            release = s.get(ModelRelease, active_release_id(s, settings))
            return {
                "api_version": app.version,
                "active_release": _release(release) if release else None,
            }

    @app.get("/monitoring")
    def monitoring() -> dict[str, Any]:
        """Operations, label-free signals per camera, and review-based accuracy."""
        from wildinbox.monitoring.metrics import load_config, summary

        with sessions()() as s:
            out = summary(
                s,
                settings.lease_seconds,
                load_config(settings.monitoring_config),
                api=latency_summary(),
            )
        out["operations"]["api_latency"] = latency_summary()
        return out

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        """Prometheus text format for scraping and alerting."""
        from wildinbox.monitoring.metrics import load_config, summary
        from wildinbox.monitoring.prometheus import render

        with sessions()() as s:
            data = summary(
                s,
                settings.lease_seconds,
                load_config(settings.monitoring_config),
                api=latency_summary(),
            )
        return render(data, latency_summary())

    @app.get("/releases")
    def list_releases() -> dict[str, Any]:
        with sessions()() as s:
            active = active_release_id(s, settings)
            return {
                "active_release_id": active,
                "releases": [
                    {
                        **_release(r),
                        "active": r.id == active,
                        "created_at": r.created_at.isoformat(),
                    }
                    for r in s.scalars(select(ModelRelease).order_by(ModelRelease.created_at))
                ],
                # Append-only: activating and rolling back are both new rows.
                "activations": [
                    {
                        "release_id": a.release_id,
                        "note": a.note,
                        "activated_at": a.activated_at.isoformat(),
                    }
                    for a in s.scalars(
                        select(ReleaseActivation).order_by(ReleaseActivation.id.desc())
                    )
                ],
            }

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return views.upload_page(settings, auth_required=settings.auth == "tokens")

    # ----------------------------------------------------------------- batches

    @app.post("/batches", status_code=202)
    async def create_batch(
        request: Request, idempotency_key: str | None = Header(default=None, max_length=200)
    ) -> JSONResponse:
        # Phase timings, returned as Server-Timing so clients can separate the
        # network transfer from the server's own work.
        marks = [("start", time.perf_counter())]
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
        marks.append(("receive", time.perf_counter()))  # body read and parsed
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
        marks.append(("validate", time.perf_counter()))
        response = await run_in_threadpool(_store_batch, files, metadata, idempotency_key, marks)
        phases = {
            name: round(1000 * (t - prev), 1)
            for (name, t), (_, prev) in zip(marks[1:], marks, strict=False)
        }
        response.headers["Server-Timing"] = ", ".join(f"{k};dur={v}" for k, v in phases.items())
        log.info("batch stored", extra={"fields": {"files": len(files), "phases_ms": phases}})
        return response

    def _put_originals(files: list[UploadedFile]) -> None:
        """Write accepted originals to object storage, several at a time: they
        are independent, content-addressed objects, and one at a time a
        1,000-file batch spends most of its upload waiting on S3 round trips."""
        from concurrent.futures import ThreadPoolExecutor

        todo = {
            f.sha256: f
            for f in files
            if f.accepted and f.data is not None and f.sha256 and f.content_type
        }

        def put(f: UploadedFile) -> None:
            assert f.sha256 and f.data is not None and f.content_type
            key = original_key(f.sha256)
            if not app.state.store.exists(key):
                app.state.store.put(key, f.data, f.content_type)

        with ThreadPoolExecutor(settings.store_concurrency) as pool:
            list(pool.map(put, todo.values()))  # re-raises the first failure

    def _store_batch(
        files: list[UploadedFile],
        metadata: Any,
        idempotency_key: str | None,
        marks: list[tuple[str, float]],
    ) -> JSONResponse:
        fingerprint = request_fingerprint(files, metadata)
        request_key = f"key:{idempotency_key}" if idempotency_key else f"content:{fingerprint}"
        with sessions()() as s:
            existing = _existing(s, request_key, fingerprint)
            if existing is not None:
                return existing
            _put_originals(files)
            marks.append(("store", time.perf_counter()))

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
                # Queued now, not when the transaction began (before the S3 writes).
                created_at=_utcnow(),
                kind="process_batch",
                status="queued",
                attempts=0,
                max_attempts=3,
                model_release_id=active_release_id(s, settings),  # pinned for all retries
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
            marks.append(("db", time.perf_counter()))
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

    @app.get("/batches")
    def list_batches(limit: int = 50) -> dict[str, Any]:
        """Most recent batches first."""
        limit = max(1, min(limit, 200))
        with sessions()() as s:
            batches = s.scalars(
                select(Batch)
                .where(Batch.workspace == settings.workspace)
                .order_by(Batch.created_at.desc())
                .limit(limit)
            ).all()
            return {
                "batches": [
                    {
                        "id": str(b.id),
                        "status": b.status,
                        "created_at": b.created_at.isoformat(),
                        "images": len(b.images),
                        "events": s.scalar(
                            select(func.count()).select_from(Event).where(Event.batch_id == b.id)
                        ),
                        "cameras": sorted({i.camera_id for i in b.images if i.camera_id}),
                    }
                    for b in batches
                ]
            }

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

    @app.get("/batches/{batch_id}/export")
    def export_batch(batch_id: uuid.UUID, format: str = "csv") -> Response:
        """Current observations with provenance, one row per capture event."""
        from wildinbox.api import export

        if format not in ("csv", "json"):
            raise ApiError(400, "invalid_format", "format must be 'csv' or 'json'")
        with sessions()() as s:
            batch = _batch(s, batch_id)
            events = list(
                s.scalars(
                    select(Event)
                    .where(Event.batch_id == batch.id)
                    .order_by(Event.start_at, Event.id)
                )
            )
            latest = {
                e.id: d
                for e in events
                if (d := max(e.decisions, key=lambda d: d.created_at, default=None)) is not None
            }
            releases = {
                r.id: r
                for r in s.scalars(
                    select(ModelRelease).where(
                        ModelRelease.id.in_({d.model_release_id for d in latest.values()})
                    )
                )
            }
            data = export.rows(events, latest, releases)
        name = f"wildinbox-{batch_id}-observations"
        if format == "json":
            return JSONResponse(
                {"batch_id": str(batch_id), "observations": data},
                headers={"Content-Disposition": f'attachment; filename="{name}.json"'},
            )
        return Response(
            export.to_csv(data),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{name}.csv"'},
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
        label: str | None = None,
        reason: str | None = None,
        reviewed: bool | None = None,
        audit: bool | None = None,
        start_after: datetime | None = None,
        start_before: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Events in time order. `label` matches the suggested label; `reason`
        a review reason; `reviewed` whether any human review exists;
        `start_after`/`start_before` bound the event's start (camera local time)."""
        limit, offset = max(1, min(limit, 500)), max(0, offset)
        with sessions()() as s:
            q = select(Event).join(Batch).where(Batch.workspace == settings.workspace)
            if batch_id:
                q = q.where(Event.batch_id == batch_id)
            if camera_id:
                q = q.where(Event.camera_id == camera_id)
            if disposition:
                q = q.where(Event.decisions.any(Decision.disposition == disposition))
            if label:
                q = q.where(Event.decisions.any(Decision.suggested_label == label))
            if reason:
                q = q.where(Event.decisions.any(Decision.reasons.contains([reason])))
            if reviewed is not None:
                q = q.where(Event.reviews.any() if reviewed else ~Event.reviews.any())
            if audit is not None:
                q = q.where(Event.decisions.any(Decision.audit_selected.is_(audit)))
            if start_after is not None:
                q = q.where(Event.start_at >= _naive(start_after))
            if start_before is not None:
                q = q.where(Event.start_at < _naive(start_before))
            total = s.scalar(select(func.count()).select_from(q.subquery())) or 0
            events = s.scalars(q.order_by(Event.start_at, Event.id).limit(limit).offset(offset))
            nxt = offset + limit if offset + limit < total else None
            return {
                "events": [event_row(s, e) for e in events],
                "total": total,
                "limit": limit,
                "offset": offset,
                "next_offset": nxt,
            }

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

    from wildinbox.api.study import add_study_routes

    add_study_routes(app, sessions, list(cfg.classes), ApiError, settings.audit_rate)

    @app.get("/images/{image_id}/thumbnail")
    def get_thumbnail(image_id: uuid.UUID, size: int = 320) -> Response:
        """A JPEG no larger than `size` px on its long side, generated once per
        original and size, then kept in object storage. Originals are untouched."""
        size = max(64, min(size, 1024))
        with sessions()() as s:
            img = s.get(Image, image_id)
            if img is None or img.batch.workspace != settings.workspace or not img.storage_key:
                raise ApiError(404, "not_found", f"image {image_id} has no stored original")
            key = f"thumbnails/{img.sha256}-{size}.jpg"
            store = app.state.store
            if not store.exists(key):
                try:
                    data = store.get(img.storage_key)
                except ObjectNotFoundError:
                    raise ApiError(
                        404, "not_found", "original missing from object storage"
                    ) from None
                try:
                    store.put(key, _thumbnail(data, size), "image/jpeg")
                except Exception:
                    raise ApiError(
                        422, "unreadable_image", "this file cannot be decoded as an image"
                    ) from None
            return Response(
                store.get(key),
                media_type="image/jpeg",
                headers={"Cache-Control": "public, max-age=31536000, immutable"},
            )

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


def _thumbnail(data: bytes, size: int) -> bytes:
    from io import BytesIO

    from wildinbox.preprocessing import load_image

    img = load_image(data)  # EXIF orientation applied, RGB
    img.thumbnail((size, size))
    out = BytesIO()
    img.save(out, "JPEG", quality=85)
    return out.getvalue()


def _naive(ts: Any) -> Any:
    return ts.replace(tzinfo=None) if ts is not None and ts.tzinfo else ts


def _utcnow() -> Any:
    from datetime import UTC, datetime

    return datetime.now(UTC)
