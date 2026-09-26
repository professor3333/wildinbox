"""PostgreSQL schema. Unique constraints make every processing step idempotent:
re-running a job updates rows instead of duplicating them.

Changes to this module need an Alembic migration (`migrations/`); CI fails if
models and migrations disagree.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, ClassVar

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

NAMING = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

BATCH_STATUSES = ("queued", "processing", "completed", "completed_with_errors", "failed")
IMAGE_STATUSES = ("pending", "valid", "invalid", "duplicate")
JOB_STATUSES = ("queued", "running", "succeeded", "failed")
DISPOSITIONS = ("likely_empty", "species_identified", "needs_review")
REVIEW_OUTCOMES = ("confirmed", "corrected", "unresolved")


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING)
    type_annotation_map: ClassVar[dict[Any, Any]] = {dict[str, Any]: JSONB, list[Any]: JSONB}


def _now() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class ModelRelease(Base):
    """Everything needed to reproduce a prediction: weights, classes, preprocessing,
    calibration, and decision policy. Immutable once written (a database trigger
    rejects UPDATE), so a job pinned to a release always runs the same thing."""

    __tablename__ = "model_releases"
    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    kind: Mapped[str] = mapped_column(String(40))
    is_test: Mapped[bool] = mapped_column(Boolean)
    weights_key: Mapped[str | None] = mapped_column(Text)
    weights_sha256: Mapped[str | None] = mapped_column(String(64))
    class_names: Mapped[list[Any]]
    class_map_fingerprint: Mapped[str] = mapped_column(String(64))
    preprocessing: Mapped[dict[str, Any]]
    preprocessing_version: Mapped[str] = mapped_column(String(64))
    calibration: Mapped[dict[str, Any] | None]
    policy: Mapped[dict[str, Any]]
    policy_version: Mapped[str] = mapped_column(String(100))
    # Where the release came from: training run, code commit, split, artifacts.
    provenance: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=text("'{}'"))
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _now()


class ReleaseActivation(Base):
    """Append-only log of which release new batches use. The latest row wins,
    so activating and rolling back are both a new row, never an edit."""

    __tablename__ = "release_activations"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    release_id: Mapped[str] = mapped_column(ForeignKey("model_releases.id"))
    note: Mapped[str | None] = mapped_column(Text)
    activated_at: Mapped[datetime] = _now()


class Batch(Base):
    __tablename__ = "batches"
    __table_args__ = (
        UniqueConstraint("workspace", "request_key"),
        CheckConstraint(_in("status", BATCH_STATUSES), name="status"),
    )
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    workspace: Mapped[str] = mapped_column(String(100))
    request_key: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(30))
    manifest: Mapped[dict[str, Any]]
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _now()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    images: Mapped[list[Image]] = relationship(back_populates="batch", order_by="Image.position")
    jobs: Mapped[list[Job]] = relationship(back_populates="batch")


class Image(Base):
    __tablename__ = "images"
    __table_args__ = (
        UniqueConstraint("batch_id", "position"),
        CheckConstraint(_in("validation_status", IMAGE_STATUSES), name="validation_status"),
        CheckConstraint(
            "validation_status != 'invalid' OR validation_error IS NOT NULL", name="error_reason"
        ),
        Index("ix_images_sha256", "sha256"),
    )
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    batch_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("batches.id", ondelete="CASCADE"))
    position: Mapped[int] = mapped_column(Integer)
    original_filename: Mapped[str] = mapped_column(Text)
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    sha256: Mapped[str | None] = mapped_column(String(64))
    storage_key: Mapped[str | None] = mapped_column(Text)
    content_type: Mapped[str | None] = mapped_column(String(50))
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    camera_id: Mapped[str | None] = mapped_column(String(200))
    # Camera clocks have no time zone; stored as the camera's local time.
    captured_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    sequence_id: Mapped[str | None] = mapped_column(String(200))
    user_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default=text("'{}'"))
    validation_status: Mapped[str] = mapped_column(String(20))
    validation_error: Mapped[str | None] = mapped_column(Text)
    # A valid image whose inference failed after the job's retries: a failed frame.
    processing_error: Mapped[str | None] = mapped_column(Text)
    # Night flag, blur, brightness (wildinbox.quality), recorded when scored.
    quality: Mapped[dict[str, Any] | None]
    duplicate_of: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("images.id"))
    event_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("events.id", ondelete="SET NULL"), index=True
    )
    created_at: Mapped[datetime] = _now()

    batch: Mapped[Batch] = relationship(back_populates="images")
    event: Mapped[Event | None] = relationship(back_populates="images")


class Event(Base):
    __tablename__ = "events"
    __table_args__ = (UniqueConstraint("batch_id", "group_key"),)
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    batch_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("batches.id", ondelete="CASCADE"))
    camera_id: Mapped[str | None] = mapped_column(String(200))
    grouping_rule: Mapped[str] = mapped_column(String(100))
    group_key: Mapped[str] = mapped_column(String(64))
    start_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    end_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False))
    created_at: Mapped[datetime] = _now()

    images: Mapped[list[Image]] = relationship(back_populates="event", order_by="Image.position")
    decisions: Mapped[list[Decision]] = relationship(back_populates="event")
    reviews: Mapped[list[Review]] = relationship(
        back_populates="event", order_by="Review.created_at"
    )


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        UniqueConstraint("batch_id", "kind"),
        CheckConstraint(_in("status", JOB_STATUSES), name="status"),
    )
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    batch_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("batches.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(20))
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    error: Mapped[str | None] = mapped_column(Text)
    model_release_id: Mapped[str] = mapped_column(ForeignKey("model_releases.id"))
    created_at: Mapped[datetime] = _now()
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Lease: the worker holding `lease_token` owns the job while `heartbeat_at`
    # is fresh. Every progress commit is conditional on still holding it.
    lease_token: Mapped[uuid.UUID | None] = mapped_column()
    worker_id: Mapped[str | None] = mapped_column(String(200))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # A failed attempt with retries left waits until this time.
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    batch: Mapped[Batch] = relationship(back_populates="jobs")
    release: Mapped[ModelRelease] = relationship()


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (UniqueConstraint("image_id", "model_release_id"),)
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    image_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("images.id", ondelete="CASCADE"))
    event_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("events.id", ondelete="SET NULL"), index=True
    )
    model_release_id: Mapped[str] = mapped_column(ForeignKey("model_releases.id"))
    class_probabilities: Mapped[dict[str, Any]]  # raw model output
    calibrated_probabilities: Mapped[dict[str, Any] | None]  # what the policy uses
    suggested_label: Mapped[str] = mapped_column(String(100))
    confidence: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = _now()


class Decision(Base):
    __tablename__ = "decisions"
    __table_args__ = (
        UniqueConstraint("event_id", "model_release_id", "policy_version"),
        CheckConstraint(_in("disposition", DISPOSITIONS), name="disposition"),
    )
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    event_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("events.id", ondelete="CASCADE"))
    model_release_id: Mapped[str] = mapped_column(ForeignKey("model_releases.id"))
    policy_version: Mapped[str] = mapped_column(String(100))
    disposition: Mapped[str] = mapped_column(String(30))
    suggested_label: Mapped[str | None] = mapped_column(String(100))
    confidence: Mapped[float | None] = mapped_column(Float)
    reasons: Mapped[list[Any]]
    # Automatic decisions sampled for a human audit, and the rule that chose them.
    audit_selected: Mapped[bool] = mapped_column(Boolean, server_default=text("false"))
    audit_rule: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _now()

    event: Mapped[Event] = relationship(back_populates="decisions")


class Review(Base):
    """Human verdicts are appended, never overwritten; each points at the review it supersedes."""

    __tablename__ = "reviews"
    __table_args__ = (
        CheckConstraint(_in("outcome", REVIEW_OUTCOMES), name="outcome"),
        # One chain per event: a single first review (the unique constraint on
        # previous_review_id cannot see NULLs), and each review superseded once.
        Index(
            "uq_reviews_one_first_review_per_event",
            "event_id",
            unique=True,
            postgresql_where=text("previous_review_id IS NULL"),
        ),
    )
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    event_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("events.id", ondelete="CASCADE"))
    reviewer: Mapped[str] = mapped_column(String(200))
    # The authenticated principal that submitted the review: the reviewer
    # itself, or an authorized delegate recording on the reviewer's behalf.
    # Null where the deployment has no authentication (local development).
    recorded_by: Mapped[str | None] = mapped_column(String(200))
    outcome: Mapped[str] = mapped_column(String(20))
    suggested_label: Mapped[str | None] = mapped_column(String(100))
    confirmed_label: Mapped[str | None] = mapped_column(String(100))
    note: Mapped[str | None] = mapped_column(Text)
    previous_review_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("reviews.id"), unique=True
    )
    created_at: Mapped[datetime] = _now()

    event: Mapped[Event] = relationship(back_populates="reviews")


class StudyPlan(Base):
    """A timed review study (configs/study/review_study.yaml): the two event
    sets, practice events, and ground truth for scoring. Separate from
    production reviews: study choices never become reviews."""

    __tablename__ = "study_plans"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200))
    protocol_sha256: Mapped[str] = mapped_column(String(64))
    sets: Mapped[dict[str, Any]]  # {"A": [event ids], "B": [...], "practice": [...]}
    truth: Mapped[dict[str, Any]]  # event id -> ground-truth label (None: mixed / unknown)
    created_at: Mapped[datetime] = _now()


class StudyParticipant(Base):
    __tablename__ = "study_participants"
    __table_args__ = (UniqueConstraint("plan_id", "code"),)
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    plan_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("study_plans.id", ondelete="CASCADE"))
    code: Mapped[str] = mapped_column(String(50))  # chosen by the participant; no personal data
    arm: Mapped[int] = mapped_column(Integer)
    joined_at: Mapped[datetime] = _now()


class StudyTrial(Base):
    """One timed decision. Append-only; one row per participant and event."""

    __tablename__ = "study_trials"
    __table_args__ = (
        UniqueConstraint("plan_id", "participant", "event_id"),
        CheckConstraint(_in("condition", ("grouped", "suggested")), name="condition"),
    )
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    plan_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("study_plans.id", ondelete="CASCADE"))
    participant: Mapped[str] = mapped_column(String(50))
    block: Mapped[int] = mapped_column(Integer)  # 0 practice, 1 and 2 timed blocks
    condition: Mapped[str] = mapped_column(String(20))
    event_id: Mapped[uuid.UUID] = mapped_column()
    label: Mapped[str | None] = mapped_column(String(100))  # None: "can't tell"
    seconds: Mapped[float] = mapped_column(Float)
    interactions: Mapped[int] = mapped_column(Integer)
    shown_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class StudyRating(Base):
    __tablename__ = "study_ratings"
    __table_args__ = (
        UniqueConstraint("plan_id", "participant", "block"),
        CheckConstraint("difficulty BETWEEN 1 AND 5", name="difficulty"),
    )
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    plan_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("study_plans.id", ondelete="CASCADE"))
    participant: Mapped[str] = mapped_column(String(50))
    block: Mapped[int] = mapped_column(Integer)
    condition: Mapped[str] = mapped_column(String(20))
    difficulty: Mapped[int] = mapped_column(Integer)


class WorkerProcess(Base):
    """One row per worker process start: liveness, memory, and how it ended.

    A row whose `last_seen_at` went stale without `stopped_at` is a worker that
    died (killed, crashed, out of memory) rather than shutting down."""

    __tablename__ = "worker_processes"
    id: Mapped[str] = mapped_column(String(200), primary_key=True)  # worker_id()
    hostname: Mapped[str] = mapped_column(String(200))
    pid: Mapped[int] = mapped_column(Integer)
    started_at: Mapped[datetime] = _now()
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    stopped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    rss_bytes: Mapped[int | None] = mapped_column(BigInteger)
    peak_rss_bytes: Mapped[int | None] = mapped_column(BigInteger)
