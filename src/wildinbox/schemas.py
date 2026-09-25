"""Shared data contracts: image metadata, model predictions, event decisions,
and reviews. The API, workers, database layer, and UI exchange these shapes.

Terminology follows docs/requirements.md.
"""

from __future__ import annotations

import math
from datetime import datetime
from enum import StrEnum
from typing import Literal, Self
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from wildinbox.class_map import EMPTY_CLASS, ClassMap

_SHA256 = r"^[0-9a-f]{64}$"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ImageMetadata(_Contract):
    """One uploaded photograph, after validation."""

    image_id: UUID
    sha256: str = Field(pattern=_SHA256, description="Content hash of the original file.")
    original_filename: str = Field(min_length=1)
    content_type: Literal["image/jpeg", "image/png"]
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    camera_id: str | None = None
    captured_at: datetime | None = None
    sequence_id: str | None = None
    frame_index: int | None = Field(default=None, ge=0)


class Prediction(_Contract):
    """The model's per-image output, stored before any event aggregation."""

    image_id: UUID
    model_version: str = Field(min_length=1)
    preprocessing_version: str = Field(min_length=1)
    class_probabilities: dict[str, float] = Field(min_length=2)
    unfamiliar_score: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _is_distribution(self) -> Self:
        probs = self.class_probabilities
        out_of_range = {k: p for k, p in probs.items() if not 0 <= p <= 1}
        if out_of_range:
            raise ValueError(f"probabilities must be in [0, 1], got {out_of_range}")
        total = sum(probs.values())
        if not math.isclose(total, 1.0, abs_tol=1e-4):
            raise ValueError(f"class probabilities must sum to 1, got {total:.6f}")
        return self

    @property
    def suggested_label(self) -> str:
        return max(self.class_probabilities, key=self.class_probabilities.__getitem__)

    def check_classes(self, class_map: ClassMap) -> None:
        """Raise if the prediction's classes differ from the deployed class map."""
        if set(self.class_probabilities) != set(class_map.names):
            raise ValueError(
                f"prediction classes {sorted(self.class_probabilities)} do not match "
                f"class map {sorted(class_map.names)}"
            )


class Disposition(StrEnum):
    LIKELY_EMPTY = "likely_empty"
    SPECIES_IDENTIFIED = "species_identified"
    NEEDS_REVIEW = "needs_review"


class ReviewReason(StrEnum):
    """Machine-readable reasons an event needs review. Every reason that
    independently blocks automation is listed."""

    LOW_CONFIDENCE = "low_confidence"
    CONFLICTING_FRAMES = "conflicting_frames"
    POSSIBLE_UNKNOWN = "possible_unknown"
    PROCESSING_FAILURE = "processing_failure"
    SPECIES_NOT_VALIDATED = "species_not_validated"
    AUTOMATION_DISABLED = "automation_disabled"


class FrameStatus(StrEnum):
    """Prediction status of one frame of an event."""

    COMPLETED = "completed"
    FAILED = "failed"
    PENDING = "pending"


class EventDecision(_Contract):
    """The decision policy's output for one capture event."""

    event_id: UUID
    image_ids: list[UUID] = Field(min_length=1)
    disposition: Disposition
    suggested_label: str | None
    confidence: float | None = Field(ge=0, le=1)
    reasons: list[ReviewReason] = Field(default_factory=list)
    model_version: str = Field(min_length=1)
    preprocessing_version: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        d, label = self.disposition, self.suggested_label
        if d is Disposition.NEEDS_REVIEW and not self.reasons:
            raise ValueError("a needs_review decision must give at least one reason")
        if d is not Disposition.NEEDS_REVIEW and self.reasons:
            raise ValueError(f"review reasons only apply to needs_review, not {d.value}")
        if d is Disposition.LIKELY_EMPTY and label != EMPTY_CLASS:
            raise ValueError(
                f"likely_empty requires suggested_label {EMPTY_CLASS!r}, got {label!r}"
            )
        if d is Disposition.SPECIES_IDENTIFIED and (label is None or label == EMPTY_CLASS):
            raise ValueError(f"species_identified requires a species label, got {label!r}")
        return self


class ReviewOutcome(StrEnum):
    CONFIRMED = "confirmed"
    CORRECTED = "corrected"
    UNRESOLVED = "unresolved"


class Review(_Contract):
    """A human's verdict on one event. The original suggestion is kept alongside."""

    review_id: UUID
    event_id: UUID
    outcome: ReviewOutcome
    suggested_label: str | None = Field(description="The model's label at review time.")
    confirmed_label: str | None
    reviewer: str = Field(min_length=1)
    reviewed_at: AwareDatetime
    note: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        o = self.outcome
        if o is ReviewOutcome.UNRESOLVED and self.confirmed_label is not None:
            raise ValueError("an unresolved review cannot have a confirmed_label")
        if o is not ReviewOutcome.UNRESOLVED and not self.confirmed_label:
            raise ValueError(f"a {o.value} review needs a confirmed_label")
        if o is ReviewOutcome.CONFIRMED and self.confirmed_label != self.suggested_label:
            raise ValueError("confirmed means the label equals the suggestion; use corrected")
        if o is ReviewOutcome.CORRECTED and self.confirmed_label == self.suggested_label:
            raise ValueError("corrected means the label differs from the suggestion; use confirmed")
        return self
