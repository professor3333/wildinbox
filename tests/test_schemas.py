from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from wildinbox.class_map import ClassMap
from wildinbox.schemas import (
    Disposition,
    EventDecision,
    ImageMetadata,
    Prediction,
    Review,
    ReviewOutcome,
    ReviewReason,
)

VERSIONS = {"model_version": "m1", "preprocessing_version": "p1"}


def _prediction(**probs: float) -> Prediction:
    return Prediction(image_id=uuid4(), class_probabilities=probs, **VERSIONS)


def _decision(**overrides: Any) -> EventDecision:
    fields: dict[str, Any] = {
        "event_id": uuid4(),
        "image_ids": [uuid4()],
        "disposition": Disposition.SPECIES_IDENTIFIED,
        "suggested_label": "raccoon",
        "confidence": 0.97,
        "policy_version": "policy-v0",
        **VERSIONS,
    }
    fields.update(overrides)
    return EventDecision(**fields)


def _review(**overrides: Any) -> Review:
    fields: dict[str, Any] = {
        "review_id": uuid4(),
        "event_id": uuid4(),
        "outcome": ReviewOutcome.CORRECTED,
        "suggested_label": "raccoon",
        "confirmed_label": "opossum",
        "reviewer": "volunteer-1",
        "reviewed_at": datetime.now(UTC),
    }
    fields.update(overrides)
    return Review(**fields)


def test_image_metadata_requires_hash_and_supported_type() -> None:
    ok = {
        "image_id": uuid4(),
        "sha256": "a" * 64,
        "original_filename": "IMG_0001.JPG",
        "content_type": "image/jpeg",
        "width": 2048,
        "height": 1536,
    }
    ImageMetadata(**ok)
    with pytest.raises(ValidationError, match="sha256"):
        ImageMetadata(**{**ok, "sha256": "xyz"})
    with pytest.raises(ValidationError, match="content_type"):
        ImageMetadata(**{**ok, "content_type": "image/gif"})


def test_prediction_must_be_a_distribution() -> None:
    p = _prediction(empty=0.1, raccoon=0.9)
    assert p.suggested_label == "raccoon"
    with pytest.raises(ValidationError, match="sum to 1"):
        _prediction(empty=0.5, raccoon=0.6)
    with pytest.raises(ValidationError, match=r"in \[0, 1\]"):
        _prediction(empty=-0.1, raccoon=1.1)


def test_prediction_classes_must_match_class_map() -> None:
    cm = ClassMap(["empty", "raccoon", "coyote"])
    _prediction(empty=0.2, raccoon=0.5, coyote=0.3).check_classes(cm)
    with pytest.raises(ValueError, match="do not match"):
        _prediction(empty=0.2, raccoon=0.8).check_classes(cm)


def test_decision_consistency_rules() -> None:
    _decision()
    _decision(disposition=Disposition.LIKELY_EMPTY, suggested_label="empty")
    _decision(
        disposition=Disposition.NEEDS_REVIEW,
        suggested_label=None,
        confidence=None,
        reasons=[ReviewReason.CONFLICTING_FRAMES],
    )
    with pytest.raises(ValidationError, match="at least one reason"):
        _decision(disposition=Disposition.NEEDS_REVIEW)
    with pytest.raises(ValidationError, match="requires a species label"):
        _decision(suggested_label="empty")
    with pytest.raises(ValidationError, match="likely_empty requires"):
        _decision(disposition=Disposition.LIKELY_EMPTY, suggested_label="raccoon")
    with pytest.raises(ValidationError, match="only apply to needs_review"):
        _decision(reasons=[ReviewReason.LOW_CONFIDENCE])


def test_review_keeps_suggestion_and_checks_outcome() -> None:
    r = _review()
    assert r.suggested_label == "raccoon" and r.confirmed_label == "opossum"
    _review(outcome=ReviewOutcome.CONFIRMED, confirmed_label="raccoon")
    _review(outcome=ReviewOutcome.UNRESOLVED, confirmed_label=None)
    with pytest.raises(ValidationError, match="use corrected"):
        _review(outcome=ReviewOutcome.CONFIRMED, confirmed_label="opossum")
    with pytest.raises(ValidationError, match="use confirmed"):
        _review(confirmed_label="raccoon")
    with pytest.raises(ValidationError, match="cannot have a confirmed_label"):
        _review(outcome=ReviewOutcome.UNRESOLVED)
    with pytest.raises(ValidationError, match="timezone"):
        _review(reviewed_at=datetime(2026, 1, 1))
