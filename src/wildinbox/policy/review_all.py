"""The first decision policy: every event goes to human review.

It aggregates frame scores into an event-level suggestion so reviewers see
something to accept or correct, but it never filters or auto-accepts. That is
the Stage 1 default until evaluation supports an operating point, and it is
mandatory for test releases.
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from wildinbox.schemas import Disposition, EventDecision, ReviewReason

POLICY_VERSION = "review-all/v0"


def decide(
    event_id: UUID,
    image_ids: Sequence[UUID],
    frame_probabilities: Sequence[dict[str, float]],
    *,
    model_version: str,
    preprocessing_version: str,
) -> EventDecision:
    if not frame_probabilities:
        raise ValueError("an event needs at least one scored frame")
    classes = frame_probabilities[0].keys()
    mean = {c: sum(p[c] for p in frame_probabilities) / len(frame_probabilities) for c in classes}
    label = max(mean, key=mean.__getitem__)
    return EventDecision(
        event_id=event_id,
        image_ids=list(image_ids),
        disposition=Disposition.NEEDS_REVIEW,
        suggested_label=label,
        confidence=round(mean[label], 6),
        reasons=[ReviewReason.AUTOMATION_DISABLED],
        model_version=model_version,
        preprocessing_version=preprocessing_version,
        policy_version=POLICY_VERSION,
    )
