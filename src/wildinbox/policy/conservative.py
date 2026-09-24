"""Conservative event policy (CLAUDE.md, "Sequence aggregation"):

- Filter an event as empty only when EVERY usable frame has P(empty) >= the
  empty threshold.
- Retain the event when any frame suggests an animal.
- Accept a species only when all animal-suggesting frames agree and their mean
  probability for it is >= the species threshold.
- Conflicting species, low confidence, or unfamiliar inputs go to review.

Separate thresholds, because a wrongly filtered animal is worse than an extra
review.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from wildinbox.class_map import EMPTY_CLASS
from wildinbox.schemas import Disposition, ReviewReason


@dataclass(frozen=True)
class Thresholds:
    empty: float
    species: float


@dataclass(frozen=True)
class EventOutcome:
    disposition: Disposition
    label: str | None
    confidence: float | None
    reasons: list[ReviewReason] = field(default_factory=list)
    # Frames that kept the event out of the empty filter (explains retentions).
    blocking_frames: tuple[int, ...] = ()


def _argmax(p: dict[str, float]) -> str:
    return max(p, key=p.__getitem__)


def decide_event(
    frames: Sequence[dict[str, float]],
    t: Thresholds,
    unfamiliar: Sequence[bool] | None = None,
) -> EventOutcome:
    if not frames:
        raise ValueError("an event needs at least one scored frame")
    p_empty = [p[EMPTY_CLASS] for p in frames]
    blocking = tuple(i for i, pe in enumerate(p_empty) if pe < t.empty)
    if not blocking:
        return EventOutcome(Disposition.LIKELY_EMPTY, EMPTY_CLASS, min(p_empty))

    animal = [i for i, p in enumerate(frames) if _argmax(p) != EMPTY_CLASS]
    if not animal:
        return EventOutcome(
            Disposition.NEEDS_REVIEW,
            EMPTY_CLASS,
            min(p_empty),
            [ReviewReason.LOW_CONFIDENCE],
            blocking,
        )
    species = sorted({_argmax(frames[i]) for i in animal})
    means = {s: sum(frames[i][s] for i in animal) / len(animal) for s in species}
    best = max(means, key=means.__getitem__)
    if len(species) > 1:
        return EventOutcome(
            Disposition.NEEDS_REVIEW,
            best,
            means[best],
            [ReviewReason.CONFLICTING_SPECIES],
            blocking,
        )
    if unfamiliar is not None and any(unfamiliar[i] for i in animal):
        return EventOutcome(
            Disposition.NEEDS_REVIEW,
            best,
            means[best],
            [ReviewReason.POSSIBLE_UNSUPPORTED_INPUT],
            blocking,
        )
    if means[best] >= t.species:
        return EventOutcome(Disposition.SPECIES_IDENTIFIED, best, means[best], [], blocking)
    return EventOutcome(
        Disposition.NEEDS_REVIEW, best, means[best], [ReviewReason.LOW_CONFIDENCE], blocking
    )
