"""Conservative event policy, versioned (CLAUDE.md, "Sequence aggregation"):

- Filter an event as empty only when EVERY frame has completed processing and
  has P(empty) >= the empty threshold. Incomplete events never filter.
- Retain the event when any completed frame suggests an animal.
- Accept a species only when all animal-suggesting frames agree, none looks
  unfamiliar, their mean probability is >= the species threshold, the species
  has enough validation evidence, and automatic acceptance is enabled.
- Everything else goes to review, listing every reason that blocks automation.

Separate thresholds, because a wrongly filtered animal is worse than an extra
review. Scores are calibrated before they reach this module.

Versions (decisions are identical; only the listed reasons differ):
- conservative/v1: a disabled species threshold counts as unreachable, so every
  animal event also gets `low_confidence`, even for a confident suggestion.
- conservative/v2: `low_confidence` only when a species threshold exists and
  the suggestion falls below it; with acceptance disabled the reason is
  `automation_disabled`, and the confidence itself is shown to reviewers.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field

from wildinbox.class_map import EMPTY_CLASS
from wildinbox.schemas import Disposition, FrameStatus, ReviewReason

POLICY_NAME = "conservative/v1"
POLICY_V2 = "conservative/v2"
POLICIES = (POLICY_NAME, POLICY_V2)


@dataclass(frozen=True)
class Thresholds:
    empty: float
    species: float


@dataclass(frozen=True)
class Frame:
    probs: Mapping[str, float] | None  # calibrated; None unless completed
    status: FrameStatus = FrameStatus.COMPLETED
    unfamiliar: bool = False


@dataclass(frozen=True)
class PolicyConfig:
    """Everything the policy needs besides the frames. A threshold of None (or
    the matching switch off) disables that automation."""

    empty_threshold: float | None
    species_threshold: float | None
    auto_filter_enabled: bool
    auto_accept_enabled: bool
    # Species with enough validation evidence to be accepted; None = no gate.
    accept_species: tuple[str, ...] | None = None

    def fingerprint(self, policy: str = POLICY_NAME) -> str:
        payload = json.dumps({"policy": policy, **asdict(self)}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def version(self, policy: str = POLICY_NAME) -> str:
        return f"{policy}+{self.fingerprint(policy)}"


@dataclass(frozen=True)
class EventOutcome:
    disposition: Disposition
    label: str | None
    confidence: float | None
    reasons: list[ReviewReason] = field(default_factory=list)
    # Frames that kept the event out of the empty filter (explains retentions).
    blocking_frames: tuple[int, ...] = ()


def _argmax(p: Mapping[str, float]) -> str:
    return max(p, key=p.__getitem__)


def _t(x: float | None) -> float:
    return math.inf if x is None else x


def decide(frames: Sequence[Frame], cfg: PolicyConfig, policy: str = POLICY_NAME) -> EventOutcome:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    if not frames:
        raise ValueError("an event needs at least one frame")
    reasons: list[ReviewReason] = []
    done = [
        (i, f.probs) for i, f in enumerate(frames) if f.status is FrameStatus.COMPLETED and f.probs
    ]
    if len(done) < len(frames):
        reasons.append(ReviewReason.PROCESSING_FAILURE)
    if not done:
        return EventOutcome(Disposition.NEEDS_REVIEW, None, None, reasons)

    t_empty, t_species = _t(cfg.empty_threshold), _t(cfg.species_threshold)
    p_empty = [p[EMPTY_CLASS] for _, p in done]
    blocking = tuple(i for i, p in done if p[EMPTY_CLASS] < t_empty)
    if not blocking:
        if not (cfg.auto_filter_enabled and cfg.empty_threshold is not None):
            reasons.append(ReviewReason.AUTOMATION_DISABLED)
        if reasons:
            return EventOutcome(Disposition.NEEDS_REVIEW, EMPTY_CLASS, min(p_empty), reasons)
        return EventOutcome(Disposition.LIKELY_EMPTY, EMPTY_CLASS, min(p_empty))

    animal = [(i, p) for i, p in done if _argmax(p) != EMPTY_CLASS]
    if not animal:
        reasons.append(ReviewReason.LOW_CONFIDENCE)
        return EventOutcome(Disposition.NEEDS_REVIEW, EMPTY_CLASS, min(p_empty), reasons, blocking)
    species = sorted({_argmax(p) for _, p in animal})
    means = {s: sum(p[s] for _, p in animal) / len(animal) for s in species}
    best = max(means, key=means.__getitem__)
    if len(species) > 1:
        reasons.append(ReviewReason.CONFLICTING_FRAMES)
    if any(frames[i].unfamiliar for i, _ in animal):
        reasons.append(ReviewReason.POSSIBLE_UNKNOWN)
    species_unsure = means[best] < t_species
    if policy == POLICY_V2:
        # A disabled threshold says nothing about the model's confidence.
        species_unsure = cfg.species_threshold is not None and means[best] < cfg.species_threshold
    if species_unsure:
        reasons.append(ReviewReason.LOW_CONFIDENCE)
    if cfg.accept_species is not None and best not in cfg.accept_species:
        reasons.append(ReviewReason.SPECIES_NOT_VALIDATED)
    if not (cfg.auto_accept_enabled and cfg.species_threshold is not None):
        reasons.append(ReviewReason.AUTOMATION_DISABLED)
    if reasons:
        return EventOutcome(Disposition.NEEDS_REVIEW, best, means[best], reasons, blocking)
    return EventOutcome(Disposition.SPECIES_IDENTIFIED, best, means[best], [], blocking)


def decide_event(
    frames: Sequence[Mapping[str, float]],
    t: Thresholds,
    unfamiliar: Sequence[bool] | None = None,
    accept_species: tuple[str, ...] | None = None,
) -> EventOutcome:
    """What the policy would decide at thresholds `t` with automation enabled,
    for fully processed frames. Used by evaluation sweeps."""
    flags = unfamiliar or [False] * len(frames)
    cfg = PolicyConfig(t.empty, t.species, True, True, accept_species)
    return decide(
        [Frame(p, FrameStatus.COMPLETED, u) for p, u in zip(frames, flags, strict=True)], cfg
    )
