"""Judge adaptation methods on each camera's later events.

For a method and a number N of reviewed events, every development camera is
adapted from its first N events and its later events are scored once. The
policy's own event logic (`decide_event`) gives each later event its
suggestion and confidence, so thresholds can be swept cheaply:

- an event is filtered as empty when every frame has P(empty) >= t_empty;
- otherwise a species is accepted when the animal frames agree and their mean
  probability is >= t_species.

Thresholds follow the release rule (operating_point_v2): the lowest grid
value whose 95% Wilson bound meets the target (accepted-species precision
lower bound >= 95%; animal events filtered as empty, upper bound <= 2%).
Leave-one-camera-out applies thresholds chosen on the other cameras to each
camera in turn: the estimate for a camera the rule has never seen.

With `filter_empty=False` no event is filtered as empty and only species
labels are automated.

Review reduction counts every event of the camera, the N reviewed ones
included, and charges a 5% audit on automated events.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from wildinbox.adaptation.data import CameraEvent, FrameOutputs
from wildinbox.adaptation.methods import OTHER, Method
from wildinbox.class_map import EMPTY_CLASS
from wildinbox.evaluation.metrics import wilson
from wildinbox.policy.conservative import Thresholds, decide_event
from wildinbox.schemas import ReviewReason

GRID = tuple(round(x, 2) for x in np.arange(0.50, 1.00, 0.01))
PRECISION_TARGET = 0.95
FALSE_EMPTY_LIMIT = 0.02
AUDIT_RATE = 0.05
ANIMAL_ROLES = {"supported_species", "unsupported_animal", "mixed_species"}


@dataclass(frozen=True)
class Scored:
    """One later event: what the adapted model says and what is true."""

    camera_id: str
    role: str
    label: str | None
    min_p_empty: float
    suggestion: str | None  # None: frames disagree, or no animal frame
    confidence: float

    @property
    def animal(self) -> bool:
        return self.role in ANIMAL_ROLES

    def filtered(self, t_empty: float) -> bool:
        return self.min_p_empty >= t_empty

    def accepted(self, t_empty: float, t_species: float) -> bool:
        return (
            not self.filtered(t_empty)
            and self.suggestion not in (None, EMPTY_CLASS, OTHER)
            and self.confidence >= t_species
        )

    @property
    def correct(self) -> bool:
        return self.role == "supported_species" and self.suggestion == self.label


def score_camera(
    method: Method, events: list[CameraEvent], n: int, out: FrameOutputs
) -> tuple[list[Scored], int]:
    """Adapt from the first `n` events; score the rest. Returns them and the
    camera's total number of events."""
    reviewed, later = events[:n], events[n:]
    adapted = method.adapt(reviewed, out)
    scored = []
    for e in later:
        probs = adapted.score(out.rows(e.image_ids))
        frames = [dict(zip(adapted.classes, map(float, p), strict=True)) for p in probs]
        o = decide_event(frames, Thresholds(empty=math.inf, species=0.0))
        agree = ReviewReason.CONFLICTING_FRAMES not in o.reasons
        scored.append(
            Scored(
                camera_id=e.camera_id,
                role=e.role,
                label=e.label,
                min_p_empty=min(f[EMPTY_CLASS] for f in frames),
                suggestion=o.label if agree else None,
                confidence=o.confidence or 0.0,
            )
        )
    return scored, len(events)


def choose(
    scored: Sequence[Scored], filter_empty: bool = True
) -> tuple[float | None, float | None]:
    """The release rule's thresholds on these events (None: nothing passes,
    or empty filtering is off)."""
    animals = [s for s in scored if s.animal]
    t_empty = next(
        (
            t
            for t in GRID
            if filter_empty
            and animals
            and wilson(sum(s.filtered(t) for s in animals), len(animals))[1] <= FALSE_EMPTY_LIMIT
        ),
        None,
    )
    te = t_empty if t_empty is not None else math.inf
    t_species = None
    for t in GRID:
        acc = [s for s in scored if s.accepted(te, t)]
        if acc and wilson(sum(s.correct for s in acc), len(acc))[0] >= PRECISION_TARGET:
            t_species = t
            break
    return t_empty, t_species


def outcome(
    scored: Sequence[Scored], total: int, t_empty: float | None, t_species: float | None
) -> dict[str, Any]:
    te = t_empty if t_empty is not None else math.inf
    ts = t_species if t_species is not None else math.inf
    filtered = [s for s in scored if s.filtered(te)]
    accepted = [s for s in scored if s.accepted(te, ts)]
    animals = [s for s in scored if s.animal]
    automated = len(filtered) + len(accepted)
    return {
        "later_events": len(scored),
        "total_events": total,
        "filtered": len(filtered),
        "animals_filtered": sum(s.animal for s in filtered),
        "animal_events": len(animals),
        "accepted": len(accepted),
        "accepted_correct": sum(s.correct for s in accepted),
        "automated": automated,
    }


def summarize(o: dict[str, Any]) -> dict[str, Any]:
    """Rates from counts; `o` may be one camera's outcome or a sum of several."""
    acc, ok = o["accepted"], o["accepted_correct"]
    an, lost = o["animal_events"], o["animals_filtered"]
    total = o["total_events"]
    return {
        **o,
        "review_reduction": o["automated"] * (1 - AUDIT_RATE) / total if total else 0.0,
        "precision": ok / acc if acc else None,
        "precision_ci": wilson(ok, acc) if acc else None,
        "retention": 1 - lost / an if an else None,
        "coverage_later": o["automated"] / o["later_events"] if o["later_events"] else 0.0,
    }


def add(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return {k: a.get(k, 0) + v for k, v in b.items()}


def leave_one_camera_out(
    per_camera: dict[str, tuple[list[Scored], int]], filter_empty: bool = True
) -> dict[str, Any]:
    cams: dict[str, Any] = {}
    pooled: dict[str, Any] = {}
    for cam, (scored, total) in per_camera.items():
        others = [s for c, (ss, _) in per_camera.items() if c != cam for s in ss]
        t_empty, t_species = choose(others, filter_empty)
        o = outcome(scored, total, t_empty, t_species)
        cams[cam] = {"t_empty": t_empty, "t_species": t_species, **summarize(o)}
        pooled = add(pooled, o)
    return {"pooled": summarize(pooled), "cameras": cams}


def nested_leave_one_camera_out(
    per_config: dict[str, dict[str, tuple[list[Scored], int]]],
    filter_empty: bool = True,
) -> dict[str, Any]:
    """Choose a configuration AND its thresholds without the held-out camera:
    for each camera, every configuration is judged on the other cameras
    (thresholds by the rule, then pooled review reduction there); the best is
    applied, with those thresholds, to the held-out camera."""
    cameras = next(iter(per_config.values())).keys()
    cams: dict[str, Any] = {}
    pooled: dict[str, Any] = {}
    for cam in cameras:
        best: tuple[float, str, float | None, float | None] | None = None
        for cfg, per_camera in per_config.items():
            others = {c: v for c, v in per_camera.items() if c != cam}
            t_empty, t_species = choose([s for ss, _ in others.values() for s in ss], filter_empty)
            inner: dict[str, Any] = {}
            for ss, total in others.values():
                inner = add(inner, outcome(ss, total, t_empty, t_species))
            reduction = summarize(inner)["review_reduction"]
            if best is None or reduction > best[0]:
                best = (reduction, cfg, t_empty, t_species)
        assert best is not None
        _, cfg, t_empty, t_species = best
        scored, total = per_config[cfg][cam]
        o = outcome(scored, total, t_empty, t_species)
        cams[cam] = {"config": cfg, "t_empty": t_empty, "t_species": t_species, **summarize(o)}
        pooled = add(pooled, o)
    return {"pooled": summarize(pooled), "cameras": cams}


def in_sample(
    per_camera: dict[str, tuple[list[Scored], int]], filter_empty: bool = True
) -> dict[str, Any]:
    """Thresholds chosen on all cameras' later events at once (optimistic)."""
    everything = [s for ss, _ in per_camera.values() for s in ss]
    t_empty, t_species = choose(everything, filter_empty)
    pooled: dict[str, Any] = {}
    for scored, total in per_camera.values():
        pooled = add(pooled, outcome(scored, total, t_empty, t_species))
    return {"t_empty": t_empty, "t_species": t_species, **summarize(pooled)}
