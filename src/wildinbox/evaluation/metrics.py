"""Image-level and event-level metrics.

Headline metrics are macro-F1 over supported classes, the rate of
animal-containing events wrongly filtered as empty, and the worst per-species
recall. Accuracy is reported only as context: with abundant empty frames a
model can score well while missing animals.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from wildinbox.class_map import EMPTY_CLASS
from wildinbox.datasets.events import Role
from wildinbox.policy.conservative import EventOutcome, Thresholds, decide_event
from wildinbox.schemas import Disposition


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def image_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], classes: Sequence[str]
) -> dict[str, Any]:
    """Per-class precision/recall/F1, macro-F1, confusion matrix (rows = true)."""
    idx = {c: i for i, c in enumerate(classes)}
    cm = [[0] * len(classes) for _ in classes]
    for t, p in zip(y_true, y_pred, strict=True):
        cm[idx[t]][idx[p]] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    for c, i in idx.items():
        tp = cm[i][i]
        support = sum(cm[i])
        predicted = sum(row[i] for row in cm)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted": predicted,
        }
    present = [c for c in classes if per_class[c]["support"]]
    n = len(y_true)
    animal_true = [t != EMPTY_CLASS for t in y_true]
    animal_pred = [p != EMPTY_CLASS for p in y_pred]
    animals = sum(animal_true)
    animal_hits = sum(t and p for t, p in zip(animal_true, animal_pred, strict=True))
    return {
        "n": n,
        "macro_f1": sum(float(per_class[c]["f1"]) for c in present) / len(present)
        if present
        else 0.0,
        "balanced_accuracy": (
            sum(float(per_class[c]["recall"]) for c in present) / len(present) if present else 0.0
        ),
        "accuracy_context_only": sum(cm[i][i] for i in range(len(classes))) / n if n else 0.0,
        "min_species_recall": min(
            (float(per_class[c]["recall"]) for c in present if c != EMPTY_CLASS), default=0.0
        ),
        "animal_image_recall": animal_hits / animals if animals else 0.0,
        "animal_images_predicted_empty": animals - animal_hits,
        "per_class": per_class,
        "confusion": {"classes": list(classes), "matrix": cm},
    }


@dataclass(frozen=True)
class ScoredEvent:
    event_id: str
    role: str
    label: str | None
    animal_present: bool
    frames: list[dict[str, float]]
    unfamiliar: tuple[bool, ...] | None = None  # per frame, when the score is in use


def event_metrics(
    events: Sequence[ScoredEvent],
    t: Thresholds,
    accept_species: tuple[str, ...] | None = None,
) -> tuple[dict[str, Any], list[EventOutcome]]:
    outcomes = [decide_event(e.frames, t, e.unfamiliar, accept_species) for e in events]
    n = len(events)
    animal = [e for e in events if e.animal_present]
    false_empty = sum(
        1
        for e, o in zip(events, outcomes, strict=True)
        if e.animal_present and o.disposition is Disposition.LIKELY_EMPTY
    )
    filtered = [
        (e, o)
        for e, o in zip(events, outcomes, strict=True)
        if o.disposition is Disposition.LIKELY_EMPTY
    ]
    accepted = [
        (e, o)
        for e, o in zip(events, outcomes, strict=True)
        if o.disposition is Disposition.SPECIES_IDENTIFIED
    ]
    correct = sum(1 for e, o in accepted if e.role == "supported_species" and e.label == o.label)
    wrong_by_role = Counter(
        e.role if e.role != "supported_species" else "other_supported_species"
        for e, o in accepted
        if not (e.role == "supported_species" and e.label == o.label)
    )
    by_species: dict[str, dict[str, int]] = {}
    for e, o in accepted:
        d = by_species.setdefault(o.label or "", {"accepted": 0, "correct": 0})
        d["accepted"] += 1
        d["correct"] += int(e.role == "supported_species" and e.label == o.label)
    review = n - len(filtered) - len(accepted)
    empties = sum(1 for e in events if e.role == Role.EMPTY)
    unsupported = sum(1 for e in events if e.role == "unsupported_animal")
    return {
        "thresholds": {"empty_filter": t.empty, "species_accept": t.species},
        "events": n,
        "animal_events": len(animal),
        "false_empty": false_empty,
        "false_empty_rate": false_empty / len(animal) if animal else 0.0,
        "false_empty_rate_ci95": wilson(false_empty, len(animal)),
        "animal_event_retention": 1 - (false_empty / len(animal)) if animal else 1.0,
        "filtered": len(filtered),
        "filtered_true_empty": sum(1 for e, _ in filtered if e.role == Role.EMPTY),
        "filtered_non_animal": sum(1 for e, _ in filtered if e.role == "non_animal"),
        "empty_events": empties,
        "accepted": len(accepted),
        "accepted_correct": correct,
        "accepted_precision": correct / len(accepted) if accepted else None,
        "accepted_precision_ci95": wilson(correct, len(accepted)) if accepted else None,
        "accepted_wrong_by_true_role": dict(wrong_by_role.most_common()),
        "unsupported_events": unsupported,
        "unsupported_accepted_as_known": wrong_by_role.get("unsupported_animal", 0),
        "accepted_by_species": dict(sorted(by_species.items())),
        "needs_review": review,
        "review_fraction": review / n if n else 0.0,
    }, outcomes
