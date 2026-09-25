"""Study design (configs/study/review_study.yaml): event sets, counterbalancing,
and scoring. Pure functions, so the design is tested without a database."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

CONDITIONS = ("grouped", "suggested")
# arm -> ((condition, set), (condition, set)); participant n gets arm n % 4.
ARMS: tuple[tuple[tuple[str, str], tuple[str, str]], ...] = (
    (("grouped", "A"), ("suggested", "B")),
    (("suggested", "A"), ("grouped", "B")),
    (("grouped", "B"), ("suggested", "A")),
    (("suggested", "B"), ("grouped", "A")),
)
OTHER = "other"
CANT_TELL = "can't tell"


def _order(key: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def build_sets(
    truth: dict[str, str | None], per_set: int, practice_per_condition: int, seed: int
) -> dict[str, list[str]]:
    """Split events into practice, A, and B. Events are shuffled
    deterministically within each ground-truth label; practice events are taken
    first (labels in turn) and never appear in A or B; then label-matched pairs
    are split between A and B, so both sets have exactly the same label mix."""
    by_label: dict[str, list[str]] = defaultdict(list)
    for eid, label in truth.items():
        by_label[str(label)].append(eid)
    for label in by_label:
        by_label[label].sort(key=lambda e: _order(e, seed))
    # Interleave labels so any prefix is balanced, then deal.
    queue: list[str] = []
    labels = sorted(by_label)
    while any(by_label[lb] for lb in labels):
        for lb in labels:
            if by_label[lb]:
                queue.append(by_label[lb].pop(0))
    n_practice = 2 * practice_per_condition
    if len(queue) < n_practice + 2 * per_set:
        raise ValueError(f"{len(queue)} events; the protocol needs {n_practice + 2 * per_set}")
    practice = queue[:n_practice]
    # Deal PAIRS within each label (one event to A, one to B), taking pairs
    # from the labels in turn, so both sets get exactly the same label mix.
    remaining: dict[str, list[str]] = defaultdict(list)
    for eid in queue[n_practice:]:
        remaining[str(truth[eid])].append(eid)
    pairs: list[tuple[str, str]] = []
    while any(len(remaining[lb]) >= 2 for lb in labels):
        for lb in labels:
            if len(remaining[lb]) >= 2:
                pairs.append((remaining[lb].pop(0), remaining[lb].pop(0)))
    if len(pairs) < per_set:
        raise ValueError(f"only {len(pairs)} label-matched pairs; the protocol needs {per_set}")
    chosen = pairs[:per_set]
    return {"practice": practice, "A": [a for a, _ in chosen], "B": [b for _, b in chosen]}


def assignment(sets: dict[str, list[str]], arm: int) -> dict[str, Any]:
    """What a participant does: practice in both conditions (in the order of
    their blocks), then two timed blocks."""
    blocks = ARMS[arm % len(ARMS)]
    half = len(sets["practice"]) // 2
    practice = {
        blocks[0][0]: sets["practice"][:half],
        blocks[1][0]: sets["practice"][half:],
    }
    return {
        "arm": arm % len(ARMS),
        "practice": [{"condition": c, "events": practice[c]} for c, _ in blocks],
        "blocks": [
            {"block": i + 1, "condition": c, "set": s, "events": sets[s]}
            for i, (c, s) in enumerate(blocks)
        ],
    }


def category(label: str | None, classes: list[str]) -> str:
    """Scoring category: a supported class, 'other' (any unsupported species),
    or "can't tell"."""
    if label is None:
        return CANT_TELL
    label = label.strip().lower()
    return label if label in classes else OTHER


def correct(choice: str | None, truth: str | None, classes: list[str]) -> bool | None:
    """None when the event has no single ground-truth label (mixed species)."""
    if truth is None:
        return None
    return category(choice, classes) == category(truth, classes)
