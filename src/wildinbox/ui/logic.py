"""Decisions the review interface makes, kept free of Streamlit so they are tested."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any

from wildinbox.class_map import EMPTY_CLASS

REASON_TEXT = {
    "low_confidence": "model is unsure",
    "conflicting_frames": "frames disagree",
    "possible_unknown": "may be an unsupported species",
    "processing_failure": "a frame could not be processed",
    "species_not_validated": "species not validated for automation",
    "automation_disabled": "automation is off",
}


def review_for(suggested: str | None, chosen: str | None) -> tuple[str, str | None]:
    """The review outcome for a reviewer's choice. None means "can't tell"."""
    if chosen is None:
        return "unresolved", None
    if chosen == suggested:
        return "confirmed", chosen
    return "corrected", chosen


def night_window(day: date, start_hour: int = 18, end_hour: int = 6) -> tuple[datetime, datetime]:
    """The night that begins on `day` (camera local time)."""
    start = datetime.combine(day, time(start_hour))
    return start, datetime.combine(day + timedelta(days=1), time(end_hour))


def last_night(latest_event_start: datetime | None, today: date) -> date:
    """The night to show by default: the one containing the newest event, so
    an old memory card still opens on its own last night."""
    ref = latest_event_start or datetime.combine(today, time(12))
    return (ref - timedelta(hours=12)).date()


def current_label(event: dict[str, Any]) -> tuple[str | None, str]:
    """(label, source) where source is reviewed, unresolved, automatic, or suggested."""
    review = event.get("latest_review")
    if review:
        if review["outcome"] == "unresolved":
            return None, "unresolved"
        return review["confirmed_label"], "reviewed"
    decision = event.get("decision") or {}
    if decision.get("disposition") in ("likely_empty", "species_identified"):
        return decision.get("suggested_label"), "automatic"
    return decision.get("suggested_label"), "suggested"


def is_visitor(event: dict[str, Any]) -> bool:
    """An animal-containing event by its current label (reviewed or suggested)."""
    label, source = current_label(event)
    if source == "unresolved":
        return True  # a reviewer saw something but could not name it
    return label is not None and label != EMPTY_CLASS


def representative_frame(detail: dict[str, Any]) -> dict[str, Any] | None:
    """The frame that best shows the event's suggested label."""
    images: list[dict[str, Any]] = detail.get("images") or []
    label = (detail.get("decision") or {}).get("suggested_label")
    scored: list[tuple[dict[str, float], dict[str, Any]]] = [
        (
            i["prediction"].get("calibrated_probabilities")
            or i["prediction"]["class_probabilities"],
            i,
        )
        for i in images
        if i.get("prediction")
    ]
    if label and scored:
        return max(scored, key=lambda pi: pi[0].get(label, 0.0))[1]
    if scored:
        return scored[0][1]
    return images[0] if images else None


def label_choices(class_names: list[str]) -> list[str]:
    """Species first, then empty; free-text 'other' and 'can't tell' are added by the UI."""
    return [c for c in class_names if c != EMPTY_CLASS] + [EMPTY_CLASS]
