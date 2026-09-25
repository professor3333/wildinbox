from __future__ import annotations

import io
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from wildinbox.ui.logic import (
    current_label,
    is_visitor,
    last_night,
    night_window,
    representative_frame,
    review_for,
)

APP = Path(__file__).resolve().parents[1] / "src/wildinbox/ui/app.py"
CLASSES = ["empty", "bobcat", "cat", "coyote", "dog", "opossum", "rabbit", "raccoon"]


# ------------------------------------------------------------------ logic


@pytest.mark.parametrize(
    ("suggested", "chosen", "expected"),
    [
        ("raccoon", "raccoon", ("confirmed", "raccoon")),
        ("raccoon", "opossum", ("corrected", "opossum")),
        ("raccoon", None, ("unresolved", None)),
        (None, "skunk", ("corrected", "skunk")),
    ],
)
def test_review_outcome_follows_the_choice(
    suggested: str | None, chosen: str | None, expected: tuple[str, str | None]
) -> None:
    assert review_for(suggested, chosen) == expected


def test_night_window_and_default_night() -> None:
    start, end = night_window(date(2024, 5, 1))
    assert (start, end) == (datetime(2024, 5, 1, 18), datetime(2024, 5, 2, 6))
    # An event at 02:00 belongs to the night that started the previous evening.
    assert last_night(datetime(2024, 5, 2, 2, 0), date(2026, 1, 1)) == date(2024, 5, 1)
    assert last_night(datetime(2024, 5, 1, 23, 0), date(2026, 1, 1)) == date(2024, 5, 1)


def _event(
    eid: str, suggestion: str | None, review: dict[str, Any] | None = None, **kw: Any
) -> dict[str, Any]:
    return {
        "id": eid,
        "camera_id": "north",
        "start_at": kw.get("start_at", "2024-05-01T21:00:00"),
        "image_ids": kw.get("image_ids", [f"{eid}-a", f"{eid}-b"]),
        "decision": {
            "disposition": kw.get("disposition", "needs_review"),
            "suggested_label": suggestion,
            "confidence": 0.62,
            "reasons": ["automation_disabled"],
        },
        "latest_review": review,
    }


def test_visitors_use_the_current_label() -> None:
    assert is_visitor(_event("a", "raccoon"))
    assert not is_visitor(_event("b", "empty"))
    corrected = _event(
        "c", "empty", {"outcome": "corrected", "confirmed_label": "skunk", "reviewer": "r"}
    )
    assert is_visitor(corrected) and current_label(corrected) == ("skunk", "reviewed")
    emptied = _event(
        "d", "raccoon", {"outcome": "corrected", "confirmed_label": "empty", "reviewer": "r"}
    )
    assert not is_visitor(emptied)
    unresolved = _event(
        "e", "empty", {"outcome": "unresolved", "confirmed_label": None, "reviewer": "r"}
    )
    assert is_visitor(unresolved)


def test_representative_frame_best_shows_the_suggestion() -> None:
    detail = {
        "decision": {"suggested_label": "raccoon"},
        "images": [
            {
                "id": "1",
                "prediction": {
                    "calibrated_probabilities": {"raccoon": 0.3},
                    "class_probabilities": {},
                },
            },
            {
                "id": "2",
                "prediction": {
                    "calibrated_probabilities": {"raccoon": 0.8},
                    "class_probabilities": {},
                },
            },
            {"id": "3", "prediction": None},
        ],
    }
    frame = representative_frame(detail)
    assert frame is not None and frame["id"] == "2"
    assert representative_frame({"images": []}) is None


# ------------------------------------------------------------------ app


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 6), (90, 120, 60)).save(buf, "PNG")
    return buf.getvalue()


class FakeApi:
    def __init__(self) -> None:
        self.reviews: list[tuple[str, str, str, str | None]] = []
        self.items = [
            _event("e1", "raccoon"),
            _event("e2", "empty", start_at="2024-05-02T02:00:00"),
            _event("e3", "coyote", start_at="2024-05-01T12:00:00"),  # daytime
        ]

    def version(self) -> dict[str, Any]:
        return {
            "active_release": {
                "id": "rel",
                "is_test": False,
                "policy_version": "p",
                "class_names": CLASSES,
            }
        }

    def batches(self) -> list[dict[str, Any]]:
        return [{"id": "b1", "created_at": "2024-05-02T08:00:00", "images": 6, "events": 3}]

    def events(self, **params: Any) -> dict[str, Any]:
        items = list(self.items)
        if params.get("reviewed") is False:
            items = [e for e in items if not e["latest_review"]]
        if params.get("start_after"):
            items = [
                e for e in items if params["start_after"] <= e["start_at"] < params["start_before"]
            ]
        off, lim = params.get("offset", 0), params.get("limit", 100)
        page = items[off : off + lim]
        return {
            "events": page,
            "total": len(items),
            "next_offset": off + lim if off + lim < len(items) else None,
        }

    def event(self, eid: str) -> dict[str, Any]:
        e = next(e for e in self.items if e["id"] == eid)
        return {**e, "images": [{"id": i, "prediction": None} for i in e["image_ids"]]}

    def thumbnail(self, image_id: str, size: int = 320) -> bytes:
        return _png()

    def review(
        self, eid: str, reviewer: str, outcome: str, label: str | None, note: str | None = None
    ) -> dict[str, Any]:
        self.reviews.append((eid, reviewer, outcome, label))
        e = next(e for e in self.items if e["id"] == eid)
        e["latest_review"] = {"outcome": outcome, "confirmed_label": label, "reviewer": reviewer}
        return {}


def _app(api: FakeApi) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.session_state["client"] = api
    at.run()
    assert not at.exception, at.exception
    return at


def test_review_queue_needs_a_name_then_records_each_kind_of_review() -> None:
    api = FakeApi()
    at = _app(api)
    assert at.button(key="accept-e1").disabled  # no reviewer name yet
    at.sidebar.text_input(key="reviewer").input("ranger").run()

    at.button(key="accept-e1").click().run()
    assert api.reviews[-1] == ("e1", "ranger", "confirmed", "raccoon")

    at.selectbox(key="pick-e2").set_value("other species…").run()
    at.text_input(key="other-e2").input("Skunk").run()
    at.button(key="save-e2").click().run()
    assert api.reviews[-1] == ("e2", "ranger", "corrected", "skunk")

    at.selectbox(key="pick-e3").set_value("can't tell").run()
    at.button(key="save-e3").click().run()
    assert api.reviews[-1] == ("e3", "ranger", "unresolved", None)
    assert not at.exception
    assert any("Nothing here" in s.value for s in at.success)


def test_last_nights_visitors_show_animal_events_of_that_night_only() -> None:
    api = FakeApi()
    at = _app(api)
    at.sidebar.radio(key="page").set_value("Last night's visitors").run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "raccoon" in text  # 21:00, suggested animal
    assert "coyote" not in text  # daytime
    captions = " ".join(c.value for c in at.caption)
    assert "1 animal event(s) of 2" in captions  # the 02:00 event is empty
