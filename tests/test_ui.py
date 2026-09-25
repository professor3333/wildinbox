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
    today = date(2026, 1, 1)
    # An event at 02:00 belongs to the night that started the previous evening.
    assert last_night([datetime(2024, 5, 2, 2, 0)], today) == date(2024, 5, 1)
    assert last_night([datetime(2024, 5, 1, 23, 0)], today) == date(2024, 5, 1)
    # The newest capture was in daylight: open on the newest night that has events.
    starts = [datetime(2024, 5, 1, 23, 0), datetime(2024, 5, 3, 9, 30)]
    assert last_night(starts, today) == date(2024, 5, 1)
    assert last_night([], today) == date(2026, 1, 1)


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
        if params.get("disposition"):
            items = [e for e in items if e["decision"]["disposition"] == params["disposition"]]
        if params.get("audit") is not None:
            items = [e for e in items if e["decision"].get("audit_selected") == params["audit"]]
        if params.get("reviewed") is False:
            items = [e for e in items if not e["latest_review"]]
        if params.get("start_after"):
            items = [
                e
                for e in items
                if params["start_after"] <= e["start_at"] < params.get("start_before", "9999")
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
        images = [
            {
                "id": i,
                "filename": f"{i}.jpg",
                "frame_status": "completed",
                "prediction": {
                    "calibrated_probabilities": {"raccoon": 0.7, "empty": 0.3},
                    "class_probabilities": {},
                },
            }
            for i in e["image_ids"]
        ]
        reviews = (
            [
                {
                    **e["latest_review"],
                    "created_at": "2024-05-02T09:00:00",
                    "suggested_label": e["decision"]["suggested_label"],
                    "note": None,
                }
            ]
            if e["latest_review"]
            else []
        )
        return {**e, "images": images, "reviews": reviews}

    def batch(self, batch_id: str) -> dict[str, Any]:
        return {
            "created_at": "2024-05-02T08:00:00",
            "status": "completed_with_errors",
            "counts": {"images": 7, "events": 3},
            "progress": {"images_scored": 6, "images_to_score": 6, "finished": True},
            "failures": [
                {"filename": "broken.jpg", "stage": "validation", "error": "unreadable image"}
            ],
        }

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


def test_monitoring_page_separates_signals_from_measured_accuracy() -> None:
    api = FakeApi()
    ops = {
        "window_days": 7,
        "jobs": {"succeeded": 2},
        "queued_waiting": 0,
        "oldest_queued_seconds": 0.0,
        "retrying": 0,
        "stale_leases": 1,
        "failed_jobs_in_window": [],
        "files_uploaded": 10,
        "files_unreadable": 1,
        "unreadable_rate": 0.1,
        "images_scored": 9,
        "frames_failed": 0,
        "processing_error_rate": 0.0,
        "images_scored_last_24h": 9,
        "cost_by_release": {"rel": {"jobs": 2, "images": 9, "seconds_per_1000_images": 80.0}},
        "api_latency": {"GET /events": {"requests": 3, "p50_ms": 10.0, "p95_ms": 20.0}},
    }
    api.monitoring = lambda: {  # type: ignore[attr-defined]
        "operations": ops,
        "signals": {
            "north": {
                "events": 3,
                "needs_review_share": 1.0,
                "low_confidence_share": 0.5,
                "night_share": 0.3,
                "comparison": None,
            }
        },
        "accuracy": {
            "by_camera": {
                "north": {
                    "events": 3,
                    "reviewed": 1,
                    "review_coverage": 0.33,
                    "correction_rate": None,
                    "correction_rate_ci95": None,
                    "unresolved": 0,
                    "reviewers": {"ranger": 1},
                }
            },
            "by_release": {},
        },
        "alerts": [
            {
                "level": "critical",
                "area": "operations",
                "camera": None,
                "message": "a worker stopped renewing its lease",
                "value": 1,
                "limit": 0,
            }
        ],
        "notes": {},
    }
    at = _app(api)
    at.sidebar.radio(key="page").set_value("Monitoring").run()
    assert not at.exception
    assert any("stopped renewing its lease" in e.value for e in at.error)
    headers = [h.value for h in at.subheader]
    assert headers == [
        "Alerts",
        "Operations",
        "Signals per camera (no labels needed)",
        "Accuracy measured from reviews",
    ]


def _auto_api() -> FakeApi:
    api = FakeApi()
    api.items.append(
        {
            **_event("f1", "empty", disposition="likely_empty"),
            "decision": {
                "disposition": "likely_empty",
                "suggested_label": "empty",
                "confidence": 0.97,
                "reasons": [],
                "audit_selected": True,
                "audit_rule": "sha256-uniform(rate=0.05, seed=s)",
            },
        }
    )
    return api


def test_filtered_view_finds_and_recovers_a_filtered_event() -> None:
    api = _auto_api()
    at = _app(api)
    at.sidebar.text_input(key="reviewer").input("ranger").run()
    at.sidebar.radio(key="page").set_value("Automatically filtered").run()
    assert not at.exception
    assert any("⚙️ empty" in m.value and "audit sample" in m.value for m in at.markdown)
    at.selectbox(key="pick-f1").set_value("bobcat").run()
    at.button(key="save-f1").click().run()
    assert api.reviews[-1] == ("f1", "ranger", "corrected", "bobcat")


def test_audit_queue_timeline_batches_details_and_guide_render() -> None:
    api = _auto_api()
    api.items[0]["latest_review"] = {
        "outcome": "confirmed",
        "confirmed_label": "raccoon",
        "reviewer": "ann",
    }
    at = _app(api)
    at.sidebar.radio(key="page").set_value("Audit queue").run()
    assert not at.exception and any("f1" in (b.key or "") for b in at.button)
    at.sidebar.radio(key="page").set_value("Timeline").run()
    text = " ".join(m.value for m in at.markdown)
    assert "✅ **raccoon**" in text and "🤖 suggested: empty" in text and "⚙️ empty" in text
    at.sidebar.radio(key="page").set_value("Batches").run()
    assert not at.exception and at.dataframe
    at.sidebar.radio(key="page").set_value("Review queue").run()
    at.radio(key="status").set_value("Reviewed").run()
    at.toggle(key="detail-e1").set_value(True).run()
    assert not at.exception
    assert any("Review history" in m.value for m in at.markdown) and at.dataframe
    at.sidebar.radio(key="page").set_value("Getting started").run()
    assert any("Accept" in m.value and "other species" in m.value for m in at.markdown)
