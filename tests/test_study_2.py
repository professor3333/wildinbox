"""Review study 2 (configs/study/review_study_2.yaml): one reviewer, many
sessions, plan-stored suggestions, and event-level analysis."""

from __future__ import annotations

import uuid
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from wildinbox.api.app import create_app
from wildinbox.study.design import (
    ASSISTED,
    SINGLE_REVIEWER,
    assignment,
    build_sets,
    plan_assignment,
    single_reviewer_design,
)
from wildinbox.study.single import analyze_single

from .test_app import NoopDispatcher, database_url, settings  # noqa: F401
from .test_study import StudyApi, _truth
from .test_ui import APP, CLASSES

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = yaml.safe_load((REPO_ROOT / "configs/study/review_study_2.yaml").read_text())
SCHEDULE = PROTOCOL["schedule"]["order"]


def _design(sets: dict[str, list[str]], per_session: int, shown: bool = True) -> dict[str, Any]:
    return {
        "kind": SINGLE_REVIEWER,
        "schedule": SCHEDULE,
        "events_per_session": per_session,
        "suggestions": {
            e: {"label": "raccoon", "confidence": 0.93, "shown": shown} for e in sets["B"]
        },
    }


# ------------------------------------------------------------------ design


def test_protocol_schedule_is_balanced() -> None:
    assert Counter(SCHEDULE) == {"grouped": 4, ASSISTED: 4}
    halves = SCHEDULE[:4], SCHEDULE[4:]
    assert all(Counter(h) == {"grouped": 2, ASSISTED: 2} for h in halves)


def test_sessions_take_each_set_in_order_and_see_every_event_once() -> None:
    sets = build_sets(_truth(60), 24, 1, seed=1)
    blocks = single_reviewer_design(sets, SCHEDULE, 6)
    assert [b["condition"] for b in blocks] == SCHEDULE
    grouped = [e for b in blocks if b["condition"] == "grouped" for e in b["events"]]
    assisted = [e for b in blocks if b["condition"] == ASSISTED for e in b["events"]]
    assert grouped == sets["A"] and assisted == sets["B"]
    with pytest.raises(ValueError):
        single_reviewer_design(sets, SCHEDULE, 7)  # 4 sessions x 7 > 24 events


def test_study_1_plans_keep_their_crossover() -> None:
    sets = build_sets(_truth(120), 40, 3, seed=1)
    assert plan_assignment(sets, None, 3) == assignment(sets, 3)


# ------------------------------------------------------------------ analysis


def _export(
    per_session: int = 20,
    speedup: float = 1.0,
    assisted_accuracy: float = 1.0,
    sessions: int = 8,
    gap_minutes: float = 90,
) -> dict[str, Any]:
    truth = _truth(2 * 4 * per_session + 24)  # spare events: labels pair unevenly
    sets = build_sets(truth, 4 * per_session, 1, seed=2)
    design = _design(sets, per_session)
    start = datetime(2026, 10, 1, 9, tzinfo=UTC)
    trials = []
    for b in plan_assignment(sets, design, 0)["blocks"][:sessions]:
        t0 = start + timedelta(minutes=(b["block"] - 1) * (gap_minutes + 30))
        for k, e in enumerate(b["events"]):
            assisted = b["condition"] == ASSISTED
            wrong = assisted and k < per_session * (1 - assisted_accuracy)
            label = "opossum" if wrong and truth[e] != "opossum" else truth[e]
            seconds = 6.0 + (k % 5) * 0.5 - (speedup if assisted else 0.0)
            shown = t0 + timedelta(seconds=10 * k)
            trials.append(
                {
                    "participant": "author",
                    "block": b["block"],
                    "condition": b["condition"],
                    "event_id": e,
                    "label": label,
                    "seconds": seconds,
                    "shown_at": shown.isoformat(),
                    "decided_at": (shown + timedelta(seconds=seconds)).isoformat(),
                    "displayed": (
                        {"suggested_label": "raccoon", "confidence": 0.93} if assisted else None
                    ),
                }
            )
    return {"plan": {"classes": CLASSES, "truth": truth, "design": design}, "trials": trials}


def test_faster_without_accuracy_loss_is_demonstrated_for_one_reviewer() -> None:
    r = analyze_single(_export(), PROTOCOL)
    assert r["complete"] and r["success"]
    assert r["median_seconds_difference"] == pytest.approx(-1.0)
    assert r["median_seconds_difference_ci95"][1] < 0
    assert r["verdict"].startswith("for this one reviewer") and "not evidence" in r["verdict"]


def test_an_accuracy_loss_beyond_the_margin_is_not_a_success() -> None:
    r = analyze_single(_export(assisted_accuracy=0.7), PROTOCOL)
    assert r["accuracy_difference"] < -0.1 and not r["success"]
    assert r["verdict"] == "not demonstrated for this reviewer"
    a = r["assisted"]  # raccoon is shown everywhere, so it is wrong on other events
    assert a["wrong_suggestions_shown"] > 0 and a["events_with_suggestion_shown"] == 80


def test_an_incomplete_schedule_and_short_breaks_are_reported() -> None:
    r = analyze_single(_export(sessions=5, gap_minutes=20), PROTOCOL)
    assert not r["complete"] and not r["success"]
    assert r["verdict"].startswith("incomplete")
    assert len(r["session_gaps_under_minimum"]) == 4


def test_following_a_wrong_suggestion_is_counted() -> None:
    export = _export()
    for t in export["trials"]:
        if t["condition"] == ASSISTED:
            t["label"] = "raccoon"  # always takes the suggestion
    a = analyze_single(export, PROTOCOL)["assisted"]
    assert a["wrong_suggestions_followed"] == a["wrong_suggestions_shown"] > 0


# ------------------------------------------------------------------ API


def test_single_reviewer_plan_serves_suggestions_resumes_and_exports(
    settings: object,  # noqa: F811
) -> None:
    truth = _truth(60)
    sets = build_sets(truth, 24, 1, seed=3)
    design = _design(sets, 6)
    design["suggestions"][sets["B"][1]]["shown"] = False
    with TestClient(create_app(settings, dispatcher=NoopDispatcher())) as c:  # type: ignore[arg-type]
        body = {"name": "study 2", "protocol_sha256": "0" * 64, "sets": sets, "truth": truth}
        plan = c.post("/study/plans", json={**body, "design": design}).json()["id"]
        me = c.post(f"/study/plans/{plan}/participants", json={"code": "author"}).json()
        assert [b["condition"] for b in me["blocks"]] == SCHEDULE
        assert set(me["suggestions"]) <= set(sets["B"]) | set(sets["practice"])
        assert "truth" not in me and me["logged"] == []
        first = me["blocks"][0]  # grouped
        now = datetime.now(UTC).isoformat()
        trial = {
            "participant": "author",
            "block": 1,
            "condition": first["condition"],
            "event_id": first["events"][0],
            "label": truth[first["events"][0]],
            "seconds": 3.0,
            "interactions": 1,
            "shown_at": now,
            "decided_at": now,
        }
        assert c.post(f"/study/plans/{plan}/trials", json=trial).status_code == 201
        assisted = me["blocks"][1]
        for e in assisted["events"][:2]:
            t = {**trial, "block": 2, "condition": ASSISTED, "event_id": e, "label": "raccoon"}
            assert c.post(f"/study/plans/{plan}/trials", json=t).status_code == 201
        late = {**trial, "block": 8, "condition": ASSISTED, "event_id": first["events"][1]}
        assert c.post(f"/study/plans/{plan}/trials", json=late).status_code == 422
        again = c.post(f"/study/plans/{plan}/participants", json={"code": "author"}).json()
        assert len(again["logged"]) == 3  # a returning reviewer resumes after these
        export = c.get(f"/study/plans/{plan}/export").json()
        shown = {t["event_id"]: t["displayed"] for t in export["trials"]}
        assert shown[assisted["events"][0]] == {
            "suggested_label": "raccoon",
            "confidence": 0.93,
            "source": "plan",
        }
        assert shown[assisted["events"][1]] is None  # not shown: below the threshold
        assert export["plan"]["design"]["kind"] == SINGLE_REVIEWER
        bad = c.post("/study/plans", json={**body, "design": {**design, "kind": "x"}})
        assert bad.status_code == 422


# ------------------------------------------------------------------ UI


class Study2Api(StudyApi):
    def __init__(self) -> None:
        super().__init__()
        e = [str(uuid.UUID(int=i + 1)) for i in range(4)]
        self.plan = {
            "arm": 0,
            "classes": CLASSES,
            "practice": [
                {"condition": "grouped", "events": []},
                {"condition": ASSISTED, "events": []},
            ],
            "blocks": [
                {"block": 1, "condition": ASSISTED, "set": "B", "events": [e[0], e[1]]},
                {"block": 2, "condition": "grouped", "set": "A", "events": [e[2]]},
                {"block": 3, "condition": "grouped", "set": "A", "events": [e[3]]},
            ],
            "suggestions": {
                e[0]: {"label": "raccoon", "confidence": 0.93, "shown": True},
                e[1]: {"label": "bobcat", "confidence": 0.4, "shown": False},
            },
            "logged": [],
        }


def _join(api: StudyApi) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.session_state["client"] = api
    at.run()
    at.sidebar.radio(key="page").set_value("Study").run()
    at.text_input(key="study-plan").input("plan-2").run()
    at.text_input(key="study-code").input("author").run()
    at.button(key="study-join").click().run()
    return at


def test_assisted_shows_only_validated_suggestions_without_an_accept_button() -> None:
    api = Study2Api()
    at = _join(api)
    at.button(key="study-consent").click().run()
    e0, e1 = api.plan["blocks"][0]["events"]
    assert any("Suggestion: **raccoon**" in m.value for m in at.markdown)
    assert at.button(key=f"study-{e0}-raccoon").proto.type == "primary"  # highlighted
    assert at.button(key=f"study-{e0}-opossum").proto.type == "secondary"
    assert not [b for b in at.button if b.key and b.key.startswith("study-accept")]
    at.button(key=f"study-{e0}-raccoon").click().run()
    assert any("No suggestion" in c.value for c in at.caption)  # e1: below the threshold
    assert not any("Suggestion:" in m.value for m in at.markdown)
    at.button(key=f"study-{e1}-opossum").click().run()
    at.radio(key="study-rating-1").set_value(3).run()
    at.button(key="study-rated-1").click().run()
    assert any("Session 1 done" in s.value for s in at.success)  # a break, not the next session
    assert at.button(key="study-continue-1").disabled
    assert [(t["condition"], t["label"]) for t in api.trials] == [
        (ASSISTED, "raccoon"),
        (ASSISTED, "opossum"),
    ]
    assert not at.exception


def test_a_returning_reviewer_resumes_at_the_next_unlogged_event() -> None:
    api = Study2Api()
    api.plan["logged"] = api.plan["blocks"][0]["events"]  # session 1 already done
    at = _join(api)
    assert not [b for b in at.button if b.key == "study-consent"]  # no consent or practice again
    assert any("Session 2 of 3" in c.value for c in at.caption)
    assert not at.exception
