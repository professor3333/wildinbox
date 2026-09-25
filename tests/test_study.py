from __future__ import annotations

import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from wildinbox.api.app import create_app
from wildinbox.study.analysis import analyze
from wildinbox.study.design import ARMS, assignment, build_sets, category, correct

from .test_app import NoopDispatcher, database_url, settings  # noqa: F401
from .test_ui import APP, CLASSES, _png

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = yaml.safe_load((REPO_ROOT / "configs/study/review_study.yaml").read_text())


def _truth(n: int) -> dict[str, str | None]:
    labels = ["opossum", "raccoon", "empty", "bobcat", "skunk", None]
    return {str(uuid.UUID(int=i + 1)): labels[i % len(labels)] for i in range(n)}


# ------------------------------------------------------------------ design


def test_sets_are_balanced_disjoint_and_deterministic() -> None:
    truth = _truth(120)
    sets = build_sets(truth, 40, 3, seed=1)
    assert [len(sets[k]) for k in ("practice", "A", "B")] == [6, 40, 40]
    ids = [e for v in sets.values() for e in v]
    assert len(ids) == len(set(ids))
    mix_a = Counter(str(truth[e]) for e in sets["A"])
    mix_b = Counter(str(truth[e]) for e in sets["B"])
    assert mix_a == mix_b and len(mix_a) == 6  # identical label mix, every label present
    assert build_sets(truth, 40, 3, seed=1) == sets
    with pytest.raises(ValueError):
        build_sets(_truth(50), 40, 3, seed=1)


def test_arms_counterbalance_order_and_set_pairing() -> None:
    firsts = Counter(arm[0][0] for arm in ARMS)
    pairings = Counter((c, s) for arm in ARMS for c, s in arm)
    assert firsts == {"grouped": 2, "suggested": 2}
    assert set(pairings.values()) == {2} and len(pairings) == 4
    sets = build_sets(_truth(120), 40, 3, seed=1)
    a = assignment(sets, 5)
    assert a["arm"] == 1 and [b["condition"] for b in a["blocks"]] == ["suggested", "grouped"]
    assert [p["condition"] for p in a["practice"]] == ["suggested", "grouped"]


def test_scoring_categories() -> None:
    assert category("Raccoon", CLASSES) == "raccoon"
    assert category("fox", CLASSES) == "other" and category(None, CLASSES) == "can't tell"
    assert correct("fisher", "skunk", CLASSES) is True  # both unsupported: "other"
    assert correct(None, "empty", CLASSES) is False
    assert correct("cat", None, CLASSES) is None  # mixed species: not scored


# ------------------------------------------------------------------ analysis


def _export(n_participants: int, speedup: float, per_set: int = 40) -> dict[str, Any]:
    truth = _truth(3 * per_set)
    sets = build_sets(truth, per_set, 3, seed=2)
    trials = []
    for p in range(n_participants):
        for b in assignment(sets, p)["blocks"]:
            for k, e in enumerate(b["events"]):
                base = 8.0 + (p % 3) + (k % 5) * 0.4
                seconds = base - speedup if b["condition"] == "suggested" else base
                trials.append(
                    {
                        "participant": f"P{p}",
                        "block": b["block"],
                        "condition": b["condition"],
                        "event_id": e,
                        "label": truth[e],
                        "seconds": seconds,
                        "interactions": 1,
                    }
                )
    return {"plan": {"classes": CLASSES, "truth": truth}, "trials": trials, "ratings": []}


def test_analysis_detects_a_real_speedup_with_enough_participants() -> None:
    result = analyze(_export(8, speedup=2.0), PROTOCOL)
    s = result["summary"]
    assert s["participants_analysed"] == 8 and s["enough_participants"]
    assert s["median_seconds_difference"] == pytest.approx(-2.0)
    assert s["median_seconds_difference_ci95"][1] < 0
    assert s["share_faster_with_suggestions"] == 1.0
    assert s["mean_accuracy"] == {"grouped": 1.0, "suggested": 1.0}
    assert s["verdict"].startswith("suggestions made review faster without")


def test_analysis_stays_descriptive_below_the_minimum_and_applies_exclusions() -> None:
    small = analyze(_export(3, speedup=2.0), PROTOCOL)
    assert small["summary"]["verdict"].startswith("descriptive only")
    export = _export(9, speedup=0.0)
    for t in export["trials"]:
        if t["participant"] == "P0":
            t["seconds"] = 999.0  # interrupted throughout
    export["trials"] = [
        t for t in export["trials"] if not (t["participant"] == "P1" and t["block"] == 2)
    ]
    s = analyze(export, PROTOCOL)["summary"]
    reasons = {e["participant"]: e["reason"] for e in s["participants_excluded"]}
    assert "over the time limit" in reasons["P0"] and "did not finish" in reasons["P1"]
    assert s["participants_analysed"] == 7 and s["verdict"].startswith("descriptive only")


# ------------------------------------------------------------------ API


def test_study_api_assigns_arms_validates_trials_and_exports(settings: object) -> None:  # noqa: F811
    truth = _truth(120)
    sets = build_sets(truth, 40, 3, seed=3)
    with TestClient(create_app(settings, dispatcher=NoopDispatcher())) as c:  # type: ignore[arg-type]
        plan = c.post(
            "/study/plans",
            json={"name": "pilot", "protocol_sha256": "0" * 64, "sets": sets, "truth": truth},
        ).json()["id"]
        arms = [
            c.post(f"/study/plans/{plan}/participants", json={"code": f"P{i}"}).json()["arm"]
            for i in range(5)
        ]
        assert arms == [0, 1, 2, 3, 0]
        again = c.post(f"/study/plans/{plan}/participants", json={"code": "P1"}).json()
        assert again["arm"] == 1 and again["classes"]
        event = again["blocks"][0]["events"][0]
        now = datetime.now(UTC).isoformat()
        trial = {
            "participant": "P1",
            "block": 1,
            "condition": "suggested",
            "event_id": event,
            "label": "raccoon",
            "seconds": 4.2,
            "interactions": 1,
            "shown_at": now,
            "decided_at": now,
        }
        assert c.post(f"/study/plans/{plan}/trials", json=trial).status_code == 201
        assert c.post(f"/study/plans/{plan}/trials", json=trial).json() == {"duplicate": True}
        wrong = {**trial, "condition": "grouped", "event_id": again["blocks"][0]["events"][1]}
        assert c.post(f"/study/plans/{plan}/trials", json=wrong).status_code == 422
        assert (
            c.post(
                f"/study/plans/{plan}/trials", json={**trial, "participant": "nobody"}
            ).status_code
            == 404
        )
        rating = {"participant": "P1", "block": 1, "condition": "suggested", "difficulty": 2}
        assert c.post(f"/study/plans/{plan}/ratings", json=rating).status_code == 201
        assert (
            c.post(f"/study/plans/{plan}/ratings", json={**rating, "difficulty": 4}).status_code
            == 201
        )
        export = c.get(f"/study/plans/{plan}/export").json()
        assert len(export["trials"]) == 1 and export["ratings"][0]["difficulty"] == 4
        assert export["plan"]["truth"][event] == truth[event]
        bad = {**sets, "B": sets["B"] + [sets["A"][0]]}
        r = c.post(
            "/study/plans",
            json={"name": "x", "protocol_sha256": "0" * 64, "sets": bad, "truth": truth},
        )
        assert r.status_code == 422


# ------------------------------------------------------------------ UI


class StudyApi:
    def __init__(self) -> None:
        self.trials: list[dict[str, Any]] = []
        self.ratings: list[dict[str, Any]] = []
        e = [str(uuid.UUID(int=i + 1)) for i in range(4)]
        self.plan = {
            "arm": 1,
            "classes": CLASSES,
            "practice": [
                {"condition": "suggested", "events": [e[0]]},
                {"condition": "grouped", "events": [e[1]]},
            ],
            "blocks": [
                {"block": 1, "condition": "suggested", "set": "A", "events": [e[2]]},
                {"block": 2, "condition": "grouped", "set": "B", "events": [e[3]]},
            ],
        }

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
        return []

    def study_join(self, plan_id: str, code: str) -> dict[str, Any]:
        return self.plan

    def event(self, eid: str) -> dict[str, Any]:
        return {
            "id": eid,
            "image_ids": [f"{eid}-a"],
            "decision": {
                "suggested_label": "raccoon",
                "confidence": 0.7,
                "reasons": ["low_confidence"],
            },
        }

    def thumbnail(self, image_id: str, size: int = 320) -> bytes:
        return _png()

    def study_trial(self, plan_id: str, trial: dict[str, Any]) -> None:
        self.trials.append(trial)

    def study_rating(self, plan_id: str, rating: dict[str, Any]) -> None:
        self.ratings.append(rating)


def test_study_page_hides_suggestions_in_the_grouped_condition_and_logs_timing() -> None:
    api = StudyApi()
    at = AppTest.from_file(str(APP), default_timeout=30)
    at.session_state["client"] = api
    at.run()
    at.sidebar.radio(key="page").set_value("Study").run()
    at.text_input(key="study-plan").input("plan-1").run()
    at.text_input(key="study-code").input("P7").run()
    at.button(key="study-join").click().run()
    at.button(key="study-consent").click().run()
    e = api.plan["practice"][0]["events"][0]
    at.button(key=f"study-accept-{e}").click().run()  # practice, suggested
    e = api.plan["practice"][1]["events"][0]
    assert not [
        b for b in at.button if b.key and b.key.startswith("study-accept")
    ]  # grouped: no Accept
    assert not any("Suggested" in m.value for m in at.markdown)
    at.button(key=f"study-{e}-cant").click().run()
    e = api.plan["blocks"][0]["events"][0]
    at.button(key=f"study-accept-{e}").click().run()  # block 1 ends: rating
    at.radio(key="study-rating-1").set_value(2).run()
    at.button(key="study-rated-1").click().run()
    e = api.plan["blocks"][1]["events"][0]
    at.button(key=f"study-{e}-other").click().run()
    at.text_input(key=f"study-{e}-name").input("Skunk").run()
    at.button(key=f"study-{e}-save").click().run()
    at.radio(key="study-rating-2").set_value(4).run()
    at.button(key="study-rated-2").click().run()
    assert not at.exception
    assert [(t["block"], t["condition"], t["label"], t["interactions"]) for t in api.trials] == [
        (0, "suggested", "raccoon", 1),
        (0, "grouped", None, 1),
        (1, "suggested", "raccoon", 1),
        (2, "grouped", "skunk", 2),
    ]
    assert all(t["seconds"] >= 0 and t["participant"] == "P7" for t in api.trials)
    assert [(r["block"], r["difficulty"]) for r in api.ratings] == [(1, 2), (2, 4)]
    assert any("Thank you" in s.value for s in at.success)
