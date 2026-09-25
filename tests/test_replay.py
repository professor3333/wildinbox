"""Acceptance gate: every event disposition is reproducible from saved
predictions and its policy version."""

from __future__ import annotations

import copy
import gzip
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from wildinbox.policy.conservative import POLICY_NAME, Frame, PolicyConfig, decide
from wildinbox.policy.replay import frames_from_saved, replay

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY = REPO_ROOT / "reports/calibration/policy.json"
DECISIONS = REPO_ROOT / "reports/calibration/decisions.jsonl.gz"


def _cfg(cfg: PolicyConfig) -> dict[str, Any]:
    return {**asdict(cfg), "policy_version": f"{POLICY_NAME}+{cfg.fingerprint()}"}


def _write(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    path = tmp_path / "decisions.jsonl.gz"
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


@pytest.fixture
def small() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    policy = {
        "policy": POLICY_NAME,
        "classes": ["empty", "raccoon"],
        "calibration": {"temperature": 2.0},
        "unfamiliar": {"threshold": 0.5, "adopted": True},
        "released": _cfg(PolicyConfig(0.9, 0.9, False, False)),
        "rule": _cfg(PolicyConfig(0.9, 0.9, True, True)),
    }
    raw = [
        [[0.999, 0.001], [0.998, 0.002]],  # empty
        [[0.001, 0.999]],  # raccoon
        [[0.001, 0.999]],  # raccoon, but unfamiliar
        [[0.999, 0.001], None],  # empty-looking with a failed frame
    ]
    rows = []
    for i, frames in enumerate(raw):
        saved = {
            "event_id": f"e{i}",
            "frames": [
                {
                    "image_id": f"i{i}{j}",
                    "status": "failed" if p is None else "completed",
                    "raw_probs": p,
                    "unfamiliar_distance": 0.9 if i == 2 else 0.1,
                }
                for j, p in enumerate(frames)
            ],
        }
        fr = frames_from_saved(saved, policy)
        for name in ("released", "rule"):
            o = decide(fr, PolicyConfig(0.9, 0.9, name == "rule", name == "rule"))
            saved[name] = {
                "disposition": o.disposition.value,
                "label": o.label,
                "confidence": o.confidence,
                "reasons": [r.value for r in o.reasons],
            }
        rows.append(saved)
    return policy, rows


def test_replay_reproduces_saved_decisions(
    tmp_path: Path, small: tuple[dict[str, Any], list[dict[str, Any]]]
) -> None:
    policy, rows = small
    out = replay(_write(tmp_path, rows), policy)
    assert out["problems"] == []
    assert out["dispositions"]["rule"] == {
        "likely_empty": 1,
        "needs_review": 2,
        "species_identified": 1,
    }
    assert out["dispositions"]["released"] == {"needs_review": 4}
    assert rows[2]["rule"]["reasons"] == ["possible_unknown"]
    assert rows[3]["rule"]["reasons"] == ["processing_failure"]


def test_replay_detects_a_changed_decision(
    tmp_path: Path, small: tuple[dict[str, Any], list[dict[str, Any]]]
) -> None:
    policy, rows = small
    rows = copy.deepcopy(rows)
    rows[1]["rule"]["disposition"] = "needs_review"
    assert len(replay(_write(tmp_path, rows), policy)["problems"]) == 1


def test_replay_detects_settings_that_do_not_match_the_version(
    tmp_path: Path, small: tuple[dict[str, Any], list[dict[str, Any]]]
) -> None:
    policy, rows = small
    policy = copy.deepcopy(policy)
    policy["rule"]["empty_threshold"] = 0.5  # version string no longer matches
    problems = replay(_write(tmp_path, rows), policy)["problems"]
    assert any("does not match its settings" in p for p in problems)


def test_committed_development_decisions_replay_exactly() -> None:
    out = replay(DECISIONS, json.loads(POLICY.read_text()))
    assert out["events"] > 1000
    assert out["problems"] == []
    # The rule's operating point automates some events, so replay covers more
    # than the all-review release.
    assert set(out["dispositions"]["rule"]) != {"needs_review"}


def test_frame_helper_is_the_policy_input() -> None:
    policy = {"classes": ["empty", "cat"], "calibration": {"temperature": 1.0}, "unfamiliar": None}
    (f,) = frames_from_saved(
        {"frames": [{"status": "completed", "raw_probs": [0.2, 0.8], "unfamiliar_distance": 9}]},
        policy,
    )
    assert isinstance(f, Frame) and f.unfamiliar is False
    assert f.probs == pytest.approx({"empty": 0.2, "cat": 0.8})
