from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from wildinbox.training.promotion import (
    INCONCLUSIVE,
    LEGACY,
    PROMOTE,
    REJECT,
    SPECIES,
    PolicyError,
    budget_check,
    decide,
    evidence_checks,
    policy_level,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["empty", "bobcat", "cat", "coyote", "dog", "opossum", "rabbit", "raccoon"]
LEGACY_GATE = {
    "min_holdout_gain": 0.02,
    "max_regression": 0.02,
    "max_false_empty_suggestion_increase": 0.10,
}
# The policy docs/retraining.md sets for cycles from now on.
GATE = {
    **LEGACY_GATE,
    "species": {"critical": CLASSES[1:], "max_recall_drop": 0.10, "min_events": 30},
    "min_evidence": {"holdout_events": 200, "animal_events": 100},
}


def _rehearsal() -> dict[str, Any]:
    out = json.loads((REPO_ROOT / "reports/update/finetune-e3-rehearsal/metrics.json").read_text())
    results: dict[str, Any] = out["results"]
    return results


def _results(recall: dict[str, tuple[float, float]], support: int = 50) -> dict[str, Any]:
    """Deployed and candidate results: equal everywhere except these recalls."""

    def model(i: int) -> dict[str, Any]:
        return {
            "holdout": {
                "macro_f1": 0.50 + 0.05 * i,
                "events_scored": 400,
                "animal_events": 400,
                "animal_events_suggested_empty": 40,
                "per_class": {
                    k: {"recall": recall.get(k, (0.6, 0.6))[i], "support": support} for k in CLASSES
                },
            },
            "regression_macro_f1": {"calibration_cameras": 0.45, "seen_camera_diagnostic": 0.75},
        }

    return {"deployed": model(0), "candidate": model(1)}


def test_the_rehearsal_candidate_is_rejected_for_bobcat_under_the_species_policy() -> None:
    r = _rehearsal()
    checks = evidence_checks(r["deployed"], r["candidate"], GATE)
    # Every aggregate check passes, as the recorded gate found ...
    aggregate = [k for k in checks if not k.startswith("species_recall_")]
    assert all(checks[k]["pass"] is True for k in aggregate)
    # ... but bobcat recall fell 0.593 -> 0.333 on 81 events.
    bobcat = checks["species_recall_bobcat"]
    assert bobcat["pass"] is False and bobcat["support"] == 81
    assert bobcat["value"] == pytest.approx(0.333333 - 0.592593)
    # Coyote (16 events) and rabbit (4) are too sparse to judge either way.
    assert checks["species_recall_coyote"]["pass"] is None
    assert checks["species_recall_rabbit"]["pass"] is None
    assert decide(checks) == REJECT


def test_the_legacy_policy_reproduces_the_recorded_promotion() -> None:
    r = _rehearsal()
    checks = evidence_checks(r["deployed"], r["candidate"], LEGACY_GATE)
    assert not any(k.startswith("species_recall_") for k in checks)
    assert decide(checks) == PROMOTE


def test_a_sparse_critical_species_makes_an_otherwise_passing_candidate_inconclusive() -> None:
    r = _results({"rabbit": (0.8, 0.2)}, support=50)
    r["deployed"]["holdout"]["per_class"]["rabbit"]["support"] = 4
    checks = evidence_checks(r["deployed"], r["candidate"], GATE)
    assert checks["species_recall_rabbit"]["pass"] is None
    assert decide(checks) == INCONCLUSIVE
    # Declared non-critical, the same numbers promote: the protocol decides.
    gate = copy.deepcopy(GATE)
    gate["species"]["critical"].remove("rabbit")
    assert decide(evidence_checks(r["deployed"], r["candidate"], gate)) == PROMOTE


def test_recall_drops_are_judged_against_the_limit_inclusively() -> None:
    at_limit = _results({"cat": (0.70, 0.60)})
    assert decide(evidence_checks(at_limit["deployed"], at_limit["candidate"], GATE)) == PROMOTE
    past = _results({"cat": (0.70, 0.59)})
    checks = evidence_checks(past["deployed"], past["candidate"], GATE)
    assert checks["species_recall_cat"]["pass"] is False and decide(checks) == REJECT


def test_a_failure_on_enough_evidence_outweighs_sparse_evidence_elsewhere() -> None:
    r = _results({"bobcat": (0.7, 0.4), "rabbit": (0.5, 0.5)})
    r["deployed"]["holdout"]["per_class"]["rabbit"]["support"] = 3
    assert decide(evidence_checks(r["deployed"], r["candidate"], GATE)) == REJECT


def test_small_holdouts_leave_the_aggregate_checks_unjudged() -> None:
    r = _results({})
    r["deployed"]["holdout"]["events_scored"] = 150
    r["deployed"]["holdout"]["animal_events"] = 60
    checks = evidence_checks(r["deployed"], r["candidate"], GATE)
    assert checks["holdout_gain"]["pass"] is None
    assert checks["false_empty_suggestions"]["pass"] is None
    assert decide(checks) == INCONCLUSIVE
    # The regression checks read fixed partitions and are always judged.
    assert checks["no_regression_calibration_cameras"]["pass"] is True


def test_a_spent_comparison_budget_is_inconclusive_not_a_rejection() -> None:
    r = _results({})
    checks = evidence_checks(r["deployed"], r["candidate"], GATE)
    assert decide({**checks, "comparison_budget": budget_check(3, 3)}) == PROMOTE
    assert decide({**checks, "comparison_budget": budget_check(4, 3)}) == INCONCLUSIVE
    r = _results({"dog": (0.8, 0.3)})
    checks = evidence_checks(r["deployed"], r["candidate"], GATE)
    assert decide({**checks, "comparison_budget": budget_check(4, 3)}) == REJECT


@pytest.mark.parametrize(
    ("gate", "message"),
    [
        ({**GATE, "species": {**GATE["species"], "critical": []}}, "at least one class"),
        (
            {**GATE, "species": {**GATE["species"], "critical": ["bobcat", "moose"]}},
            "moose",
        ),
        ({**GATE, "species": {**GATE["species"], "max_recall_drop": 1.5}}, "max_recall_drop"),
        ({**GATE, "species": {**GATE["species"], "min_events": 0}}, "min_events"),
        ({**GATE, "min_evidence": {"holdout_events": 200}}, "animal_events"),
        ({**GATE, "min_evidence": {"holdout_events": True, "animal_events": 1}}, "holdout"),
        ({k: v for k, v in GATE.items() if k != "min_evidence"}, "not the other block"),
    ],
)
def test_malformed_policies_are_refused(gate: dict[str, Any], message: str) -> None:
    with pytest.raises(PolicyError, match=message):
        policy_level(gate, CLASSES, allow_legacy=True)


def test_a_protocol_without_the_policy_needs_the_legacy_flag() -> None:
    assert policy_level(GATE, CLASSES, allow_legacy=False) == SPECIES
    with pytest.raises(PolicyError, match="--legacy-policy"):
        policy_level(LEGACY_GATE, CLASSES, allow_legacy=False)
    assert policy_level(LEGACY_GATE, CLASSES, allow_legacy=True) == LEGACY


def test_the_report_states_the_three_way_decision_and_each_species(tmp_path: Path) -> None:
    from wildinbox.training.gate_report import write_report

    out = json.loads((REPO_ROOT / "reports/update/finetune-e3-rehearsal/metrics.json").read_text())
    r = out["results"]
    out["checks"] = evidence_checks(r["deployed"], r["candidate"], GATE)
    out["decision"] = decide(out["checks"])
    out["promote"] = False
    out["promotion_policy"] = SPECIES
    md = write_report(tmp_path, out).read_text()
    assert "**Reject** `finetune-e3-rehearsal@" in md
    assert "Recall change, bobcat (0.593 -> 0.333, 81 events) | -0.259 | >= -0.10" in md
    assert "insufficient evidence (4 < 30)" in md  # rabbit
    assert ">= +0.02, on >= 200 events | yes" in md

    out["checks"] = evidence_checks(r["deployed"], r["candidate"], LEGACY_GATE)
    out["decision"], out["promote"], out["promotion_policy"] = PROMOTE, True, LEGACY
    md = write_report(tmp_path, out).read_text()
    assert "**Promote**" in md and "legacy policy" in md and "Recall change" not in md
