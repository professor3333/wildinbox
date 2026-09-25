"""Stage 13 metric definitions, clustered intervals, and gallery selection."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from wildinbox.evaluation.final_eval import (
    EventRecord,
    FinalEvaluationError,
    against_targets,
    check_decisions,
    cluster_bootstrap,
    night_key,
    ratios,
    select_gallery,
    summarize,
    wilson,
)

UNC = {"resamples": 400, "seed": 1, "level": 0.95}


def rec(
    i: int,
    role: str,
    disp: str,
    label: str | None = "raccoon",
    suggested: str | None = "raccoon",
    camera: str = "7",
    night: str = "2012-01-01",
    audited: bool = False,
    confidence: float = 0.5,
) -> EventRecord:
    return EventRecord(
        event_id=f"e{i}",
        camera_id=camera,
        night=True,
        camera_night=f"{camera}|{night}",
        role=role,
        label=label,
        suggested=suggested,
        confidence=confidence,
        dispositions={"p": disp},
        audited={"p": audited},
        frames=[("img", {"raccoon": 0.5})],
    )


def value(records: list[EventRecord], name: str) -> float | None:
    return ratios(records, "p")[name].value()


def test_retention_counts_animals_only_and_by_group() -> None:
    rs = [
        rec(1, "supported_species", "likely_empty"),
        rec(2, "supported_species", "needs_review"),
        rec(3, "unsupported_animal", "needs_review", label="skunk"),
        rec(4, "mixed_species", "needs_review", label=None),
        rec(5, "empty", "likely_empty", label="empty"),  # not an animal
        rec(6, "non_animal", "likely_empty", label="car"),  # neither animal nor empty
    ]
    assert value(rs, "animal_event_retention") == 3 / 4
    assert value(rs, "retention_supported_species") == 1 / 2
    assert value(rs, "retention_unsupported_animal") == 1.0
    assert value(rs, "retention_mixed_species") == 1.0


def test_precision_is_undefined_without_accepted_labels() -> None:
    rs = [rec(1, "supported_species", "needs_review")]
    assert value(rs, "accepted_species_precision") is None
    rs.append(rec(2, "supported_species", "species_identified", suggested="coyote"))
    rs.append(rec(3, "supported_species", "species_identified"))
    assert value(rs, "accepted_species_precision") == 1 / 2


def test_coverage_and_review_reduction_include_audit_work() -> None:
    rs = [
        rec(1, "empty", "likely_empty", label="empty"),
        rec(2, "empty", "likely_empty", label="empty", audited=True),
        rec(3, "supported_species", "species_identified"),
        rec(4, "supported_species", "needs_review"),
    ]
    assert value(rs, "automatic_coverage") == 3 / 4
    # reviewed: 1 needing review + 1 audited automatic event = 2 of 4 grouped events
    assert value(rs, "review_reduction") == 0.5
    s = summarize(rs, "p", UNC)["review_reduction"]
    assert (s["numerator"], s["denominator"]) == (2, 4)  # events no longer reviewed


def test_released_policy_with_automation_off_reduces_nothing() -> None:
    rs = [rec(i, "supported_species", "needs_review") for i in range(5)]
    assert value(rs, "review_reduction") == 0.0
    assert value(rs, "automatic_coverage") == 0.0


def test_unsupported_false_acceptance() -> None:
    rs = [
        rec(1, "unsupported_animal", "species_identified", label="skunk"),
        rec(2, "unsupported_animal", "needs_review", label="skunk"),
        rec(3, "supported_species", "species_identified"),
    ]
    assert value(rs, "unsupported_false_acceptance") == 1 / 2


def test_cluster_bootstrap_is_wider_than_naive_when_clusters_differ() -> None:
    # Two camera-nights: one loses every animal, one keeps every animal.
    rs = [rec(i, "supported_species", "likely_empty", night="a") for i in range(50)]
    rs += [rec(100 + i, "supported_species", "needs_review", night="b") for i in range(50)]
    r = ratios(rs, "p")["animal_event_retention"]
    lo, hi = cluster_bootstrap(r, [x.camera_night for x in rs], 400, 1, 0.95)
    naive = wilson(50, 100)
    assert naive is not None and hi - lo > naive[1] - naive[0]
    assert lo == 0.0 and hi == 1.0


def test_cluster_bootstrap_complement_and_empty_denominator() -> None:
    rs = [rec(i, "supported_species", "needs_review", night=str(i % 3)) for i in range(9)]
    r = ratios(rs, "p")["review_reduction"]
    assert cluster_bootstrap(r, [x.camera_night for x in rs], 100, 1, 0.95) == (0.0, 0.0)
    p = ratios(rs, "p")["accepted_species_precision"]
    assert cluster_bootstrap(p, [x.camera_night for x in rs], 100, 1, 0.95) is None


def test_targets_distinguish_met_and_met_with_confidence() -> None:
    rs = [rec(i, "supported_species", "needs_review", night=str(i)) for i in range(40)]
    rs.append(rec(99, "supported_species", "likely_empty", night="x"))
    s = summarize(rs, "p", UNC)
    t = against_targets(s, {"animal_event_retention": 0.97, "accepted_species_precision": 0.95})
    assert t["animal_event_retention"]["met"] is True
    assert t["animal_event_retention"]["met_with_confidence"] is False
    assert t["accepted_species_precision"]["met"] is None  # undefined, never 0 or 100%


def test_night_key_runs_noon_to_noon() -> None:
    assert night_key("7", "2012-04-27T22:57:56") == night_key("7", "2012-04-28T03:10:00")
    assert night_key("7", "2012-04-28T13:00:00") != night_key("7", "2012-04-28T03:10:00")
    assert night_key("7", None) == "7|unknown"


def test_gallery_selection_is_deterministic_and_rule_based() -> None:
    rs = [
        rec(i, "supported_species", "needs_review", suggested="coyote", confidence=0.95)
        for i in range(10)
    ]
    rs.append(rec(50, "supported_species", "needs_review", suggested="coyote", confidence=0.5))
    for r in rs:
        r.dispositions["rule"] = r.dispositions["released"] = "needs_review"
    cats = [{"name": "confident_wrong_species", "rule": "x", "count": 3}]
    a, b = select_gallery(rs, cats), select_gallery(list(reversed(rs)), cats)
    assert a["confident_wrong_species"]["matching_events"] == 10
    assert [r.event_id for r in a["confident_wrong_species"]["shown"]] == [
        r.event_id for r in b["confident_wrong_species"]["shown"]
    ]


def test_decisions_must_match_the_recorded_final_test(tmp_path: Path) -> None:
    r = rec(1, "supported_species", "needs_review")
    r.dispositions = {
        k: "needs_review" for k in ("v1_released", "v1_rule", "v2_released", "v2_rule")
    }
    path = tmp_path / "d.jsonl.gz"
    row = {"event_id": "e1", "role": "supported_species", "label": "raccoon"}
    with gzip.open(path, "wt") as f:
        f.write(json.dumps({**row, "released": "needs_review", "rule": "needs_review"}) + "\n")
    assert check_decisions([r], path)["v2_equals_v1"] is True
    r.dispositions["v2_rule"] = "likely_empty"
    with pytest.raises(FinalEvaluationError, match="v2 decides 1"):
        check_decisions([r], path)
    with gzip.open(path, "wt") as f:
        f.write(json.dumps({**row, "released": "needs_review", "rule": "likely_empty"}) + "\n")
    r.dispositions["v2_rule"] = "needs_review"
    with pytest.raises(FinalEvaluationError, match="decided differently"):
        check_decisions([r], path)


def test_wilson_matches_known_value() -> None:
    lo, hi = wilson(75, 7154) or (0, 0)
    assert np.isclose(lo, 0.0084, atol=2e-4) and np.isclose(hi, 0.0131, atol=2e-4)
