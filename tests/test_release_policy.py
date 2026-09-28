"""Which gate records may release a candidate: the shared check, and the release
script stopping before it contacts the deployment. Uses the committed
historical gate records, which predate the promotion-policy marker."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from wildinbox.training.promotion import (
    INCONCLUSIVE,
    REJECT,
    SPECIES,
    PolicyError,
    decide,
    evidence_checks,
    release_authorization,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ["finetune-e3-update1", "finetune-e3-rehearsal"]
LEGACY_GATE = {
    "min_holdout_gain": 0.02,
    "max_regression": 0.02,
    "max_false_empty_suggestion_increase": 0.10,
}


def _record(candidate: str) -> dict[str, Any]:
    out: dict[str, Any] = json.loads(
        (REPO_ROOT / "reports/update" / candidate / "metrics.json").read_text()
    )
    return out


def _species_record(critical: list[str]) -> dict[str, Any]:
    """The rehearsal's results, re-decided under the species policy."""
    out = _record("finetune-e3-rehearsal")
    gate = {
        **LEGACY_GATE,
        "species": {"critical": critical, "max_recall_drop": 0.10, "min_events": 30},
        "min_evidence": {"holdout_events": 200, "animal_events": 100},
    }
    r = out["results"]
    out["checks"] = evidence_checks(r["deployed"], r["candidate"], gate)
    out["decision"] = decide(out["checks"])
    out["promote"] = out["decision"] == "promote"
    out["promotion_policy"] = SPECIES
    return out


# Holdout recall on these moves by at most 0.026, each on >= 32 events.
PASSING = ["cat", "dog", "opossum", "raccoon"]


@pytest.mark.parametrize("candidate", HISTORICAL)
def test_historical_records_are_legacy_and_need_the_opt_in(candidate: str) -> None:
    record = _record(candidate)
    assert record["promote"] is True and "promotion_policy" not in record
    with pytest.raises(PolicyError, match="records no promotion policy"):
        release_authorization(record, allow_legacy=False)
    assert release_authorization(record, allow_legacy=True) == "legacy"


def test_a_current_promotion_is_authorized_and_rejections_are_not() -> None:
    assert release_authorization(_species_record(PASSING), allow_legacy=False) == SPECIES
    rejected = _species_record(["bobcat", *PASSING])
    assert rejected["decision"] == REJECT
    with pytest.raises(PolicyError, match="decided reject"):
        release_authorization(rejected, allow_legacy=True)
    sparse = _species_record(["rabbit", *PASSING])
    assert sparse["decision"] == INCONCLUSIVE
    with pytest.raises(PolicyError, match="decided inconclusive"):
        release_authorization(sparse, allow_legacy=True)


def _edit(record: dict[str, Any], **changes: Any) -> dict[str, Any]:
    out = copy.deepcopy(record)
    out.update(changes)
    return out


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"promotion_policy": "species-v2"}, "unrecognized promotion policy"),
        ({"promotion_policy": None}, "unrecognized promotion policy"),
        ({"decision": "maybe"}, "unrecognized decision"),
        ({"promote": False}, "inconsistent"),
        ({"decision": "reject", "promote": False}, "inconsistent"),
        ({"checks": {}}, "no checks"),
    ],
)
def test_unknown_or_inconsistent_records_are_refused(change: dict[str, Any], message: str) -> None:
    with pytest.raises(PolicyError, match=message):
        release_authorization(_edit(_species_record(PASSING), **change), allow_legacy=True)


def test_a_species_record_without_species_checks_is_refused() -> None:
    record = _species_record(PASSING)
    record["checks"] = {
        k: v for k, v in record["checks"].items() if not k.startswith("species_recall_")
    }
    with pytest.raises(PolicyError, match="no species checks"):
        release_authorization(record, allow_legacy=True)


def test_an_edited_check_cannot_turn_a_rejection_into_a_release() -> None:
    record = _species_record(["bobcat", *PASSING])
    record["decision"], record["promote"] = "promote", True
    with pytest.raises(PolicyError, match="checks give 'reject'"):
        release_authorization(record, allow_legacy=True)


# The release script, end to end up to the deployment boundary.


class ReachedDeployment(Exception):
    pass


def _script(monkeypatch: pytest.MonkeyPatch, deployed: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "release_rollback", REPO_ROOT / "scripts/release_rollback.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    contacted: list[str] = []

    class Response:
        def json(self) -> dict[str, Any]:
            return {"active_release": {"id": deployed}}

    class Client:
        def __init__(self, **kwargs: Any) -> None:
            contacted.append("client")

        def get(self, path: str) -> Response:
            return Response()

    def compose(*args: str) -> str:
        raise ReachedDeployment(" ".join(args))

    monkeypatch.setattr(module.httpx, "Client", Client)
    monkeypatch.setattr(module, "compose", compose)
    module.contacted = contacted  # type: ignore[attr-defined]
    return module


def _run(
    monkeypatch: pytest.MonkeyPatch, gate_dir: Path, deployed: str, *flags: str
) -> tuple[str, list[str]]:
    """'stopped' or 'deployed', and whether the deployment was contacted."""
    module = _script(monkeypatch, deployed)
    monkeypatch.setattr(sys, "argv", ["release_rollback.py", "--gate", str(gate_dir), *flags])
    try:
        module.main()
    except SystemExit as e:
        assert e.code not in (0, None)
        return "stopped", module.contacted  # type: ignore[attr-defined]
    except ReachedDeployment as e:
        assert "register" in str(e) and "--activate" in str(e)
        return "deployed", module.contacted  # type: ignore[attr-defined]
    raise AssertionError("the script neither stopped nor reached the deployment")


@pytest.mark.parametrize("candidate", HISTORICAL)
def test_the_script_stops_on_historical_records_unless_opted_in(
    monkeypatch: pytest.MonkeyPatch, candidate: str
) -> None:
    gate_dir = REPO_ROOT / "reports/update" / candidate
    deployed = _record(candidate)["deployed_release"]
    assert _run(monkeypatch, gate_dir, deployed) == ("stopped", [])
    outcome, _ = _run(monkeypatch, gate_dir, deployed, "--legacy-policy")
    assert outcome == "deployed"


@pytest.mark.parametrize(
    ("record", "flags", "expected"),
    [
        (lambda: _edit(_record(HISTORICAL[0]), promotion_policy="legacy"), (), "stopped"),
        (
            lambda: _edit(_species_record(PASSING), promotion_policy="v9"),
            ("--legacy-policy",),
            "stopped",
        ),
        (lambda: _species_record(["bobcat", *PASSING]), ("--legacy-policy",), "stopped"),
        (lambda: _species_record(["rabbit", *PASSING]), ("--legacy-policy",), "stopped"),
        (lambda: _species_record(PASSING), (), "deployed"),
    ],
    ids=["explicit-legacy", "unknown-policy", "reject", "inconclusive", "current-promote"],
)
def test_the_script_releases_only_authorized_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    record: Any,
    flags: tuple[str, ...],
    expected: str,
) -> None:
    out = record()
    (tmp_path / "metrics.json").write_text(json.dumps(out))
    outcome, contacted = _run(monkeypatch, tmp_path, out["deployed_release"], *flags)
    assert outcome == expected
    if expected == "stopped":
        assert contacted == []  # refused before contacting the deployment
