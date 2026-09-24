from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

from wildinbox.evaluation.compare import compare, load_rule, load_summary, verdict

from .conftest import REPO_ROOT

BASELINE = REPO_ROOT / "reports/baseline"


def _candidate(tmp: Path, name: str, f1_gain: float, fe_factor: float, lat_factor: float) -> Path:
    m: dict[str, Any] = copy.deepcopy(json.loads((BASELINE / "metrics.json").read_text()))
    m["model"] = name
    u = m["groups"]["unseen_cameras"]
    u["image"]["macro_f1"] += f1_gain
    u["event_reference"]["false_empty"] = round(u["event_reference"]["false_empty"] * fe_factor)
    for x in m["latency"]:
        x["p50_ms"] *= lat_factor
    out = tmp / "reports" / "experiments" / name
    out.mkdir(parents=True)
    (out / "metrics.json").write_text(json.dumps(m))
    return out


def _rule(tmp: Path) -> Path:
    rule = yaml.safe_load((REPO_ROOT / "configs/experiments/selection.yaml").read_text())
    rule["reference"] = str(BASELINE)
    p = tmp / "rule.yaml"
    p.write_text(yaml.safe_dump(rule))
    return p


def test_rule_accepts_only_candidates_meeting_every_condition(tmp_path: Path) -> None:
    rule = load_rule(_rule(tmp_path))
    ref = load_summary(BASELINE)
    good = load_summary(_candidate(tmp_path, "good", 0.10, 1.0, 1.1))
    assert verdict(good, ref, rule)[0] is True
    small_gain = load_summary(_candidate(tmp_path, "small", 0.01, 1.0, 1.0))
    assert verdict(small_gain, ref, rule)[0] is False
    more_false_empty = load_summary(_candidate(tmp_path, "fe", 0.20, 1.5, 1.0))
    ok, why = verdict(more_false_empty, ref, rule)
    assert ok is False and any("false-empty" in w for w in why)
    slow = load_summary(_candidate(tmp_path, "slow", 0.20, 1.0, 3.0))
    assert verdict(slow, ref, rule)[0] is False


def test_compare_picks_best_passing_candidate(tmp_path: Path) -> None:
    dirs = [
        _candidate(tmp_path, "a", 0.08, 1.0, 1.0),
        _candidate(tmp_path, "b", 0.12, 1.0, 1.0),
        _candidate(tmp_path, "c", 0.30, 2.0, 1.0),
    ]  # c is best on F1 but fails false-empty
    text, best = compare(dirs, _rule(tmp_path))
    assert best is not None and best.name == "b"
    assert "Selected by the rule:** b" in text


def test_compare_retains_baseline_when_nothing_passes(tmp_path: Path) -> None:
    text, best = compare([_candidate(tmp_path, "weak", 0.0, 1.0, 1.0)], _rule(tmp_path))
    assert best is None and "baseline is retained" in text
