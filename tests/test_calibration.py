from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from wildinbox.evaluation.calibration import (
    Grid,
    OperatingPointRule,
    OperatingPointSpec,
    apply_temperature,
    choose_operating_point,
    fit_temperature,
    load_rule,
    nll,
)
from wildinbox.evaluation.metrics import ScoredEvent

REPO_ROOT = Path(__file__).resolve().parents[1]
RULE = REPO_ROOT / "configs/experiments/operating_point.yaml"


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    out: np.ndarray = e / e.sum(axis=1, keepdims=True)
    return out


def test_fit_temperature_recovers_known_temperature() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(0, 3, size=(20000, 5))
    true_t = 2.0
    y = np.array([rng.choice(5, p=p) for p in _softmax(z / true_t)])
    t = fit_temperature(_softmax(z), y)
    assert abs(t - true_t) / true_t < 0.05
    assert nll(apply_temperature(_softmax(z), t), y) < nll(_softmax(z), y)


def test_temperature_keeps_predicted_class_and_identity_at_one() -> None:
    p = _softmax(np.random.default_rng(1).normal(0, 2, size=(50, 4)))
    np.testing.assert_allclose(apply_temperature(p, 1.0), p, atol=1e-9)
    assert (apply_temperature(p, 3.0).argmax(1) == p.argmax(1)).all()


def test_grid_values_are_clean_and_include_extras() -> None:
    v = Grid(start=0.5, stop=0.99, step=0.01, extra=[0.995, 0.999]).values()
    assert v[0] == 0.5 and 0.99 in v and v[-2:] == [0.995, 0.999] and len(v) == 52
    assert 0.81 in v  # no float drift such as 0.8100000000000001


def _spec(**kw: float) -> OperatingPointSpec:
    g = {"start": 0.5, "stop": 0.99, "step": 0.01, "extra": [0.995, 0.999]}
    base = {
        "choose_on": "policy_validation",
        "policy": "conservative",
        "interval": "wilson95",
        "max_false_empty_rate": 0.02,
        "min_accepted_precision": 0.95,
        "empty_grid": g,
        "species_grid": g,
    }
    return OperatingPointSpec.model_validate({**base, **kw})


def _ev(i: int, role: str, label: str | None, animal: bool, pe: float, pc: float) -> ScoredEvent:
    frame = {"empty": pe, "cat": pc, "dog": round(1 - pe - pc, 6)}
    return ScoredEvent(f"e{i}", role, label, animal, [frame])


def test_chooses_lowest_threshold_that_passes_at_the_interval_bound() -> None:
    events = (
        [_ev(i, "supported_species", "cat", True, 0.3, 0.7) for i in range(1000)]
        # 30 animals look empty-ish: filtering at <= 0.80 loses them (3% > 2%).
        + [_ev(1000 + i, "supported_species", "cat", True, 0.8, 0.1) for i in range(30)]
        # 3 animals look very empty: lost at <= 0.96, within the 2% bound.
        + [_ev(2000 + i, "unsupported_animal", "squirrel", True, 0.96, 0.02) for i in range(3)]
        + [_ev(3000 + i, "empty", "empty", False, 0.97, 0.02) for i in range(100)]
        # 100 dogs predicted cat at 0.6: accepting at <= 0.60 drops precision to 91%.
        + [_ev(4000 + i, "supported_species", "dog", True, 0.1, 0.6) for i in range(100)]
    )
    op = choose_operating_point(events, _spec())
    assert op.empty_threshold == 0.81
    assert op.metrics["false_empty"] == 3
    assert op.species_threshold == 0.61
    assert op.metrics["accepted"] == 1000 and op.metrics["accepted_correct"] == 1000
    failing = [e for e in op.empty_sweep if e["thresholds"]["empty_filter"] <= 0.8]
    assert failing and not any(e["passes"] for e in failing)


def test_automation_stays_disabled_when_nothing_passes() -> None:
    events = [_ev(i, "supported_species", "cat", True, 0.9995, 0.0004) for i in range(50)] + [
        _ev(100 + i, "supported_species", "dog", True, 0.1, 0.8) for i in range(50)
    ]
    op = choose_operating_point(events, _spec())
    assert op.empty_threshold is None and op.species_threshold is None
    assert op.metrics["filtered"] == 0 and op.metrics["accepted"] == 0


def test_committed_rule_is_valid() -> None:
    rule = load_rule(RULE)
    assert rule.calibration.fit_partition == "calibration"
    assert rule.operating_point.choose_on == "policy_validation"


@pytest.mark.parametrize(
    ("section", "key"), [("calibration", "fit_partition"), ("operating_point", "choose_on")]
)
def test_rule_cannot_use_the_final_test(section: str, key: str) -> None:
    raw = yaml.safe_load(RULE.read_text())
    raw[section][key] = "final_test"
    with pytest.raises(ValidationError):
        OperatingPointRule.model_validate(raw)
