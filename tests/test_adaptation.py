"""v2 adaptation experiment: what each method may see, and how results count."""

from __future__ import annotations

import numpy as np
import pytest

from wildinbox.adaptation.data import CameraEvent, FrameOutputs
from wildinbox.adaptation.evaluate import (
    AUDIT_RATE,
    Scored,
    choose,
    leave_one_camera_out,
    outcome,
    score_camera,
    summarize,
)
from wildinbox.adaptation.methods import PriorShift, Release, Scorer

CLASSES = ("empty", "bobcat", "opossum")


def _event(i: int, label: str, role: str = "supported_species") -> CameraEvent:
    return CameraEvent(
        event_id=f"e{i}",
        camera_id="c",
        start=f"2014-01-01T00:{i:02d}:00",
        role=role,
        label=label,
        image_ids=(f"f{i}",),
        image_labels=(label,),
    )


def _outputs(n: int, probs: list[float]) -> FrameOutputs:
    p = np.tile(np.array(probs), (n, 1))
    return FrameOutputs(
        index={f"f{i}": i for i in range(n)},
        log_probs=np.log(p),
        features=np.zeros((n, 4)),
        classes=CLASSES,
    )


class Spy:
    name = "spy"

    def __init__(self) -> None:
        self.seen: list[str] = []

    def adapt(self, reviewed: list[CameraEvent], out: FrameOutputs) -> Scorer:
        self.seen = [e.event_id for e in reviewed]
        return Release(1.0).adapt(reviewed, out)


def test_a_method_sees_only_the_first_n_events_and_is_judged_on_the_rest() -> None:
    events = [_event(i, "bobcat") for i in range(10)]
    spy = Spy()
    scored, total = score_camera(spy, events, 4, _outputs(10, [0.05, 0.9, 0.05]))
    assert spy.seen == ["e0", "e1", "e2", "e3"]
    assert len(scored) == 6 and total == 10
    assert all(s.suggestion == "bobcat" and s.correct for s in scored)


def test_prior_shift_moves_probability_toward_the_cameras_classes() -> None:
    out = _outputs(3, [0.2, 0.4, 0.4])
    reviewed = [_event(i, "opossum") for i in range(3)]
    p = PriorShift(temperature=1.0).adapt(reviewed, out)(np.array([0]))[0]
    assert p[2] > 0.4 > p[1] and p.sum() == pytest.approx(1.0)


def _scored(conf: float, correct: bool, role: str = "supported_species") -> Scored:
    return Scored("c", role, "bobcat", 0.0, "bobcat" if correct else "opossum", conf)


def test_species_threshold_needs_the_wilson_lower_bound() -> None:
    # 100 correct at 0.9; 10 wrong at 0.7: below 0.8 precision is 100/110
    high = [_scored(0.9, True) for _ in range(100)]
    low = [_scored(0.7, False) for _ in range(10)]
    _, t_species = choose(high + low)
    assert t_species is not None and 0.7 < t_species <= 0.9
    assert choose(low)[1] is None  # nothing passes: acceptance stays off


def test_review_reduction_counts_reviewed_events_and_audits() -> None:
    later = [_scored(0.99, True) for _ in range(50)]
    o = summarize(outcome(later, total=100, t_empty=None, t_species=0.9))
    assert o["automated"] == 50
    assert o["review_reduction"] == pytest.approx(50 * (1 - AUDIT_RATE) / 100)


def test_pooled_rates_come_from_pooled_counts() -> None:
    """Two cameras at 10% each pool to 10%, not 20%."""
    cam = [_scored(0.99, True) for _ in range(100)] + [_scored(0.1, True) for _ in range(900)]
    loco = leave_one_camera_out({"a": (cam, 1000), "b": (list(cam), 1000)})
    assert loco["pooled"]["review_reduction"] == pytest.approx(0.10 * (1 - AUDIT_RATE))


def test_leave_one_camera_out_never_uses_the_camera_itself() -> None:
    good = [_scored(0.99, True) for _ in range(200)]
    bad = [_scored(0.99, False) for _ in range(5)]
    loco = leave_one_camera_out({"a": (good, 200), "b": (bad, 5)})
    assert loco["cameras"]["a"]["t_species"] is None  # b alone cannot pass the rule
    assert loco["cameras"]["b"]["t_species"] is not None  # chosen on a, applied to b
    assert loco["cameras"]["b"]["precision"] == 0.0
