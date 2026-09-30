"""v2 adaptation experiment: what each method may see, and how results count."""

from __future__ import annotations

import numpy as np
import pytest

from wildinbox.adaptation.data import CameraEvent, FrameOutputs
from wildinbox.adaptation.evaluate import (
    AUDIT_RATE,
    Scored,
    choose,
    in_sample,
    leave_one_camera_out,
    nested_leave_one_camera_out,
    outcome,
    score_camera,
    summarize,
)
from wildinbox.adaptation.methods import OTHER, Adapted, PriorShift, Release, reviewed_frames

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

    def adapt(self, reviewed: list[CameraEvent], out: FrameOutputs) -> Adapted:
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
    p = PriorShift(temperature=1.0).adapt(reviewed, out).score(np.array([0]))[0]
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


def test_an_other_animal_suggestion_is_never_accepted() -> None:
    s = Scored("c", "unsupported_animal", "bird", 0.0, OTHER, 0.999)
    assert not s.accepted(t_empty=1.1, t_species=0.5)


def test_reviews_of_unsupported_animals_teach_other_only_when_asked() -> None:
    bird = _event(1, "bird", role="unsupported_animal")
    assert reviewed_frames([bird, _event(2, "bobcat")], with_other=True) == (
        ["f1", "f2"],
        [OTHER, "bobcat"],
    )
    assert reviewed_frames([bird], with_other=False) == ([], [])


def test_nested_selection_ignores_the_held_out_camera() -> None:
    """Config x is best on camera a alone; y is better on the others. For
    camera a, the choice must come from b and c, so y is used."""
    good = [_scored(0.99, True) for _ in range(200)]
    none = [_scored(0.1, True) for _ in range(200)]
    per_config = {
        "x": {"a": (good, 200), "b": (none, 200), "c": (none, 200)},
        "y": {"a": (none, 200), "b": (good, 200), "c": (good, 200)},
    }
    loco = nested_leave_one_camera_out(per_config)
    assert loco["cameras"]["a"]["config"] == "y"
    assert loco["cameras"]["a"]["accepted"] == 0


def test_species_only_never_filters_empty() -> None:
    """Empty events the rule would filter stay in review; species thresholds
    are still chosen."""
    empties = [Scored("c", "empty", "empty", 0.99, "empty", 0.99) for _ in range(200)]
    species = [_scored(0.99, True) for _ in range(300)]
    assert choose(empties + species)[0] is not None
    t_empty, t_species = choose(empties + species, filter_empty=False)
    assert t_empty is None and t_species is not None
    o = in_sample({"a": (empties + species, 500)}, filter_empty=False)
    assert o["filtered"] == 0 and o["accepted"] == 300
