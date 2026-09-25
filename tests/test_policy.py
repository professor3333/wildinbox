from __future__ import annotations

import pytest

from wildinbox.policy.conservative import Frame, PolicyConfig, decide
from wildinbox.schemas import Disposition, FrameStatus, ReviewReason

ON = PolicyConfig(0.9, 0.9, auto_filter_enabled=True, auto_accept_enabled=True)
EMPTY = {"empty": 0.99, "raccoon": 0.01}
RACCOON = {"empty": 0.01, "raccoon": 0.99}


def _f(p: dict[str, float] | None, status: FrameStatus = FrameStatus.COMPLETED) -> Frame:
    return Frame(p, status)


def test_complete_empty_event_is_filtered() -> None:
    assert decide([_f(EMPTY), _f(EMPTY)], ON).disposition is Disposition.LIKELY_EMPTY


@pytest.mark.parametrize("status", [FrameStatus.PENDING, FrameStatus.FAILED])
def test_incomplete_event_is_never_filtered(status: FrameStatus) -> None:
    o = decide([_f(EMPTY), _f(None, status)], ON)
    assert o.disposition is Disposition.NEEDS_REVIEW
    assert o.reasons == [ReviewReason.PROCESSING_FAILURE]
    assert o.label == "empty"  # the suggestion is kept for the reviewer


def test_failed_frame_blocks_species_acceptance() -> None:
    o = decide([_f(RACCOON), _f(None, FrameStatus.FAILED)], ON)
    assert o.disposition is Disposition.NEEDS_REVIEW and o.label == "raccoon"
    assert o.reasons == [ReviewReason.PROCESSING_FAILURE]


def test_event_with_no_completed_frame_has_no_suggestion() -> None:
    o = decide([_f(None, FrameStatus.FAILED)], ON)
    assert (o.disposition, o.label, o.reasons) == (
        Disposition.NEEDS_REVIEW,
        None,
        [ReviewReason.PROCESSING_FAILURE],
    )


def test_every_blocking_reason_is_listed() -> None:
    frames = [
        Frame({"empty": 0.0, "raccoon": 0.6, "coyote": 0.4}, unfamiliar=True),
        Frame({"empty": 0.0, "raccoon": 0.4, "coyote": 0.6}),
        _f(None, FrameStatus.FAILED),
    ]
    cfg = PolicyConfig(0.9, 0.9, True, False, accept_species=("dog",))
    assert decide(frames, cfg).reasons == [
        ReviewReason.PROCESSING_FAILURE,
        ReviewReason.CONFLICTING_FRAMES,
        ReviewReason.POSSIBLE_UNKNOWN,
        ReviewReason.LOW_CONFIDENCE,
        ReviewReason.SPECIES_NOT_VALIDATED,
        ReviewReason.AUTOMATION_DISABLED,
    ]


def test_low_support_species_stays_in_review() -> None:
    cfg = PolicyConfig(0.9, 0.9, True, True, accept_species=("coyote",))
    o = decide([_f(RACCOON)], cfg)
    assert o.disposition is Disposition.NEEDS_REVIEW
    assert o.reasons == [ReviewReason.SPECIES_NOT_VALIDATED]


def test_disabled_automation_sends_everything_to_review() -> None:
    off = PolicyConfig(0.9, 0.9, auto_filter_enabled=False, auto_accept_enabled=False)
    for frames in ([_f(EMPTY)], [_f(RACCOON)]):
        o = decide(frames, off)
        assert o.disposition is Disposition.NEEDS_REVIEW
        assert o.reasons == [ReviewReason.AUTOMATION_DISABLED]
    unset = PolicyConfig(None, None, auto_filter_enabled=True, auto_accept_enabled=True)
    assert decide([_f(EMPTY)], unset).disposition is Disposition.NEEDS_REVIEW


def test_any_setting_change_changes_the_policy_fingerprint() -> None:
    variants = [
        ON,
        PolicyConfig(0.91, 0.9, True, True),
        PolicyConfig(0.9, 0.9, False, True),
        PolicyConfig(0.9, 0.9, True, True, accept_species=("raccoon",)),
    ]
    assert len({v.fingerprint() for v in variants}) == len(variants)
