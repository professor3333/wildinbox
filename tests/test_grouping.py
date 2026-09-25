from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from wildinbox.datasets.grouping import (
    DEFAULT_GAP_SECONDS,
    SEQUENCE_RULE,
    SEQUENCE_SPLIT_SECONDS,
    GroupableImage,
    group_by_sequence,
    group_by_time_gap,
    time_gap_rule_id,
)

T0 = datetime(2012, 5, 9, 7, 33, 45)


def _img(iid: str, cam: str | None, secs: float | None, seq: str | None = None) -> GroupableImage:
    return GroupableImage(iid, cam, T0 + timedelta(seconds=secs) if secs is not None else None, seq)


def test_time_gap_groups_bursts_and_splits_on_gaps() -> None:
    imgs = [
        _img("a", "c1", 0),
        _img("b", "c1", 1),
        _img("c", "c1", 2),
        _img("d", "c1", 60),
        _img("e", "c1", 64),
    ]
    assert group_by_time_gap(imgs, 5) == [["a", "b", "c"], ["d", "e"]]


def test_time_gap_never_crosses_cameras_and_isolates_missing_metadata() -> None:
    imgs = [_img("a", "c1", 0), _img("b", "c2", 1), _img("c", None, 2), _img("d", "c1", None)]
    assert group_by_time_gap(imgs, 5) == [["a"], ["b"], ["c"], ["d"]]


def test_time_gap_is_order_independent() -> None:
    imgs = [_img("b", "c1", 1), _img("a", "c1", 0), _img("c", "c1", 30)]
    assert group_by_time_gap(imgs, 5) == group_by_time_gap(list(reversed(imgs)), 5)


def test_gap_boundary_is_inclusive_and_validated() -> None:
    assert group_by_time_gap([_img("a", "c", 0), _img("b", "c", 5)], 5) == [["a", "b"]]
    with pytest.raises(ValueError):
        group_by_time_gap([], -1)
    assert time_gap_rule_id(DEFAULT_GAP_SECONDS) == "time_gap/v1(gap_s=5)"


def test_group_by_sequence() -> None:
    imgs = [_img("a", "c", 0, "s1"), _img("b", "c", 1, "s1"), _img("c", "c", 2, None)]
    assert group_by_sequence(imgs) == [["a", "b"], ["c"]]


def test_the_same_sequence_id_on_two_cameras_is_two_events() -> None:
    """External review, issue 3: camera counters overlap."""
    imgs = [_img("n1", "north", 0, "001"), _img("s1", "south", 0, "001")]
    assert group_by_sequence(imgs) == [["n1"], ["s1"]]


def test_a_restarted_counter_on_one_camera_splits_by_time() -> None:
    imgs = [
        _img("a", "north", 0, "001"),
        _img("b", "north", 1, "001"),  # burst: one event
        _img("c", "north", 6 * 3600, "001"),  # same id six hours later: a new trigger
        _img("d", "north", 6 * 3600 + 2, "001"),
    ]
    assert group_by_sequence(imgs) == [["a", "b"], ["c", "d"]]


def test_a_sequence_is_split_only_beyond_the_split_gap() -> None:
    imgs = [_img("a", "c", 0, "s"), _img("b", "c", SEQUENCE_SPLIT_SECONDS, "s")]
    assert group_by_sequence(imgs) == [["a", "b"]]
    imgs.append(_img("c", "c", 2 * SEQUENCE_SPLIT_SECONDS + 1, "s"))
    assert group_by_sequence(imgs) == [["a", "b"], ["c"]]


def test_frames_without_a_time_stay_with_their_sequence() -> None:
    imgs = [_img("a", "c", 0, "s"), _img("x", "c", None, "s"), _img("b", "c", 1, "s")]
    assert group_by_sequence(imgs) == [["a", "b", "x"]]
    # Restarted counter: an untimed frame joins the first run (and, if it
    # failed, sends that run to review) rather than being dropped.
    imgs.append(_img("c", "c", 7200, "s"))
    assert group_by_sequence(imgs) == [["a", "b", "x"], ["c"]]


def test_sequences_without_a_camera_group_by_id_alone() -> None:
    imgs = [_img("a", None, 0, "s"), _img("b", None, 1, "s"), _img("c", "north", 1, "s")]
    assert group_by_sequence(imgs) == [["a", "b"], ["c"]]
    assert SEQUENCE_RULE == "sequence_id/v2"
