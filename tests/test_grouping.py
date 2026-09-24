from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from wildinbox.datasets.grouping import (
    DEFAULT_GAP_SECONDS,
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
