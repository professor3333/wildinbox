from __future__ import annotations

import pytest

from wildinbox.datasets.events import (
    Event,
    FitExclusion,
    Frame,
    Role,
    apply_ground_truth,
    assign_role,
    fit_exclusion,
)
from wildinbox.datasets.spec import Category, Kind, Partition, Taxonomy
from wildinbox.ingestion.inventory import Status

TAX = Taxonomy(
    name="toy",
    categories={
        "empty": Category(kind=Kind.EMPTY),
        "raccoon": Category(kind=Kind.ANIMAL),
        "coyote": Category(kind=Kind.ANIMAL),
        "badger": Category(kind=Kind.ANIMAL),
        "car": Category(kind=Kind.NON_ANIMAL),
    },
)
SUPPORTED = frozenset({"raccoon", "coyote"})


def _f(sid: str, *labels: str, status: Status = Status.ACCEPTED) -> Frame:
    return Frame(
        source_id=sid,
        status=status,
        camera_id="c",
        sequence_id="s",
        source_file="f",
        frame_num=None,
        captured_at=None,
        storage_path=f"{sid}.jpg",
        sha256=sid,
        labels=tuple(sorted(labels)),
    )


def _event(*frames: Frame, partition: Partition = Partition.TRAIN) -> Event:
    e = Event(
        event_id="toy:s", camera_id="c", sequence_id="s", frames=list(frames), partition=partition
    )
    apply_ground_truth(e, TAX)
    assign_role(e, SUPPORTED)
    return e


def test_all_empty_frames_make_an_empty_event() -> None:
    e = _event(_f("a", "empty"), _f("b", "empty"))
    assert (e.role, e.label, e.animal_present) == (Role.EMPTY, "empty", False)


def test_one_animal_frame_makes_an_animal_event() -> None:
    e = _event(_f("a", "empty"), _f("b", "raccoon"), _f("c", "empty"))
    assert (e.role, e.label, e.animal_present) == (Role.SUPPORTED, "raccoon", True)
    assert e.notes  # empty-annotated frames inside an animal event are noted


def test_excluded_duplicate_still_counts_for_ground_truth() -> None:
    """Dropping a duplicate image must never turn an animal event into an empty one."""
    e = _event(_f("a", "empty"), _f("b", "raccoon", status=Status.EXCLUDED))
    assert e.animal_present and e.label == "raccoon"
    assert [f.source_id for f in e.images] == ["a"]


def test_quarantined_frame_excludes_the_event() -> None:
    e = _event(_f("a", "raccoon"), _f("b", "coyote", status=Status.QUARANTINED))
    assert e.excluded_reason == "contains_quarantined_frame" and e.role is None


def test_conflicting_species_are_left_unresolved() -> None:
    e = _event(_f("a", "raccoon"), _f("b", "coyote"))
    assert (e.role, e.label, e.animal_present) == (Role.MIXED, None, True)


def test_unsupported_animal_stays_an_animal() -> None:
    e = _event(_f("a", "badger"))
    assert (e.role, e.label, e.animal_present) == (Role.UNSUPPORTED, "badger", True)


def test_non_animal_is_neither_empty_nor_animal() -> None:
    e = _event(_f("a", "car"), _f("b", "empty"))
    assert (e.role, e.label, e.animal_present) == (Role.NON_ANIMAL, "car", False)
    e2 = _event(_f("a", "car"), _f("b", "raccoon"))
    assert (e2.role, e2.label) == (Role.SUPPORTED, "raccoon")


def test_unknown_category_fails_loudly() -> None:
    from wildinbox.config import ConfigError

    with pytest.raises(ConfigError, match="not in taxonomy"):
        _event(_f("a", "unicorn"))


@pytest.mark.parametrize(
    ("frames", "partition", "fit_frame", "expected"),
    [
        (("raccoon",), Partition.TRAIN, 0, None),
        (("empty",), Partition.TRAIN, 0, None),
        (("raccoon",), Partition.CALIBRATION, 0, FitExclusion.NOT_TRAINING_PARTITION),
        (("badger",), Partition.TRAIN, 0, FitExclusion.EVENT_NOT_TRAINABLE),
        (("car",), Partition.TRAIN, 0, FitExclusion.EVENT_NOT_TRAINABLE),
        (("raccoon", "coyote"), Partition.TRAIN, 0, FitExclusion.EVENT_NOT_TRAINABLE),
        (("raccoon", "empty"), Partition.TRAIN, 1, FitExclusion.FRAME_LABEL_DIFFERS),
    ],
)
def test_fit_exclusions(
    frames: tuple[str, ...], partition: Partition, fit_frame: int, expected: FitExclusion | None
) -> None:
    e = _event(*(_f(str(i), lab) for i, lab in enumerate(frames)), partition=partition)
    assert fit_exclusion(e, e.frames[fit_frame]) == expected
