from __future__ import annotations

import pytest

from wildinbox.class_map import ClassMap
from wildinbox.config import load_config

from .conftest import EXAMPLE_CONFIG


def test_order_comes_from_config() -> None:
    cfg = load_config(EXAMPLE_CONFIG)
    cm = cfg.class_map
    assert cm.names == tuple(cfg.classes)
    for i, name in enumerate(cfg.classes):
        assert cm.index_of(name) == i
        assert cm.name_of(i) == name
    assert "empty" not in cm.species
    assert cm.name_of(cm.empty_index) == "empty"


def test_fingerprint_tracks_order() -> None:
    a = ClassMap(["empty", "raccoon", "coyote"])
    b = ClassMap(["empty", "coyote", "raccoon"])
    assert a.fingerprint() == ClassMap(["empty", "raccoon", "coyote"]).fingerprint()
    assert a.fingerprint() != b.fingerprint()
    assert a != b


def test_unknown_label_error_lists_supported_classes() -> None:
    cm = ClassMap(["empty", "raccoon"])
    with pytest.raises(KeyError, match=r"'badger' is not a supported class.*raccoon"):
        cm.index_of("badger")
    with pytest.raises(IndexError):
        cm.name_of(2)


@pytest.mark.parametrize("classes", [["raccoon"], ["empty", "raccoon", "empty"]])
def test_rejects_invalid(classes: list[str]) -> None:
    with pytest.raises(ValueError):
        ClassMap(classes)
