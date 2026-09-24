"""Guardrails: class names and preprocessing must not be reimplemented
outside their owning modules."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "wildinbox"


def _sources_except(*allowed: str) -> list[tuple[Path, str]]:
    return [
        (p, p.read_text())
        for p in sorted(SRC.rglob("*.py"))
        if p.relative_to(SRC).as_posix() not in allowed
    ]


@pytest.mark.parametrize(
    "pattern",
    [r"\bNormalize\(", r"\bResize\(", r"CenterCrop\(", r"RandomResizedCrop\(", r"\.transforms\(\)"],
)
def test_transforms_only_in_preprocessing(pattern: str) -> None:
    offenders = [p for p, text in _sources_except("preprocessing.py") if re.search(pattern, text)]
    assert not offenders, f"{pattern} used outside preprocessing.py: {offenders}"


def test_empty_class_literal_only_in_class_map() -> None:
    offenders = [
        p for p, text in _sources_except("class_map.py") if re.search(r"[\"']empty[\"']", text)
    ]
    assert not offenders, f"use class_map.EMPTY_CLASS instead of the literal in: {offenders}"


def test_normalization_constants_not_hard_coded() -> None:
    offenders = [p for p, text in _sources_except() if "0.485" in text or "0.229" in text]
    assert not offenders, f"normalization constants belong in configs/, found in: {offenders}"
