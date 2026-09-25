"""The committed sample batch is complete, decodable, licensed, and never
drawn from the locked final test."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

SAMPLE = Path(__file__).resolve().parents[1] / "samples/cct-dev"


def test_sample_batch_is_complete_and_licensed() -> None:
    images = sorted(p.name for p in (SAMPLE / "images").iterdir())
    meta = json.loads((SAMPLE / "metadata.json").read_text())["files"]
    truth = list(csv.DictReader((SAMPLE / "truth.csv").open()))
    assert images and sorted(meta) == images == sorted(r["filename"] for r in truth)
    assert all(m["camera_id"] and m["sequence_id"] and m["captured_at"] for m in meta.values())
    assert {r["partition"] for r in truth} <= {"calibration", "policy_validation"}
    license_text = (SAMPLE / "LICENSE.md").read_text()
    assert (
        "Community Data License Agreement" in license_text
        and "Caltech Camera Traps" in license_text
    )
    assert all(r["rights_holder"] for r in truth)


def test_sample_images_decode_and_cover_the_supported_classes() -> None:
    for p in (SAMPLE / "images").iterdir():
        with Image.open(p) as img:
            img.load()
    truth = list(csv.DictReader((SAMPLE / "truth.csv").open()))
    labels = {r["image_label"] for r in truth}
    supported = {"empty", "bobcat", "cat", "coyote", "dog", "opossum", "rabbit", "raccoon"}
    assert supported <= labels
    assert any(r["event_role"] == "unsupported_animal" for r in truth)
