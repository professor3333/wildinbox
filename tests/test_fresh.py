"""Fresh cameras for the v2 experiment: events, leakage checks against CCT20,
and the locked fresh-test partition."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from PIL import Image
from pydantic import ValidationError

from wildinbox.datasets.fresh import FreshSpec, build_fresh, lock_payload
from wildinbox.datasets.spec import Partition, load_taxonomy
from wildinbox.evaluation.data import FinalTestAccessError, assert_development
from wildinbox.ingestion.inventory import Paths, ingest
from wildinbox.ingestion.sources import SourceConfig

CATS = {"empty": 30, "raccoon": 3, "badger": 21, "lizard": 40}
TAXONOMY = {
    "name": "toy",
    "categories": {
        "empty": {"kind": "empty"},
        "raccoon": {"kind": "animal"},
        "badger": {"kind": "animal"},
        "lizard": {"kind": "animal"},
    },
}
REFERENCE = [("R1", "r1-a", [["raccoon"], ["raccoon"]]), ("R1", "r1-b", [["empty"]])]
FRESH = [
    ("N1", "n1-a", [["raccoon"], ["raccoon"]]),
    ("N1", "n1-b", [["lizard"]]),  # a category only outside CCT20: unsupported
    ("N2", "n2-a", [["empty"], ["empty"]]),
    ("N2", "n2-b", [["badger"]]),
]


def _jpeg(seed: int) -> bytes:
    arr = np.random.default_rng(seed).integers(0, 256, (24, 32, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG")
    return buf.getvalue()


def _inventory(root: Path, name: str, sequences: list[Any], seed: int) -> str:
    paths = Paths(root=root, name=name)
    paths.images.mkdir(parents=True)
    paths.annotations.mkdir(parents=True)
    images, anns = [], []
    for n, (cam, seq, frames) in enumerate(sequences):
        for k, labels in enumerate(frames, start=1):
            sid = f"{seq}-{k}"
            (paths.images / f"{sid}.jpg").write_bytes(_jpeg(seed + 10 * n + k))
            images.append(
                {
                    "id": sid,
                    "file_name": f"{sid}.jpg",
                    "location": cam,
                    "seq_id": seq,
                    "seq_num_frames": len(frames),
                    "frame_num": k,
                    "date_captured": f"2014-05-01 00:{n:02d}:0{k}",
                    "width": 32,
                    "height": 24,
                }
            )
            anns += [
                {"id": f"a-{sid}-{lab}", "image_id": sid, "category_id": CATS[lab]}
                for lab in labels
            ]
    (paths.annotations / f"{name}.json").write_text(
        json.dumps(
            {
                "images": images,
                "annotations": anns,
                "categories": [{"id": v, "name": k} for k, v in CATS.items()],
            }
        )
    )
    source = SourceConfig(name=name, annotation_files=[f"{name}.json"])  # no archives
    return ingest(source, paths).version


def _spec(tmp: Path, version: str, **overrides: Any) -> FreshSpec:
    tax = tmp / "taxonomy.yaml"
    tax.write_text(yaml.safe_dump(TAXONOMY))
    selection = tmp / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "adaptation_development": [{"location": "N1"}, {"location": "N2"}],
                "fresh_test": [{"location": "N1"}, {"location": "N2"}],
            }
        )
    )
    lock = tmp / "split.lock.json"
    lock.write_text(json.dumps({"supported_classes": ["empty", "raccoon"]}))
    fields: dict[str, Any] = {
        "name": "toy-fresh",
        "source": "fresh",
        "inventory_manifest_version": version,
        "taxonomy": tax,
        "grouping_rule": "sequence_id/v1",
        "partition": "adaptation_development",
        "selection": selection,
        "supported_from": lock,
        "reference_source": "ref",
    }
    return FreshSpec(**(fields | overrides))


@pytest.fixture
def data(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "data"
    _inventory(root, "ref", REFERENCE, seed=0)
    return root, _inventory(root, "fresh", FRESH, seed=1000)


def test_fresh_events_use_the_released_classes_and_pass_leakage_checks(
    data: tuple[Path, str], tmp_path: Path
) -> None:
    root, version = data
    spec = _spec(tmp_path, version)
    result = build_fresh(spec, load_taxonomy(spec.taxonomy), root)
    assert result.passed, [c for c in result.checks if not c.passed]
    ev = {e.sequence_id: e for e in result.events}
    assert ev["n1-a"].role.value == "supported_species"
    assert ev["n1-b"].role.value == "unsupported_animal" and ev["n1-b"].label == "lizard"
    assert ev["n2-b"].role.value == "unsupported_animal"  # badger: not a released class
    assert {e.partition for e in result.events} == {Partition.ADAPTATION_DEVELOPMENT}
    cams = lock_payload(spec, result)["cameras"]
    assert cams["N1"]["supported_species"] == 1  # development: composition recorded


def test_a_copy_of_a_cct20_image_fails_the_leakage_checks(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _inventory(root, "ref", REFERENCE, seed=0)
    # n1-a-1 gets r1-a-1's pixels (seed 0 + 10*0 + 1), so the file is identical
    version = _inventory(root, "fresh", FRESH, seed=0)
    spec = _spec(tmp_path, version)
    checks = {c.name: c for c in build_fresh(spec, load_taxonomy(spec.taxonomy), root).checks}
    assert not checks["no_file_shared_with_cct20"].passed
    assert not checks["no_near_duplicate_of_cct20"].passed


def test_a_camera_outside_the_selection_fails(data: tuple[Path, str], tmp_path: Path) -> None:
    root, version = data
    spec = _spec(tmp_path, version)
    spec.selection.write_text(json.dumps({"adaptation_development": [{"location": "N1"}]}))
    checks = {c.name: c for c in build_fresh(spec, load_taxonomy(spec.taxonomy), root).checks}
    assert not checks["cameras_are_the_selected_group"].passed


def test_fresh_test_lock_records_totals_only(data: tuple[Path, str], tmp_path: Path) -> None:
    root, version = data
    spec = _spec(tmp_path, version, partition="fresh_test")
    result = build_fresh(spec, load_taxonomy(spec.taxonomy), root)
    cams = lock_payload(spec, result)["cameras"]
    assert cams == {"N1": {"events": 2, "images": 3}, "N2": {"events": 2, "images": 3}}


def test_locked_partitions_are_refused_in_development(tmp_path: Path) -> None:
    for locked in ("final_test", "fresh_test"):
        with pytest.raises(FinalTestAccessError):
            assert_development([locked])
    assert assert_development(["adaptation_development"]) == [Partition.ADAPTATION_DEVELOPMENT]
    with pytest.raises(ValidationError):
        _spec(tmp_path, "v", partition="final_test")
