from __future__ import annotations

import gzip
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from PIL import Image

from wildinbox.config import ConfigError, load_config
from wildinbox.datasets.build import BuildError, build, leakage_checks
from wildinbox.datasets.events import Event, Frame
from wildinbox.datasets.report import write_split_report
from wildinbox.datasets.spec import (
    CameraAssignment,
    DiagnosticSample,
    Partition,
    SpeciesSelection,
    SplitSpec,
    Taxonomy,
    load_split_spec,
    load_taxonomy,
)
from wildinbox.ingestion.inventory import Paths, Status, ingest
from wildinbox.ingestion.sources import Archive, SourceConfig

from .conftest import REPO_ROOT

CATS = {"empty": 30, "raccoon": 3, "coyote": 9, "badger": 21, "car": 33}
TAXONOMY = {
    "name": "toy",
    "categories": {
        "empty": {"kind": "empty"},
        "raccoon": {"kind": "animal"},
        "coyote": {"kind": "animal"},
        "badger": {"kind": "animal"},
        "car": {"kind": "non_animal"},
    },
}
CAMERAS = {
    "final_test": ["F1"],
    "policy_validation": ["P1"],
    "calibration": ["C1"],
    "train": ["T1", "T2"],
}

# (camera, sequence, [frame labels]) - one entry per sequence
SEQUENCES = [
    ("T1", "t1-a", [["raccoon"], ["raccoon"]]),
    ("T1", "t1-b", [["raccoon"]]),
    ("T1", "t1-c", [["empty"], ["empty"]]),
    ("T1", "t1-d", [["coyote"]]),  # coyote only on T1 -> unsupported
    ("T1", "t1-e", [["raccoon"], ["empty"]]),  # animal event with an empty frame
    ("T2", "t2-a", [["raccoon"]]),
    ("T2", "t2-b", [["raccoon"]]),
    ("T2", "t2-c", [["car"]]),
    ("T2", "t2-d", [["raccoon"], ["coyote"]]),  # mixed species
    ("T2", "t2-e", [["empty"]]),
    ("C1", "c1-a", [["raccoon"]]),
    ("C1", "c1-b", [["empty"]]),
    ("P1", "p1-a", [["coyote"]]),
    ("P1", "p1-b", [["badger"]]),
    ("F1", "f1-a", [["raccoon"], ["raccoon"]]),
    ("F1", "f1-b", [["badger"]]),
    ("F1", "f1-c", [["empty"]]),
]


def _jpeg(seed: int) -> bytes:
    arr = np.random.default_rng(seed).integers(0, 256, (24, 32, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture
def toy(tmp_path: Path) -> tuple[SplitSpec, Taxonomy, Path, Path]:
    paths = Paths(root=tmp_path / "data", name="toy")
    paths.images.mkdir(parents=True)
    paths.annotations.mkdir(parents=True)
    images, anns = [], []
    n = 0
    for cam, seq, frames in SEQUENCES:
        for k, labels in enumerate(frames, start=1):
            n += 1
            sid = f"{seq}-{k}"
            (paths.images / f"{sid}.jpg").write_bytes(_jpeg(n))
            images.append(
                {
                    "id": sid,
                    "file_name": f"{sid}.jpg",
                    "location": cam,
                    "seq_id": seq,
                    "seq_num_frames": len(frames),
                    "frame_num": k,
                    "date_captured": f"2012-01-01 00:{n:02d}:0{k}",
                    "width": 32,
                    "height": 24,
                }
            )
            anns += [
                {"id": f"a-{sid}-{lab}", "image_id": sid, "category_id": CATS[lab]}
                for lab in labels
            ]
    (paths.annotations / "all.json").write_text(
        json.dumps(
            {
                "images": images,
                "annotations": anns,
                "categories": [{"id": v, "name": k} for k, v in CATS.items()],
            }
        )
    )
    arch = Archive(url="https://example.invalid/x", filename="x", size=1, md5="0" * 32)
    result = ingest(
        SourceConfig(
            name="toy", images_archive=arch, annotations_archive=arch, annotation_files=["all.json"]
        ),
        paths,
    )
    tax_path = tmp_path / "taxonomy.yaml"
    tax_path.write_text(yaml.safe_dump(TAXONOMY))
    spec = SplitSpec(
        name="toy-splits",
        inventory_manifest_version=result.version,
        taxonomy=tax_path,
        grouping_rule="sequence_id/v1",
        cameras=CameraAssignment(**CAMERAS),
        species_selection=SpeciesSelection(
            min_train_events=2, min_train_cameras=2, min_events_per_camera=1
        ),
    )
    return spec, load_taxonomy(tax_path), paths.database, tmp_path / "splits"


def _events(result: Any) -> dict[str, Event]:
    return {e.sequence_id: e for e in result.events}


def test_toy_build_passes_checks_and_applies_rules(toy: Any) -> None:
    spec, tax, db, out = toy
    result = build(spec, tax, db, out)
    assert result.passed, [c for c in result.checks if not c.passed]
    assert result.supported_classes == ["empty", "raccoon"]  # coyote: one training camera
    ev = _events(result)
    assert ev["t1-d"].role.value == "unsupported_animal"
    assert ev["t2-c"].role.value == "non_animal" and not ev["t2-c"].animal_present
    assert ev["t2-d"].role.value == "mixed_species" and ev["t2-d"].label is None
    assert ev["f1-a"].partition is Partition.FINAL_TEST
    with gzip.open(result.images_path) as f:
        images = {r["source_id"]: r for r in map(json.loads, f)}
    assert images["t1-e-2"]["use_for_fit"] is False  # empty frame inside an animal event
    assert images["t1-c-1"]["use_for_fit"] is True  # all-empty event
    assert images["t1-d-1"]["use_for_fit"] is False  # unsupported species never a negative
    assert not any(r["use_for_fit"] for r in images.values() if r["partition"] != "train")


def test_build_is_reproducible(toy: Any) -> None:
    spec, tax, db, out = toy
    first = build(spec, tax, db, out)
    events_bytes = first.events_path.read_bytes()
    second = build(spec, tax, db, out)
    assert second.version == first.version
    assert second.events_path.read_bytes() == events_bytes


def test_diagnostic_sample_is_deterministic_and_training_only(toy: Any) -> None:
    spec, tax, db, out = toy
    spec = spec.model_copy(
        update={
            "seen_camera_diagnostic": DiagnosticSample(fraction=0.5, salt="s"),
            "species_selection": SpeciesSelection(
                min_train_events=1, min_train_cameras=1, min_events_per_camera=1
            ),
        }
    )
    a, b = build(spec, tax, db, out), build(spec, tax, db, out)
    diag = {e.sequence_id for e in a.events if e.partition is Partition.SEEN_CAMERA_DIAGNOSTIC}
    assert diag == {
        e.sequence_id for e in b.events if e.partition is Partition.SEEN_CAMERA_DIAGNOSTIC
    }
    assert diag and all(s.startswith(("t1-", "t2-")) for s in diag)
    assert a.passed


def test_unassigned_camera_fails(toy: Any) -> None:
    spec, tax, db, out = toy
    cams = dict(CAMERAS, train=["T1"])
    with pytest.raises(BuildError, match=r"not assigned.*T2"):
        build(spec.model_copy(update={"cameras": CameraAssignment(**cams)}), tax, db, out)


def test_inventory_version_mismatch_fails(toy: Any) -> None:
    spec, tax, db, out = toy
    with pytest.raises(BuildError, match="pins"):
        build(
            spec.model_copy(update={"inventory_manifest_version": "toy-000000000000"}), tax, db, out
        )


def test_category_missing_from_taxonomy_fails(toy: Any, tmp_path: Path) -> None:
    spec, _, db, out = toy
    cats = {k: v for k, v in TAXONOMY["categories"].items() if k != "car"}
    tax = Taxonomy.model_validate({"name": "toy", "categories": cats})
    with pytest.raises(BuildError, match=r"missing from taxonomy.*car"):
        build(spec, tax, db, out)


def test_camera_in_two_partitions_is_rejected() -> None:
    with pytest.raises(ValueError, match="assigned to both"):
        CameraAssignment(final_test=["A"], policy_validation=["B"], calibration=["C"], train=["A"])


def _frame(sid: str, cam: str, seq: str, sha: str) -> Frame:
    return Frame(
        source_id=sid,
        status=Status.ACCEPTED,
        camera_id=cam,
        sequence_id=seq,
        source_file="f",
        frame_num=None,
        captured_at=None,
        storage_path=None,
        sha256=sha,
        labels=("raccoon",),
    )


def _checks(events: list[Event], near: list[tuple[str, str]] | None = None) -> dict[str, bool]:
    frames = [f for e in events for f in e.frames]
    spec = load_split_spec(REPO_ROOT / "configs/splits/cct20.yaml")
    return {c.name: c.passed for c in leakage_checks(events, frames, near or [], spec)}


def test_checks_detect_planted_leaks() -> None:
    seq_leak = [
        Event("e1", "A", "s1", [_frame("i1", "A", "s1", "h1")], partition=Partition.TRAIN),
        Event("e2", "B", "s1", [_frame("i2", "B", "s1", "h2")], partition=Partition.FINAL_TEST),
    ]
    assert not _checks(seq_leak)["no sequence id in more than one partition"]

    dup_leak = [
        Event("e1", "A", "s1", [_frame("i1", "A", "s1", "same")], partition=Partition.TRAIN),
        Event("e2", "B", "s2", [_frame("i2", "B", "s2", "same")], partition=Partition.FINAL_TEST),
    ]
    assert not _checks(dup_leak)["no exact image duplicate (SHA-256) across partitions"]
    assert not _checks(dup_leak, near=[("i1", "i2")])[
        "no suspected near-duplicate pair across partitions"
    ]

    cam_leak = [
        Event("e1", "A", "s1", [_frame("i1", "A", "s1", "h1")], partition=Partition.TRAIN),
        Event("e2", "A", "s2", [_frame("i2", "A", "s2", "h2")], partition=Partition.FINAL_TEST),
        Event("e3", "A", "s3", [_frame("i3", "A", "s3", "h3")], partition=Partition.CALIBRATION),
    ]
    c = _checks(cam_leak)
    assert not c["final-test cameras used in no other partition"]
    assert not c["development cameras separate from training cameras"]


def test_report_is_written(toy: Any, tmp_path: Path) -> None:
    spec, tax, db, out = toy
    result = build(spec, tax, db, out)
    text = write_split_report(spec, tax, result, tmp_path / "report").read_text()
    for heading in (
        "Leakage checks",
        "Partitions",
        "Supported species selection",
        "Taxonomy mapping",
        "deviations",
        "Excluded events",
        "Upload grouping rule",
        "Limitations",
    ):
        assert heading in text


def test_repository_configs_agree_with_committed_locks() -> None:
    """The run config, split spec, and locks must describe the same dataset."""
    spec = load_split_spec(REPO_ROOT / "configs/splits/cct20.yaml")
    split_lock = json.loads((REPO_ROOT / "manifests" / f"{spec.name}.lock.json").read_text())
    inv_lock = json.loads((REPO_ROOT / "manifests/cct20.lock.json").read_text())
    cfg = load_config(REPO_ROOT / "configs/example.yaml")
    assert spec.inventory_manifest_version == inv_lock["manifest_version"]
    assert split_lock["inventory_manifest_version"] == inv_lock["manifest_version"]
    assert cfg.dataset.manifest_version == inv_lock["manifest_version"]
    assert cfg.classes == split_lock["supported_classes"]
    assert split_lock["checks_passed"] is True
    load_taxonomy(REPO_ROOT / spec.taxonomy)


def test_taxonomy_requires_exactly_one_empty() -> None:
    with pytest.raises(ValueError, match="exactly one category"):
        Taxonomy.model_validate({"name": "x", "categories": {"raccoon": {"kind": "animal"}}})
    with pytest.raises(ConfigError):
        load_taxonomy(Path("does/not/exist.yaml"))
