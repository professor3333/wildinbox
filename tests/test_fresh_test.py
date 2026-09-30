from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from wildinbox.adaptation.data import load_camera_events
from wildinbox.adaptation.evaluate import Scored
from wildinbox.adaptation.fresh_test import (
    FreshTestError,
    PassIf,
    accepted_errors,
    camera_bootstrap,
    judge,
    load_protocol,
    open_fresh_test,
    pooled,
    verify,
)
from wildinbox.datasets.spec import Partition
from wildinbox.evaluation.data import FinalTestAccessError

PROTOCOL = Path("configs/experiments/fresh_test.yaml")
RULE = PassIf(min_accepted=30, precision_point=0.95, precision_wilson_lower=0.90)


def _s(conf: float, correct: bool, role: str = "supported_species", empty: float = 0.99) -> Scored:
    return Scored("c", role, "bobcat", empty, "bobcat" if correct else "coyote", conf)


def test_the_committed_protocol_is_the_frozen_species_only_head() -> None:
    p = load_protocol(PROTOCOL)
    assert p.method.name == "camera_head_other"
    assert p.method.thresholds.empty is None
    assert (p.method.c, p.method.camera_share, p.method.n_reviewed) == (0.001, 0.1, 50)
    assert p.method.thresholds.species == 0.82
    assert p.target.pass_if == RULE
    assert len(p.cameras.locations) == 12


def _pinned(tmp_path: Path, **method: object) -> Path:
    """A protocol whose pins point at small files in tmp_path."""
    raw = load_protocol(PROTOCOL).model_dump(mode="json")
    model = tmp_path / "model"
    model.mkdir()
    files = {
        "rule": tmp_path / "rule.yaml",
        "manifest": tmp_path / "manifest.json",
        "weights": model / "model.pt",
        "policy": model / "policy.json",
    }
    files["rule"].write_text("rule")
    files["manifest"].write_text(
        json.dumps({"fresh_test": [{"location": c} for c in raw["cameras"]["locations"]]})
    )
    files["weights"].write_bytes(b"weights")
    files["policy"].write_text("{}")

    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    raw["cameras"] |= {
        "rule": str(files["rule"]),
        "rule_sha256": sha(files["rule"]),
        "manifest": str(files["manifest"]),
        "manifest_sha256": sha(files["manifest"]),
    }
    raw["model"] = {
        "dir": str(model),
        "weights_sha256": sha(files["weights"]),
        "policy_sha256": sha(files["policy"]),
    }
    raw["method"] |= method
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_verify_passes_when_every_pin_matches(tmp_path: Path) -> None:
    assert set(verify(load_protocol(_pinned(tmp_path)))) == {
        "camera rule",
        "camera manifest",
        "model weights",
        "policy artifact",
    }


def test_verify_refuses_a_changed_artifact(tmp_path: Path) -> None:
    protocol = load_protocol(_pinned(tmp_path))
    (protocol.model.dir / "model.pt").write_bytes(b"retrained")
    with pytest.raises(FreshTestError, match="model weights"):
        verify(protocol)


def test_verify_refuses_a_method_this_command_does_not_implement(tmp_path: Path) -> None:
    protocol = load_protocol(_pinned(tmp_path, thresholds={"empty": 0.78, "species": 0.82}))
    with pytest.raises(FreshTestError, match="species-only"):
        verify(protocol)


def test_species_only_never_filters_even_confident_empties() -> None:
    per_camera = {"a": ([_s(0.9, True) for _ in range(10)], 60)}
    r = pooled(per_camera, 0.82)
    assert r["pooled"]["filtered"] == 0
    assert r["pooled"]["accepted"] == 10
    assert r["pooled"]["retention"] == 1.0


def test_target_needs_enough_labels_the_point_and_the_lower_bound() -> None:
    def res(ok: int, n: int) -> dict[str, int]:
        return {"accepted": n, "accepted_correct": ok}

    assert judge(res(29, 29), RULE) == "inconclusive"
    assert judge(res(200, 205), RULE) == "pass"  # 97.6%, lower bound 94.4%
    assert judge(res(188, 200), RULE) == "fail"  # 94.0% point
    assert judge(res(30, 31), RULE) == "fail"  # 96.8%, lower bound 83.8%


def test_camera_bootstrap_is_seeded_and_brackets_the_pooled_rate() -> None:
    outcomes = {
        str(i): {
            "automated": 10 * i,
            "total_events": 100,
            "accepted": 10 * i,
            "accepted_correct": 9 * i,
        }
        for i in range(1, 7)
    }
    a = camera_bootstrap(outcomes, 500, 1)
    assert a == camera_bootstrap(outcomes, 500, 1)
    assert a["precision"] == pytest.approx([0.9, 0.9])  # every camera is 90%
    lo, hi = a["review_reduction"]
    assert lo < 0.35 * 0.95 < hi


def test_camera_bootstrap_counts_resamples_without_accepted_labels() -> None:
    outcomes = {"a": {"automated": 0, "total_events": 50, "accepted": 0, "accepted_correct": 0}}
    b = camera_bootstrap(outcomes, 20, 0)
    assert b["precision"] is None and b["precision_undefined_resamples"] == 20


def test_accepted_errors_name_what_the_event_really_was() -> None:
    bird = Scored("c", "unsupported_animal", "bird", 0.0, "bobcat", 0.95)
    errors = accepted_errors(
        {"a": ([_s(0.95, False), bird, _s(0.95, True), _s(0.5, False)], 4)}, 0.82
    )
    assert errors == {"bobcat called coyote": 1, "unsupported_animal called bobcat": 1}


def _split(tmp_path: Path) -> Path:
    split = tmp_path / "split"
    split.mkdir()
    images, events = [], []
    for part in (Partition.ADAPTATION_DEVELOPMENT, Partition.FRESH_TEST):
        cam = "1" if part == Partition.FRESH_TEST else "2"
        for i in range(3):
            eid, sid = f"{part}-e{i}", f"{part}-f{i}"
            images.append(
                {
                    "source_id": sid,
                    "event_id": eid,
                    "partition": part.value,
                    "camera_id": cam,
                    "storage_path": f"{sid}.jpg",
                    "image_label": "bobcat",
                    "event_label": "bobcat",
                    "event_role": "supported_species",
                    "use_for_fit": False,
                }
            )
            events.append(
                {
                    "event_id": eid,
                    "partition": part.value,
                    "camera_id": cam,
                    "start": f"2014-01-0{i + 1}T00:00:00",
                    "role": "supported_species",
                    "label": "bobcat",
                    "excluded_reason": None,
                    "image_ids": [sid],
                }
            )
    for name, recs in (("images", images), ("events", events)):
        with gzip.open(split / f"{name}.jsonl.gz", "wt") as f:
            f.writelines(json.dumps(r) + "\n" for r in recs)
    return split


def test_only_the_fresh_test_path_opens_the_locked_partition(tmp_path: Path) -> None:
    split = _split(tmp_path)
    with pytest.raises(FinalTestAccessError):
        load_camera_events(split, [Partition.FRESH_TEST])
    cams, rows = open_fresh_test(split)
    assert list(cams) == ["1"] and len(cams["1"]) == 3
    assert {r.partition for r in rows} == {Partition.FRESH_TEST}
