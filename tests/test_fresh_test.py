from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from wildinbox.adaptation import fresh_test
from wildinbox.adaptation.data import load_camera_events
from wildinbox.adaptation.evaluate import Scored
from wildinbox.adaptation.fresh_test import (
    FreshTestError,
    PassIf,
    accepted_errors,
    camera_bootstrap,
    create_once,
    judge,
    load_protocol,
    open_fresh_test,
    open_locked,
    pooled,
    split_version,
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


def _pinned(tmp_path: Path, cameras: dict[str, object] | None = None, **method: object) -> Path:
    """A protocol whose pins point at small files in tmp_path."""
    raw = load_protocol(PROTOCOL).model_dump(mode="json")
    raw["cameras"] |= cameras or {}
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


# ------------------------------------------------------------------ locked inputs

CAMS = {"1": 3, "2": 2}  # fresh-test camera -> events, one image each


def _fresh(tmp_path: Path, quarantined: str | None = None) -> Path:
    """A fresh split as the build writes it, all fresh test; `quarantined`
    names an event whose only frame the ingest rejected (excluded, no images)."""
    split = tmp_path / "data" / "splits" / "fresh-v1"
    split.mkdir(parents=True)
    images, events = [], []
    ft = Partition.FRESH_TEST.value
    for cam, n in CAMS.items():
        for i in range(n):
            eid, sid = f"c{cam}-e{i}", f"c{cam}-f{i}"
            excluded = eid == quarantined
            events.append(
                {
                    "event_id": eid,
                    "partition": None if excluded else ft,
                    "camera_id": cam,
                    "start": f"2014-01-0{i + 1}T00:00:00",
                    "role": None if excluded else "supported_species",
                    "label": "bobcat",
                    "excluded_reason": "contains_quarantined_frame" if excluded else None,
                    "image_ids": [] if excluded else [sid],
                }
            )
            if not excluded:
                images.append(
                    {
                        "source_id": sid,
                        "event_id": eid,
                        "partition": ft,
                        "camera_id": cam,
                        "storage_path": f"{sid}.jpg",
                        "image_label": "bobcat",
                        "event_label": "bobcat",
                        "event_role": "supported_species",
                        "use_for_fit": False,
                    }
                )
    for name, recs in (("events", events), ("images", images)):
        with gzip.open(split / f"{name}.jsonl.gz", "wt") as f:
            f.writelines(json.dumps(r, sort_keys=True) + "\n" for r in recs)
    return split


def _lock(
    tmp_path: Path,
    split: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    expected: dict[str, int] | None = None,
    quarantined: int = 0,
    cameras: dict[str, dict[str, int]] | None = None,
) -> Path:
    """Split and ingest locks for `split`, as the build and ingest write them,
    and a protocol expecting their counts; returns the protocol."""
    images = sum(CAMS.values()) - quarantined
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "fresh_images.lock.json").write_text(
        json.dumps(
            {
                "manifest_version": "fresh_images-abc",
                "counts": {
                    "source_records": images + quarantined,
                    "by_status": {"accepted": images, "excluded": 0, "quarantined": quarantined},
                },
            }
        )
    )
    per_camera = cameras or {
        cam: {"events": n, "images": n - (quarantined if cam == "1" else 0)}
        for cam, n in CAMS.items()
    }
    (manifests / f"{split.name}.lock.json").write_text(
        json.dumps(
            {
                "split_version": split_version(split),
                "inventory_manifest_version": "fresh_images-abc",
                "partition": Partition.FRESH_TEST.value,
                "checks": {"no_camera_shared_with_cct20": True},
                "cameras": per_camera,
            }
        )
    )
    monkeypatch.setattr(fresh_test, "MANIFESTS", manifests)
    monkeypatch.setattr(fresh_test, "_committed_and_clean", lambda _: True)
    want = expected or {"sequences": sum(CAMS.values()), "images": images + quarantined}
    return _pinned(tmp_path, {"locations": list(CAMS), "expected": want}, n_reviewed=1)


def _open(protocol: Path, split: Path, report: Path) -> tuple[Any, Any, dict[str, Any]]:
    return open_locked(
        load_protocol(protocol), protocol, split, "fresh_images", report / "opened.json", {}
    )


def test_the_locked_split_opens_and_the_opening_records_its_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split = _fresh(tmp_path)
    protocol = _lock(tmp_path, split, monkeypatch)
    cams, rows, inputs = _open(protocol, split, tmp_path / "report")
    assert {c: len(ev) for c, ev in cams.items()} == CAMS and len(rows) == 5
    assert inputs["split_version"] == split_version(split)
    assert inputs["excluded_events"] == 0
    opened = json.loads((tmp_path / "report" / "opened.json").read_text())
    assert opened["inputs"] == {k: v for k, v in inputs.items() if not k.startswith("excluded")}
    _open(protocol, split, tmp_path / "report")  # a retry with the same inputs


def test_an_unexplained_missing_record_is_refused_before_the_test_is_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split = _fresh(tmp_path)
    protocol = _lock(tmp_path, split, monkeypatch, expected={"sequences": 6, "images": 6})
    with pytest.raises(FreshTestError, match="data contract"):
        _open(protocol, split, tmp_path / "report")
    assert not (tmp_path / "report" / "opened.json").exists()


def test_a_declared_quarantine_is_accounted_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split = _fresh(tmp_path, quarantined="c1-e2")
    protocol = _lock(tmp_path, split, monkeypatch, quarantined=1)
    cams, rows, inputs = _open(protocol, split, tmp_path / "report")
    assert len(cams["1"]) == 2 and len(rows) == 4
    assert (inputs["ingest_rejected"], inputs["excluded_events"]) == (1, 1)


def test_a_modified_label_is_refused_and_the_opening_stays_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split = _fresh(tmp_path)
    protocol = _lock(tmp_path, split, monkeypatch)
    path = split / "images.jsonl.gz"
    path.write_bytes(
        gzip.compress(gzip.decompress(path.read_bytes()).replace(b"bobcat", b"coyote", 1))
    )
    with pytest.raises(FreshTestError, match="its lock is"):
        _open(protocol, split, tmp_path / "report")
    assert (tmp_path / "report" / "opened.json").exists()


def test_a_lock_that_does_not_match_the_loaded_cameras_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split = _fresh(tmp_path)
    swapped = {"1": {"events": 2, "images": 2}, "2": {"events": 3, "images": 3}}
    protocol = _lock(tmp_path, split, monkeypatch, cameras=swapped)
    with pytest.raises(FreshTestError, match="differs from its lock"):
        _open(protocol, split, tmp_path / "report")


def test_a_retry_with_different_inputs_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split = _fresh(tmp_path)
    protocol = _lock(tmp_path, split, monkeypatch)
    _open(protocol, split, tmp_path / "report")
    opened = tmp_path / "report" / "opened.json"
    record = json.loads(opened.read_text())
    record["inputs"]["split_version"] = "fresh-v1-000000000000"
    opened.write_text(json.dumps(record))
    with pytest.raises(FreshTestError, match="different inputs"):
        _open(protocol, split, tmp_path / "report")


def test_create_once_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "opened.json"
    create_once(path, {"n": 1})
    with pytest.raises(FreshTestError):
        create_once(path, {"n": 2})
    assert json.loads(path.read_text()) == {"n": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["opened.json"]


def _runnable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: list[bool]
) -> tuple[Path, Path, Path]:
    """`run` over the locked synthetic split, with the model, the development
    check, and Git stubbed; fresh inference raises while `fail[0]` is set."""
    import wildinbox.adaptation.run as adaptation_run
    import wildinbox.training.run as training_run

    split = _fresh(tmp_path)
    protocol = _lock(tmp_path, split, monkeypatch)
    dev_metrics = tmp_path / "dev_metrics.json"
    dev_metrics.write_text(
        json.dumps(
            {
                "results": {
                    "camera_head_other_species_only@1": {"leave_one_camera_out": {"pooled": {}}}
                }
            }
        )
    )
    (tmp_path / "model" / "policy.json").write_text('{"calibration": {"temperature": 1.0}}')

    def score_frames(*args: Any) -> None:
        if args[5] == "fresh-test-v1" and fail[0]:
            raise RuntimeError("inference failed")

    def score_all(methods: Any, cams: dict[str, Any], n: int, out: Any) -> dict[str, Any]:
        per_camera = {c: ([_s(0.9, True) for _ in ev[n:]], len(ev)) for c, ev in cams.items()}
        return {k: per_camera for k in ("adapted", "release", "no_camera_reviews")}

    monkeypatch.setattr(fresh_test, "verify", lambda _: {})
    monkeypatch.setattr(fresh_test, "DEV_METRICS", dev_metrics)
    monkeypatch.setattr(fresh_test, "load_camera_events", lambda *_: ({}, []))
    monkeypatch.setattr(fresh_test, "score_frames", score_frames)
    monkeypatch.setattr(fresh_test, "_methods", lambda *_: {})
    monkeypatch.setattr(fresh_test, "_score_all", score_all)
    monkeypatch.setattr(fresh_test, "dev_check", lambda *_: {})
    monkeypatch.setattr(adaptation_run, "unseen_base", lambda *_: (None, ()))
    monkeypatch.setattr(training_run, "git_state", lambda: {"commit": "abc", "dirty": False})
    return protocol, tmp_path / "data", tmp_path / "report"


def test_an_inference_failure_leaves_the_opening_recorded_and_a_retry_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fail = [True]
    protocol, data, report = _runnable(tmp_path, monkeypatch, fail)

    def go(protocol: Path = protocol) -> dict[str, Any]:
        return fresh_test.run(
            protocol, Path("c.yaml"), data, "fresh-v1", "fresh_images", report,
            device="cpu", dev_only=False,
        )  # fmt: skip

    with pytest.raises(RuntimeError, match="inference failed"):
        go()
    opened = json.loads((report / "opened.json").read_text())
    assert opened["inputs"]["split_version"] == split_version(data / "splits" / "fresh-v1")
    assert not (report / "metrics.json").exists()

    changed = tmp_path / "changed.yaml"
    changed.write_text(protocol.read_text() + "# edited after opening\n")
    with pytest.raises(FreshTestError, match="different protocol"):
        go(changed)

    fail[0] = False
    first = go()
    assert first["inputs"]["split_version"] == opened["inputs"]["split_version"]
    assert json.loads((report / "opened.json").read_text()) == opened  # never rewritten
    assert go() == first  # a rerun reproduces the record
