from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from wildinbox.evaluation.data import FinalTestAccessError, load_rows
from wildinbox.evaluation.final_test import (
    FinalTestError,
    bootstrap_macro_f1,
    compare_results,
    load_protocol,
    max_difference,
    verify,
)
from wildinbox.evaluation.metrics import image_metrics

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPO_ROOT / "configs/experiments/final_test.yaml"
LOCK = REPO_ROOT / "manifests/cct20-splits-v1.lock.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_committed_frozen_artifacts_still_match_the_protocol() -> None:
    """The final test was measured with these; changing one silently would
    detach the published results from the artifacts."""
    p = load_protocol(PROTOCOL)
    assert _sha(LOCK) == p.split_lock_sha256
    assert _sha(REPO_ROOT / p.policy_artifact.path) == p.policy_artifact.sha256
    assert _sha(REPO_ROOT / p.unfamiliar.rule) == p.unfamiliar.rule_sha256


def test_verify_refuses_changed_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for rel in ("configs/experiments", "reports/calibration", "manifests"):
        (tmp_path / rel).mkdir(parents=True)
    for rel in (
        "configs/experiments/unfamiliar.yaml",
        "reports/calibration/policy.json",
        "manifests/cct20-splits-v1.lock.json",
    ):
        shutil.copy(REPO_ROOT / rel, tmp_path / rel)
    for d, name in (
        ("models/finetune-e3-deep-balanced", "model.pt"),
        ("models/baseline-frozen-effnetb0-logreg-v1", "classifier.npz"),
    ):
        (tmp_path / d).mkdir(parents=True)
        (tmp_path / d / name).write_bytes(b"not the frozen weights")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FinalTestError, match="E3 weights"):
        verify(load_protocol(PROTOCOL), Path("manifests/cct20-splits-v1.lock.json"))


def test_development_loaders_still_refuse_the_final_test(tmp_path: Path) -> None:
    with pytest.raises(FinalTestAccessError):
        load_rows(tmp_path, ["final_test"])


def test_event_bootstrap_brackets_macro_f1_and_is_deterministic() -> None:
    classes = ["empty", "cat", "dog"]
    y_true = ["cat"] * 30 + ["dog"] * 30 + ["empty"] * 30
    y_pred = ["cat"] * 24 + ["dog"] * 6 + ["dog"] * 21 + ["cat"] * 9 + ["empty"] * 27 + ["cat"] * 3
    groups = [f"e{i // 3}" for i in range(90)]  # three frames per event
    point = image_metrics(y_true, y_pred, classes)["macro_f1"]
    lo, hi = bootstrap_macro_f1(y_true, y_pred, groups, classes, 500, 7)
    assert lo < point < hi
    assert (lo, hi) == bootstrap_macro_f1(y_true, y_pred, groups, classes, 500, 7)


RECORD = {
    "macro_f1": 0.447123,
    "events": {"needs_review": 8982, "likely_empty": 0},
    "per_class": [{"label": "raccoon", "recall": 0.61}],
    "ci": [0.41, 0.48],
    "undefined": None,
}


def test_same_device_reruns_must_be_identical() -> None:
    assert compare_results(RECORD, RECORD, 0.0) == []
    nudged = {**RECORD, "macro_f1": 0.447124}
    assert compare_results(RECORD, nudged, 0.0) == ["/macro_f1: 0.447124 vs recorded 0.447123"]
    assert max_difference(RECORD, nudged) == pytest.approx(1e-6)


def test_other_devices_may_move_numbers_only_within_the_tolerance() -> None:
    nudged = {**RECORD, "macro_f1": 0.447124, "ci": [0.410001, 0.48]}
    assert compare_results(RECORD, nudged, 1e-5) == []
    assert compare_results(RECORD, {**RECORD, "macro_f1": 0.4472}, 1e-5) != []
    # Counts, labels, and missing values are never tolerated.
    moved = {**RECORD, "events": {"needs_review": 8981, "likely_empty": 1}}
    assert len(compare_results(RECORD, moved, 1.0)) == 2
    assert compare_results(RECORD, {**RECORD, "undefined": 0.0}, 1.0) != []
    relabelled = {**RECORD, "per_class": [{"label": "bobcat", "recall": 0.61}]}
    assert compare_results(RECORD, relabelled, 1.0) == [
        "/per_class/0/label: 'bobcat' vs recorded 'raccoon'"
    ]
    assert compare_results(RECORD, {**RECORD, "ci": [0.41]}, 1.0) == ["/ci: 1 items, recorded 2"]
