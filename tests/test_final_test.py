from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest

from wildinbox.evaluation.data import FinalTestAccessError, load_rows
from wildinbox.evaluation.final_test import (
    FinalTestError,
    bootstrap_macro_f1,
    load_protocol,
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
