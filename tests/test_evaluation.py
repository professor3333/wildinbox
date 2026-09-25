from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image, ImageFilter

from wildinbox.config import ConfigError
from wildinbox.datasets.spec import Partition
from wildinbox.evaluation.data import FinalTestAccessError, assert_development, load_rows
from wildinbox.evaluation.metrics import ScoredEvent, event_metrics, image_metrics, wilson
from wildinbox.evaluation.run import compare
from wildinbox.policy.conservative import Thresholds, decide_event
from wildinbox.schemas import Disposition, ReviewReason
from wildinbox.training.classifier import LinearClassifier, fit
from wildinbox.training.embeddings import extract, image_stats, load_cache, save_cache
from wildinbox.training.spec import ClassifierSpec, EvaluationSpec, load_baseline_config

from .conftest import EXAMPLE_CONFIG, REPO_ROOT

T = Thresholds(empty=0.9, species=0.8)


def _p(empty: float, **species: float) -> dict[str, float]:
    rest = 1 - empty - sum(species.values())
    return {"empty": empty, **species, "raccoon": species.get("raccoon", 0) + max(rest, 0)}


# ------------------------------------------------------------------ policy


def test_filters_only_when_every_frame_is_confidently_empty() -> None:
    assert decide_event([_p(0.95), _p(0.97)], T).disposition is Disposition.LIKELY_EMPTY
    kept = decide_event([_p(0.95), _p(0.89)], T)
    assert kept.disposition is Disposition.NEEDS_REVIEW and kept.blocking_frames == (1,)


def test_any_animal_frame_keeps_the_event() -> None:
    o = decide_event([_p(0.99), _p(0.99), _p(0.05, raccoon=0.95)], T)
    assert o.disposition is Disposition.SPECIES_IDENTIFIED and o.label == "raccoon"


def test_conflicting_species_go_to_review() -> None:
    frames = [
        {"empty": 0.0, "raccoon": 0.9, "coyote": 0.1},
        {"empty": 0.0, "raccoon": 0.1, "coyote": 0.9},
    ]
    o = decide_event(frames, T)
    assert o.disposition is Disposition.NEEDS_REVIEW
    assert o.reasons == [ReviewReason.CONFLICTING_FRAMES, ReviewReason.LOW_CONFIDENCE]


def test_low_confidence_and_unfamiliar_go_to_review() -> None:
    low = decide_event([{"empty": 0.3, "raccoon": 0.7}], T)
    assert low.reasons == [ReviewReason.LOW_CONFIDENCE]
    odd = decide_event([{"empty": 0.0, "raccoon": 1.0}], T, unfamiliar=[True])
    assert odd.reasons == [ReviewReason.POSSIBLE_UNKNOWN]


# ----------------------------------------------------------------- metrics


def test_image_metrics_macro_f1_and_animal_recall() -> None:
    classes = ["empty", "raccoon", "coyote"]
    y_true = ["empty", "empty", "raccoon", "raccoon", "coyote", "coyote"]
    y_pred = ["empty", "raccoon", "raccoon", "empty", "coyote", "raccoon"]
    m = image_metrics(y_true, y_pred, classes)
    assert m["confusion"]["matrix"] == [[1, 1, 0], [1, 1, 0], [0, 1, 1]]
    # F1: empty 0.5, raccoon 0.4 (p=1/3, r=1/2), coyote 2/3
    assert m["macro_f1"] == pytest.approx((0.5 + 0.4 + 2 / 3) / 3)
    assert m["min_species_recall"] == pytest.approx(0.5)  # empty excluded
    assert m["animal_image_recall"] == pytest.approx(3 / 4)
    assert m["animal_images_predicted_empty"] == 1


def test_accuracy_can_look_good_while_animals_are_missed() -> None:
    """Why accuracy is not a headline: all-empty predictions on an empty-heavy set."""
    y_true = ["empty"] * 90 + ["raccoon"] * 10
    m = image_metrics(y_true, ["empty"] * 100, ["empty", "raccoon"])
    assert m["accuracy_context_only"] == pytest.approx(0.9)
    assert m["animal_image_recall"] == 0 and m["macro_f1"] < 0.5


def test_event_metrics_count_false_empty_and_acceptance_errors() -> None:
    events = [
        ScoredEvent("a", "supported_species", "raccoon", True, [_p(0.95)]),  # false empty
        ScoredEvent("b", "empty", "empty", False, [_p(0.99)]),  # correctly filtered
        ScoredEvent("c", "supported_species", "raccoon", True, [_p(0.0, raccoon=1.0)]),
        ScoredEvent("d", "unsupported_animal", "badger", True, [_p(0.0, raccoon=1.0)]),
        ScoredEvent("e", "supported_species", "raccoon", True, [_p(0.5)]),  # review
    ]
    m, outcomes = event_metrics(events, T)
    assert m["false_empty"] == 1 and m["animal_events"] == 4
    assert m["false_empty_rate"] == pytest.approx(0.25)
    assert m["filtered"] == 2 and m["filtered_true_empty"] == 1
    assert m["accepted"] == 2 and m["accepted_correct"] == 1
    assert m["unsupported_accepted_as_known"] == 1
    assert m["needs_review"] == 1 and len(outcomes) == 5


def test_wilson_interval() -> None:
    lo, hi = wilson(0, 100)
    assert lo == 0 and 0.03 < hi < 0.04
    lo, hi = wilson(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)


# -------------------------------------------------------------- classifier


def test_classifier_matches_sklearn_and_keeps_config_order(tmp_path: Path) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    classes = ["empty", "raccoon", "coyote"]  # deliberately not alphabetical
    x = np.concatenate([rng.normal(i * 3, 1, (40, 5)) for i in range(3)]).astype(np.float32)
    y = [c for c in classes for _ in range(40)]
    spec = ClassifierSpec(kind="logistic_regression", c=1.0, class_weight="balanced", max_iter=1000)
    clf = fit(x, y, classes, spec, seed=1)
    assert clf.classes == tuple(classes)
    scaler = StandardScaler().fit(x)
    ref = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=1)
    ref.fit(scaler.transform(x), y)
    ref_p = ref.predict_proba(scaler.transform(x))[
        :, [list(ref.classes_).index(c) for c in classes]
    ]
    assert np.allclose(clf.predict_proba(x), ref_p, atol=1e-4)

    clf.save(tmp_path / "c.npz")
    again = LinearClassifier.load(tmp_path / "c.npz")
    assert np.array_equal(again.predict_proba(x), clf.predict_proba(x))
    with pytest.raises(ValueError, match="no training examples"):
        fit(x, ["empty"] * len(y), classes, spec, seed=1)


# ------------------------------------------------------------ final test


def test_final_test_is_refused_everywhere(tmp_path: Path) -> None:
    with pytest.raises(FinalTestAccessError):
        assert_development(["calibration", "final_test"])
    with pytest.raises(FinalTestAccessError):
        load_rows(tmp_path, [Partition.FINAL_TEST])
    with pytest.raises(ValueError, match="locked final test"):
        EvaluationSpec(
            partitions=[Partition.FINAL_TEST],
            unseen_camera_partitions=[Partition.CALIBRATION],
            empty_thresholds=[0.9],
            species_thresholds=[0.9],
            reference_empty_threshold=0.9,
            reference_species_threshold=0.9,
            small_animal_area=0.01,
            gallery_size=4,
        )


def test_repository_baseline_config_is_valid_and_development_only() -> None:
    cfg = load_baseline_config(REPO_ROOT / "configs/experiments/baseline.yaml")
    assert Partition.FINAL_TEST not in cfg.evaluation.partitions
    assert (REPO_ROOT / cfg.run_config) == EXAMPLE_CONFIG
    with pytest.raises(ConfigError):
        load_baseline_config(REPO_ROOT / "configs/missing.yaml")


# ------------------------------------------------------------- embeddings


def _noise(seed: int, gray: bool = False) -> Image.Image:
    arr = np.random.default_rng(seed).integers(0, 256, (120, 160, 3), dtype=np.uint8)
    if gray:
        arr[..., 1] = arr[..., 2] = arr[..., 0]
    return Image.fromarray(arr)


def test_image_stats_detect_infrared_and_blur() -> None:
    assert image_stats(_noise(1, gray=True))[0] is True
    assert image_stats(_noise(1))[0] is False
    sharp = image_stats(_noise(2))[1]
    blurred = image_stats(_noise(2).filter(ImageFilter.GaussianBlur(4)))[1]
    assert blurred < sharp / 10


def test_extract_and_cache(tmp_path: Path) -> None:
    from wildinbox.config import load_config

    paths = []
    for i in range(5):
        p = tmp_path / f"{i}.png"
        _noise(i, gray=i % 2 == 0).save(p)
        paths.append(p)

    def fake_model(x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(2, 3))  # (B, 3): deterministic stand-in for the backbone

    pre = load_config(EXAMPLE_CONFIG).preprocessing
    ext = extract(fake_model, paths, pre, batch_size=2)
    assert ext.embeddings.shape == (5, 3) and ext.night.tolist() == [True, False, True, False, True]
    again = extract(fake_model, paths, pre, batch_size=3)
    assert np.array_equal(ext.embeddings, again.embeddings)  # batch size doesn't matter
    ids = [p.name for p in paths]
    save_cache(tmp_path / "c.npz", ids, ext)
    cached = load_cache(tmp_path / "c.npz", ids)
    assert cached is not None and np.array_equal(cached.embeddings, ext.embeddings)
    assert load_cache(tmp_path / "c.npz", ids[::-1]) is None  # different images -> recompute


# ---------------------------------------------------------- reproducibility


def _metrics(f1: float, recall: float, fe: float) -> dict[str, Any]:
    return {
        "groups": {
            "g": {
                "image": {"macro_f1": f1, "per_class": {"empty": {"recall": recall}}},
                "event_empty_sweep": [
                    {"thresholds": {"empty_filter": 0.9}, "false_empty_rate": fe}
                ],
            }
        }
    }


def test_compare_applies_tolerances() -> None:
    tol = load_baseline_config(REPO_ROOT / "configs/experiments/baseline.yaml").reproducibility
    ref = _metrics(0.70, 0.80, 0.010)
    assert compare(_metrics(0.702, 0.805, 0.012), ref, tol) == []
    problems = compare(_metrics(0.72, 0.80, 0.010), ref, tol)
    assert len(problems) == 1 and "macro-F1" in problems[0]
    assert compare(_metrics(0.70, 0.83, 0.02), ref, tol) != []
