from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from wildinbox.evaluation.unfamiliar import UnfamiliarRule, auroc, flag_threshold, load_rule
from wildinbox.inference.unfamiliar import knn_cosine_distance, normalize

REPO_ROOT = Path(__file__).resolve().parents[1]
RULE = REPO_ROOT / "configs/experiments/unfamiliar.yaml"


@pytest.mark.parametrize("rate", [0.0, 0.05, 0.1, 0.333])
def test_flag_threshold_respects_the_false_flag_budget(rate: float) -> None:
    known = np.random.default_rng(0).normal(size=997)
    t = flag_threshold(known, rate)
    assert (known > t).mean() <= rate
    # And it is the tightest such threshold: one step lower would exceed it.
    assert (known >= t).sum() == int(np.floor(rate * len(known))) + 1


def test_auroc_matches_pairwise_count_with_ties() -> None:
    rng = np.random.default_rng(1)
    known = rng.integers(0, 5, 60).astype(float)
    unknown = rng.integers(2, 7, 40).astype(float)
    pairs = [(u > k) + 0.5 * (u == k) for u in unknown for k in known]
    assert auroc(known, unknown) == pytest.approx(np.mean(pairs))


def test_knn_distance_is_zero_for_training_points_and_grows_away() -> None:
    ref = normalize(np.random.default_rng(2).normal(size=(50, 8)))
    assert np.allclose(knn_cosine_distance(ref[:5], ref, k=1), 0, atol=1e-6)
    far = normalize(-ref[:5].mean(axis=0, keepdims=True))
    assert knn_cosine_distance(far, ref, k=3)[0] > knn_cosine_distance(ref[:1], ref, k=3)[0]
    with pytest.raises(ValueError):
        knn_cosine_distance(ref, ref, k=51)


def test_committed_rule_is_valid_and_splits_species() -> None:
    rule = load_rule(RULE)
    assert not set(rule.species.tuning) & set(rule.species.held_out)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("fit_on", "final_test"),
        ("check_on", "final_test"),
        ("species", {"tuning": ["squirrel"], "held_out": ["squirrel", "skunk"]}),
    ],
)
def test_rule_rejects_final_test_and_overlapping_species(key: str, value: object) -> None:
    raw = yaml.safe_load(RULE.read_text())
    raw[key] = value
    with pytest.raises(ValidationError):
        UnfamiliarRule.model_validate(raw)
