"""`wildinbox unfamiliar`: evaluate the unfamiliar-input score against ordinary
confidence, under the pre-registered rule in configs/experiments/unfamiliar.yaml.

- Known inputs: supported-species animal images. Unknown inputs: images of the
  rule's tuning species (unsupported, never fit by the classifier).
- Each score's flag threshold is fit on `fit_on` so that at most
  `max_false_flag_rate` of known images are flagged; detection is the share of
  tuning-species images flagged. `check_on` is reported with the same thresholds.
- Held-out species are dropped before anything is summarised; they are for the
  final test only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from wildinbox.class_map import EMPTY_CLASS
from wildinbox.evaluation.metrics import wilson


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ScoreSpec(_Strict):
    features: Literal["penultimate"]
    reference: Literal["train_fit"]
    method: Literal["knn_cosine"]
    k: int = Field(ge=1)
    baseline: Literal["max_softmax"]


class SpeciesSplit(_Strict):
    tuning: list[str] = Field(min_length=1)
    held_out: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _disjoint(self) -> Self:
        both = set(self.tuning) & set(self.held_out)
        if both:
            raise ValueError(f"species cannot be both tuning and held out: {sorted(both)}")
        return self


class UnfamiliarRule(_Strict):
    model: Path
    score: ScoreSpec
    species: SpeciesSplit
    fit_on: Literal["calibration"]
    check_on: Literal["policy_validation"]
    max_false_flag_rate: float = Field(gt=0, lt=1)
    adopt_if_detection_gain: float = Field(ge=0, lt=1)


def load_rule(path: Path) -> UnfamiliarRule:
    return UnfamiliarRule.model_validate(yaml.safe_load(path.read_text()))


def flag_threshold(known: np.ndarray, max_false_flag_rate: float) -> float:
    """Smallest threshold such that at most `max_false_flag_rate` of known
    scores are > threshold (frames are flagged when score > threshold)."""
    s = np.sort(known)
    allowed = int(np.floor(max_false_flag_rate * len(s)))
    return float(s[len(s) - allowed - 1])


def auroc(known: np.ndarray, unknown: np.ndarray) -> float:
    """P(unknown score > known score), ties counted half (Mann-Whitney)."""
    scores = np.concatenate([known, unknown])
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    r_unknown = ranks[len(known) :].sum()
    n0, n1 = len(known), len(unknown)
    return float((r_unknown - n1 * (n1 + 1) / 2) / (n0 * n1))


def _rate(flags: np.ndarray) -> dict[str, Any]:
    k, n = int(flags.sum()), len(flags)
    return {"flagged": k, "n": n, "rate": k / n if n else None, "ci95": wilson(k, n)}


def summarise(
    known: np.ndarray, unknown: np.ndarray, threshold: float, per_species: dict[str, np.ndarray]
) -> dict[str, Any]:
    return {
        "false_flags": _rate(known > threshold),
        "detection": _rate(unknown > threshold),
        "auroc": auroc(known, unknown) if len(known) and len(unknown) else None,
        "detection_by_species": {s: _rate(v > threshold) for s, v in per_species.items()},
    }


def run(rule_path: Path, config_path: Path, report_dir: Path) -> dict[str, Any]:
    from wildinbox.datasets.spec import Partition
    from wildinbox.evaluation.calibration import apply_temperature
    from wildinbox.evaluation.data import load_rows
    from wildinbox.evaluation.predictors import FinetunedPredictor
    from wildinbox.evaluation.run import _round, _score
    from wildinbox.inference.unfamiliar import knn_cosine_distance, normalize
    from wildinbox.settings import Settings
    from wildinbox.training.run import git_state, load_context

    rule = load_rule(rule_path)
    code = git_state()
    ctx = load_context(config_path, Settings().data_dir)
    meta = json.loads((rule.model / "meta.json").read_text())
    predictor = FinetunedPredictor(ctx, rule.model, meta["device"])
    classes = list(predictor.classes)
    calibration = json.loads((rule.model / "calibration.json").read_text())
    temperature = float(calibration["temperature"])

    fit_part, check_part = Partition(rule.fit_on), Partition(rule.check_on)
    rows, _ = load_rows(ctx.split_dir, [Partition.TRAIN, fit_part, check_part])
    reference_rows = [r for r in rows if r.partition is Partition.TRAIN and r.use_for_fit]
    reference = normalize(predictor.features("train-fit", reference_rows).embeddings)
    ref_digest = hashlib.sha256(
        "\n".join(sorted(r.source_id for r in reference_rows)).encode()
    ).hexdigest()[:16]
    np.savez_compressed(
        rule.model / "unfamiliar-reference.npz",
        ids=np.array([r.source_id for r in reference_rows]),
        features=reference,
    )

    held_out = set(rule.species.held_out)
    tuning = set(rule.species.tuning)
    result_parts: dict[str, Any] = {}
    all_scores: dict[str, float] = {}
    thresholds: dict[str, float] = {}
    for part in (fit_part, check_part):
        prows = [r for r in rows if r.partition is part]
        scored = _score(predictor, prows)
        feats = normalize(predictor.features(part.value, prows).embeddings)
        dist = knn_cosine_distance(feats, reference, rule.score.k)
        probs = apply_temperature(
            np.array([[s.probs[c] for c in classes] for s in scored]), temperature
        )
        conf = 1.0 - probs.max(axis=1)
        all_scores.update({r.source_id: float(d) for r, d in zip(prows, dist, strict=True)})

        known = np.array([s.supported and s.row.image_label != EMPTY_CLASS for s in scored])
        is_tuning = np.array(
            [
                s.row.event_role == "unsupported_animal" and s.row.image_label in tuning
                for s in scored
            ]
        )
        # Held-out species appear in these partitions but are in neither group,
        # so they never enter a summary or a threshold.
        assert not any(
            (k or t) and s.row.image_label in held_out
            for s, k, t in zip(scored, known, is_tuning, strict=True)
        )
        labels = np.array([s.row.image_label or "" for s in scored])

        methods: dict[str, Any] = {}
        for name, score in (("knn_cosine", dist), ("max_softmax", conf)):
            if part is fit_part:
                thresholds[name] = flag_threshold(score[known], rule.max_false_flag_rate)
            per_species = {sp: score[is_tuning & (labels == sp)] for sp in sorted(tuning)}
            methods[name] = {
                "threshold": thresholds[name],
                **summarise(score[known], score[is_tuning], thresholds[name], per_species),
            }
        result_parts[part.value] = {
            "images": len(prows),
            "known_images": int(known.sum()),
            "tuning_species_images": int(is_tuning.sum()),
            "methods": methods,
        }

    fit_methods = result_parts[fit_part.value]["methods"]
    gain = (
        fit_methods["knn_cosine"]["detection"]["rate"]
        - fit_methods["max_softmax"]["detection"]["rate"]
    )
    adopted = gain >= rule.adopt_if_detection_gain
    artifact = {
        "method": rule.score.method,
        "k": rule.score.k,
        "features": rule.score.features,
        "threshold": thresholds["knn_cosine"],
        "adopted": adopted,
        "reference": {"images": len(reference_rows), "ids_digest": ref_digest},
        "weights_digest": predictor.weights_digest,
    }
    artifact["version"] = hashlib.sha256(json.dumps(artifact, sort_keys=True).encode()).hexdigest()[
        :12
    ]
    (rule.model / "unfamiliar.json").write_text(json.dumps(artifact, indent=2) + "\n")
    ids = sorted(all_scores)
    np.savez_compressed(
        rule.model / "unfamiliar-scores.npz",
        ids=np.array(ids),
        scores=np.array([all_scores[i] for i in ids], dtype=np.float32),
    )

    result = {
        "model": meta["name"],
        "rule": str(rule_path),
        "code": code,
        "split_version": meta["split_version"],
        "temperature": temperature,
        "species": rule.species.model_dump(),
        "max_false_flag_rate": rule.max_false_flag_rate,
        "adopt_if_detection_gain": rule.adopt_if_detection_gain,
        "detection_gain_on_fit": gain,
        "adopted": adopted,
        "artifact": artifact,
        "partitions": result_parts,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "metrics.json").write_text(json.dumps(_round(result), indent=2) + "\n")
    from wildinbox.evaluation.unfamiliar_report import write_report

    write_report(report_dir, result)
    return result
