"""`wildinbox calibrate`: fit score calibration and choose the operating point.

Applies the pre-registered rule in configs/experiments/operating_point.yaml:

1. Fit temperature scaling on the calibration partition only.
2. On policy validation, with calibrated scores and the conservative event
   policy, take the lowest empty threshold whose false-empty rate has a Wilson
   95% upper bound within the limit, then (at that threshold) the lowest species
   threshold whose accepted-label precision has a 95% lower bound above the
   target. A step with no passing threshold leaves that automation disabled.

The final test is never read (load_rows refuses it).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from wildinbox.class_map import EMPTY_CLASS
from wildinbox.evaluation.metrics import ScoredEvent, event_metrics, wilson
from wildinbox.policy.conservative import Thresholds

# decide_event filters only when every frame has P(empty) >= threshold, so a
# threshold above 1 disables filtering; likewise for species acceptance.
DISABLED = math.inf
P_FLOOR = 1e-12  # cached scores are probabilities; log(0) would be -inf


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Grid(_Strict):
    start: float = Field(gt=0, le=1)
    stop: float = Field(gt=0, le=1)
    step: float = Field(gt=0)
    extra: list[float] = []

    def values(self) -> list[float]:
        n = round((self.stop - self.start) / self.step)
        base = [round(self.start + i * self.step, 6) for i in range(n + 1)]
        return sorted(set(base) | set(self.extra))


class CalibrationSpec(_Strict):
    method: Literal["temperature"]
    fit_partition: Literal["calibration"]
    objective: Literal["nll"]


class OperatingPointSpec(_Strict):
    choose_on: Literal["policy_validation"]
    policy: Literal["conservative"]
    interval: Literal["wilson95"]
    max_false_empty_rate: float = Field(gt=0, lt=1)
    min_accepted_precision: float = Field(gt=0, lt=1)
    empty_grid: Grid
    species_grid: Grid


class OperatingPointRule(_Strict):
    model: Path
    calibration: CalibrationSpec
    operating_point: OperatingPointSpec


def load_rule(path: Path) -> OperatingPointRule:
    return OperatingPointRule.model_validate(yaml.safe_load(path.read_text()))


# ------------------------------------------------------------ temperature


def _log_probs(probs: np.ndarray) -> np.ndarray:
    # softmax(log p / T) == softmax(z / T) for the logits z behind p, because
    # log p differs from z by a per-row constant.
    out: np.ndarray = np.log(np.clip(probs, P_FLOOR, 1.0))
    return out


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    z = _log_probs(probs) / temperature
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    out: np.ndarray = e / e.sum(axis=1, keepdims=True)
    return out


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], P_FLOOR, 1.0))))


def fit_temperature(probs: np.ndarray, y: np.ndarray, lo: float = 0.05, hi: float = 20.0) -> float:
    """Temperature minimising NLL. NLL is convex in 1/T, so a golden-section
    search over 1/T finds the optimum deterministically."""
    if len(y) == 0:
        raise ValueError("no labeled images to fit calibration on")
    logp = _log_probs(probs)

    def loss(beta: float) -> float:
        z = logp * beta
        z = z - z.max(axis=1, keepdims=True)
        lse = np.log(np.exp(z).sum(axis=1))
        return float(np.mean(lse - z[np.arange(len(y)), y]))

    a, b = 1 / hi, 1 / lo
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = loss(c), loss(d)
    for _ in range(100):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = loss(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = loss(d)
    return 1 / ((a + b) / 2)


def top_label_ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in pairwise(edges):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


EMPTY_BINS = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)


def empty_reliability(p_empty: np.ndarray, is_empty: np.ndarray) -> list[dict[str, Any]]:
    """How often images scored in each P(empty) band really are empty."""
    rows = []
    for lo, hi in pairwise(EMPTY_BINS):
        m = (p_empty >= lo) & ((p_empty < hi) if hi < 1 else (p_empty <= hi))
        n = int(m.sum())
        k = int(is_empty[m].sum())
        rows.append(
            {
                "band": [lo, hi],
                "images": n,
                "mean_p_empty": float(p_empty[m].mean()) if n else None,
                "observed_empty": k / n if n else None,
                "observed_empty_ci95": wilson(k, n) if n else None,
            }
        )
    return rows


def calibration_summary(
    raw: np.ndarray, calibrated: np.ndarray, y: np.ndarray, is_empty: np.ndarray, empty_idx: int
) -> dict[str, Any]:
    """y: class index for supported images, -1 otherwise (unsupported animals).
    is_empty: whether each image is truly empty (unsupported animals are not)."""
    sup = y >= 0
    return {
        "images": len(y),
        "supported_images": int(sup.sum()),
        "nll": {"raw": nll(raw[sup], y[sup]), "calibrated": nll(calibrated[sup], y[sup])},
        "top_label_ece": {
            "raw": top_label_ece(raw[sup], y[sup]),
            "calibrated": top_label_ece(calibrated[sup], y[sup]),
        },
        "empty_reliability": {
            "raw": empty_reliability(raw[:, empty_idx], is_empty),
            "calibrated": empty_reliability(calibrated[:, empty_idx], is_empty),
        },
    }


# ------------------------------------------------------- operating point


@dataclass(frozen=True)
class OperatingPoint:
    empty_threshold: float | None  # None: automatic filtering disabled
    species_threshold: float | None  # None: automatic acceptance disabled
    empty_sweep: list[dict[str, Any]]
    species_sweep: list[dict[str, Any]]
    metrics: dict[str, Any]


def _t(x: float | None) -> float:
    return DISABLED if x is None else x


def choose_operating_point(
    events: Sequence[ScoredEvent], spec: OperatingPointSpec
) -> OperatingPoint:
    empty_sweep = []
    chosen_empty: float | None = None
    for t in spec.empty_grid.values():
        m = event_metrics(events, Thresholds(t, DISABLED))[0]
        upper = m["false_empty_rate_ci95"][1]
        passes = m["animal_events"] > 0 and upper <= spec.max_false_empty_rate
        empty_sweep.append({**m, "passes": passes})
        if passes and chosen_empty is None:
            chosen_empty = t

    species_sweep = []
    chosen_species: float | None = None
    for s in spec.species_grid.values():
        m = event_metrics(events, Thresholds(_t(chosen_empty), s))[0]
        ci = m["accepted_precision_ci95"]
        passes = ci is not None and ci[0] >= spec.min_accepted_precision
        species_sweep.append({**m, "passes": passes})
        if passes and chosen_species is None:
            chosen_species = s

    final = event_metrics(events, Thresholds(_t(chosen_empty), _t(chosen_species)))[0]
    return OperatingPoint(chosen_empty, chosen_species, empty_sweep, species_sweep, final)


def _jsonable(m: dict[str, Any]) -> dict[str, Any]:
    th = m.get("thresholds", {})
    fixed = {k: (None if v == DISABLED else v) for k, v in th.items()}
    return {**m, "thresholds": fixed}


# ------------------------------------------------------------------ run


def run(rule_path: Path, config_path: Path, report_dir: Path) -> dict[str, Any]:
    from wildinbox.datasets.spec import Partition
    from wildinbox.evaluation.data import load_rows
    from wildinbox.evaluation.predictors import predictor_for
    from wildinbox.evaluation.run import _events, _score
    from wildinbox.settings import Settings
    from wildinbox.training.run import git_state, load_context

    rule = load_rule(rule_path)
    code = git_state()
    ctx = load_context(config_path, Settings().data_dir)
    meta = json.loads((rule.model / "meta.json").read_text())
    predictor = predictor_for(ctx, rule.model, meta["device"])
    classes = list(predictor.classes)
    if classes != ctx.classes:
        raise ValueError(f"model classes {classes} differ from config {ctx.classes}")
    empty_idx = classes.index(EMPTY_CLASS)

    fit_part = Partition(rule.calibration.fit_partition)
    choose_part = Partition(rule.operating_point.choose_on)
    rows, events = load_rows(ctx.split_dir, [fit_part, choose_part])
    scored = _score(predictor, rows)

    def arrays(part: Partition) -> tuple[list[Any], np.ndarray, np.ndarray, np.ndarray]:
        items = [s for s in scored if s.row.partition is part]
        probs = np.array([[s.probs[c] for c in classes] for s in items])
        y = np.array([classes.index(s.row.image_label or "") if s.supported else -1 for s in items])
        is_empty = np.array([s.row.image_label == EMPTY_CLASS for s in items])
        return items, probs, y, is_empty

    _, fit_probs, fit_y, _ = arrays(fit_part)
    sup = fit_y >= 0
    temperature = fit_temperature(fit_probs[sup], fit_y[sup])

    calib: dict[str, Any] = {}
    calibrated_items: list[Any] = []
    for part in (fit_part, choose_part):
        items, probs, y, is_empty = arrays(part)
        cal = apply_temperature(probs, temperature)
        calib[part.value] = calibration_summary(probs, cal, y, is_empty, empty_idx)
        for s, p in zip(items, cal, strict=True):
            calibrated_items.append((s, {c: float(v) for c, v in zip(classes, p, strict=True)}))

    def scored_events(part: Partition, calibrated: bool) -> list[ScoredEvent]:
        from dataclasses import replace

        items = [
            replace(s, probs=p) if calibrated else s
            for s, p in calibrated_items
            if s.row.partition is part
        ]
        return _events(items, {k: v for k, v in events.items() if v.partition is part})

    op = choose_operating_point(scored_events(choose_part, True), rule.operating_point)
    # For context only: the same thresholds on the calibration cameras.
    fit_side = event_metrics(
        scored_events(fit_part, True),
        Thresholds(_t(op.empty_threshold), _t(op.species_threshold)),
    )[0]

    result: dict[str, Any] = {
        "model": meta["name"],
        "rule": str(rule_path),
        "code": code,
        "split_version": meta["split_version"],
        "calibration": {
            "method": rule.calibration.method,
            "temperature": temperature,
            "fit_partition": fit_part.value,
            "fit_images": int(sup.sum()),
            "summary": calib,
        },
        "operating_point": {
            "chosen_on": choose_part.value,
            "empty_filter": op.empty_threshold,
            "species_accept": op.species_threshold,
            "auto_filter_enabled": op.empty_threshold is not None,
            "auto_accept_enabled": op.species_threshold is not None,
            "targets": {
                "max_false_empty_rate": rule.operating_point.max_false_empty_rate,
                "min_accepted_precision": rule.operating_point.min_accepted_precision,
                "interval": rule.operating_point.interval,
            },
            "metrics": _jsonable(op.metrics),
            "metrics_on_fit_partition": _jsonable(fit_side),
        },
        "sweeps": {
            "empty_filter": [_jsonable(m) for m in op.empty_sweep],
            "species_accept": [_jsonable(m) for m in op.species_sweep],
        },
    }
    from wildinbox.evaluation.run import _round

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "metrics.json").write_text(json.dumps(_round(result), indent=2) + "\n")
    (rule.model / "calibration.json").write_text(
        json.dumps(
            {
                "method": "temperature",
                "temperature": temperature,
                "operating_point": {
                    k: result["operating_point"][k]
                    for k in (
                        "empty_filter",
                        "species_accept",
                        "auto_filter_enabled",
                        "auto_accept_enabled",
                    )
                },
            },
            indent=2,
        )
        + "\n"
    )
    from wildinbox.evaluation.calibration_report import write_report

    write_report(report_dir, result)
    return result
