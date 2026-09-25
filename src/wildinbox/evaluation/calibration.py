"""`wildinbox calibrate`: fit score calibration and choose the operating point.

Applies a pre-registered rule (configs/experiments/operating_point_v2.yaml;
v1 is kept as the earlier record):

1. Fit temperature scaling on the calibration partition only.
2. On policy validation, with calibrated scores and the conservative event
   policy, take the lowest empty threshold whose false-empty rate has a Wilson
   95% upper bound within the limit, then (at that threshold) the lowest species
   threshold whose accepted-label precision has a 95% lower bound above the
   target. A step with no passing threshold leaves that automation disabled.
3. v2: the policy is conservative/v1 with unfamiliar-input flags (if adopted),
   and each species is enabled only if its own accepted events pass the bound.
4. Write the versioned policy artifact and every event's saved predictions and
   decisions, so each disposition can be replayed (wildinbox replay).

The final test is never read (load_rows refuses it).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from wildinbox.class_map import EMPTY_CLASS
from wildinbox.evaluation.metrics import ScoredEvent, event_metrics, wilson
from wildinbox.inference.calibration import P_FLOOR, apply_temperature, log_probs
from wildinbox.policy.conservative import Thresholds

# decide_event filters only when every frame has P(empty) >= threshold, so a
# threshold above 1 disables filtering; likewise for species acceptance.
DISABLED = math.inf


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
    policy: Literal["conservative", "conservative/v1"]
    interval: Literal["wilson95"]
    max_false_empty_rate: float = Field(gt=0, lt=1)
    min_accepted_precision: float = Field(gt=0, lt=1)
    empty_grid: Grid
    species_grid: Grid
    per_species_gate: bool = False
    unfamiliar_rule: Path | None = None


class OperatingPointRule(_Strict):
    version: Literal[1, 2] = 1
    model: Path
    calibration: CalibrationSpec
    operating_point: OperatingPointSpec


def load_rule(path: Path) -> OperatingPointRule:
    return OperatingPointRule.model_validate(yaml.safe_load(path.read_text()))


class Deviation(_Strict):
    """A recorded decision to release less automation than the rule allows.
    It can only switch automation off, never enable what the rule rejected."""

    disable_auto_filter: bool = False
    disable_auto_accept: bool = False
    reason: str = Field(min_length=20)
    decided: date


def load_deviation(path: Path | None) -> Deviation | None:
    if path is None or not path.exists():
        return None
    return Deviation.model_validate(yaml.safe_load(path.read_text()))


def released(
    empty_threshold: float | None, species_threshold: float | None, dev: Deviation | None
) -> dict[str, bool]:
    filter_ok = empty_threshold is not None
    accept_ok = species_threshold is not None
    return {
        "auto_filter_enabled": filter_ok and not (dev is not None and dev.disable_auto_filter),
        "auto_accept_enabled": accept_ok and not (dev is not None and dev.disable_auto_accept),
    }


# ------------------------------------------------------------ temperature


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], P_FLOOR, 1.0))))


def fit_temperature(probs: np.ndarray, y: np.ndarray, lo: float = 0.05, hi: float = 20.0) -> float:
    """Temperature minimising NLL. NLL is convex in 1/T, so a golden-section
    search over 1/T finds the optimum deterministically."""
    if len(y) == 0:
        raise ValueError("no labeled images to fit calibration on")
    logp = log_probs(probs)

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
    # Species enabled by the per-species gate (None: no gate).
    accept_species: tuple[str, ...] | None = None
    species_gate: dict[str, Any] | None = None


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

    chosen = Thresholds(_t(chosen_empty), _t(chosen_species))
    accept: tuple[str, ...] | None = None
    gate: dict[str, Any] | None = None
    if spec.per_species_gate:
        # Each species must pass the precision bound on its OWN accepted events.
        by_species = (
            event_metrics(events, chosen)[0]["accepted_by_species"]
            if chosen_species is not None
            else {}
        )
        gate = {}
        for sp, v in by_species.items():
            ci = wilson(v["correct"], v["accepted"])
            gate[sp] = {**v, "precision_ci95": ci, "passes": ci[0] >= spec.min_accepted_precision}
        accept = tuple(sp for sp, g in gate.items() if g["passes"])
    final = event_metrics(events, chosen, accept)[0]
    return OperatingPoint(
        chosen_empty, chosen_species, empty_sweep, species_sweep, final, accept, gate
    )


def _jsonable(m: dict[str, Any]) -> dict[str, Any]:
    th = m.get("thresholds", {})
    fixed = {k: (None if v == DISABLED else v) for k, v in th.items()}
    return {**m, "thresholds": fixed}


# ------------------------------------------------------------------ run


def _version(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


def _outcome(o: Any) -> dict[str, Any]:
    return {
        "disposition": o.disposition.value,
        "label": o.label,
        "confidence": o.confidence,
        "reasons": [r.value for r in o.reasons],
    }


def run(
    rule_path: Path, config_path: Path, report_dir: Path, deviation_path: Path | None = None
) -> dict[str, Any]:
    from wildinbox.datasets.spec import Partition
    from wildinbox.evaluation.data import load_rows
    from wildinbox.evaluation.predictors import predictor_for
    from wildinbox.evaluation.run import _round, _score
    from wildinbox.policy.conservative import POLICY_NAME, Frame, PolicyConfig, decide
    from wildinbox.settings import Settings
    from wildinbox.training.run import git_state, load_context

    rule = load_rule(rule_path)
    deviation = load_deviation(deviation_path)
    code = git_state()
    ctx = load_context(config_path, Settings().data_dir)
    meta = json.loads((rule.model / "meta.json").read_text())
    predictor = predictor_for(ctx, rule.model, meta["device"])
    classes = list(predictor.classes)
    if classes != ctx.classes:
        raise ValueError(f"model classes {classes} differ from config {ctx.classes}")
    empty_idx = classes.index(EMPTY_CLASS)

    # Unfamiliar-input flags, if the rule names the score and it was adopted.
    unfamiliar: dict[str, Any] | None = None
    distance: dict[str, float] = {}
    if rule.operating_point.unfamiliar_rule is not None:
        unfamiliar = json.loads((rule.model / "unfamiliar.json").read_text())
        with np.load(rule.model / "unfamiliar-scores.npz") as z:
            distance = dict(zip(z["ids"].tolist(), z["scores"].tolist(), strict=True))
    flag_threshold = unfamiliar["threshold"] if unfamiliar and unfamiliar["adopted"] else None

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
    calibrated: dict[str, dict[str, float]] = {}
    for part in (fit_part, choose_part):
        items, probs, y, is_empty = arrays(part)
        cal = apply_temperature(probs, temperature)
        calib[part.value] = calibration_summary(probs, cal, y, is_empty, empty_idx)
        for s, p in zip(items, cal, strict=True):
            calibrated[s.row.source_id] = {c: float(v) for c, v in zip(classes, p, strict=True)}

    by_image = {s.row.source_id: s for s in scored}

    def frame_ids(part: Partition) -> list[tuple[Any, list[str]]]:
        out = []
        for ev in events.values():
            if ev.partition is part:
                ids = [i for i in ev.image_ids if i in by_image]
                if ids:
                    out.append((ev, ids))
        return out

    def scored_events(part: Partition) -> list[ScoredEvent]:
        return [
            ScoredEvent(
                ev.event_id,
                ev.role,
                ev.label,
                ev.animal_present,
                [calibrated[i] for i in ids],
                tuple(distance[i] > flag_threshold for i in ids)
                if flag_threshold is not None
                else None,
            )
            for ev, ids in frame_ids(part)
        ]

    op = choose_operating_point(scored_events(choose_part), rule.operating_point)
    rule_t = Thresholds(_t(op.empty_threshold), _t(op.species_threshold))
    # Replication check: the chosen thresholds on the calibration cameras (not
    # used to choose them) and on every development camera separately.
    fit_events = scored_events(fit_part)
    fit_side = event_metrics(fit_events, rule_t, op.accept_species)[0]
    sweeps_fit = {
        "empty_filter": [
            _jsonable(event_metrics(fit_events, Thresholds(t, DISABLED))[0])
            for t in rule.operating_point.empty_grid.values()
        ],
        "species_accept": [
            _jsonable(event_metrics(fit_events, Thresholds(_t(op.empty_threshold), sp))[0])
            for sp in rule.operating_point.species_grid.values()
        ],
    }
    per_camera = {}
    for part in (fit_part, choose_part):
        evs = scored_events(part)
        cams = {e.event_id: events[e.event_id].camera_id for e in evs}
        for cam in sorted(set(cams.values()), key=lambda c: (len(c), c)):
            m = event_metrics(
                [e for e in evs if cams[e.event_id] == cam], rule_t, op.accept_species
            )[0]
            per_camera[cam] = {"partition": part.value, **_jsonable(m)}
    enabled = released(op.empty_threshold, op.species_threshold, deviation)

    # The versioned decision policy: what is released, and what the rule chose.
    released_cfg = PolicyConfig(
        op.empty_threshold,
        op.species_threshold,
        enabled["auto_filter_enabled"],
        enabled["auto_accept_enabled"],
        op.accept_species,
    )
    rule_cfg = PolicyConfig(
        op.empty_threshold,
        op.species_threshold,
        op.empty_threshold is not None,
        op.species_threshold is not None,
        op.accept_species,
    )
    calibration_artifact = {
        "method": "temperature",
        "temperature": temperature,
        "fit_partition": fit_part.value,
        "weights_digest": getattr(predictor, "weights_digest", None),
    }
    calibration_artifact["version"] = _version(calibration_artifact)
    policy = {
        "policy": POLICY_NAME,
        "model": meta["name"],
        "classes": classes,
        "calibration": calibration_artifact,
        "unfamiliar": (
            {k: unfamiliar[k] for k in ("version", "method", "k", "threshold", "adopted")}
            if unfamiliar
            else None
        ),
        "released": {**asdict(released_cfg), "policy_version": _pv(released_cfg)},
        "rule": {**asdict(rule_cfg), "policy_version": _pv(rule_cfg)},
    }
    policy["artifact_version"] = _version(policy)

    # Saved predictions and decisions for every development event (replayable).
    report_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(report_dir / "decisions.jsonl.gz", "wt") as f:
        for part in (fit_part, choose_part):
            for ev, ids in frame_ids(part):
                frames = [
                    Frame(
                        calibrated[i],
                        unfamiliar=bool(
                            flag_threshold is not None and distance[i] > flag_threshold
                        ),
                    )
                    for i in ids
                ]
                row = {
                    "event_id": ev.event_id,
                    "partition": part.value,
                    "camera_id": ev.camera_id,
                    "frames": [
                        {
                            "image_id": i,
                            "status": "completed",
                            "raw_probs": [by_image[i].probs[c] for c in classes],
                            "unfamiliar_distance": distance.get(i),
                        }
                        for i in ids
                    ],
                    "released": _outcome(decide(frames, released_cfg)),
                    "rule": _outcome(decide(frames, rule_cfg)),
                }
                f.write(json.dumps(row) + "\n")

    result: dict[str, Any] = {
        "model": meta["name"],
        "rule": str(rule_path),
        "rule_version": rule.version,
        "code": code,
        "split_version": meta["split_version"],
        "policy": policy,
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
            "rule_auto_filter_enabled": op.empty_threshold is not None,
            "rule_auto_accept_enabled": op.species_threshold is not None,
            **enabled,
            "accept_species": list(op.accept_species) if op.accept_species is not None else None,
            "species_gate": op.species_gate,
            "deviation": deviation.model_dump(mode="json") if deviation else None,
            "targets": {
                "max_false_empty_rate": rule.operating_point.max_false_empty_rate,
                "min_accepted_precision": rule.operating_point.min_accepted_precision,
                "interval": rule.operating_point.interval,
            },
            "metrics": _jsonable(op.metrics),
            "metrics_on_fit_partition": _jsonable(fit_side),
            "per_camera": per_camera,
        },
        "sweeps": {
            "empty_filter": [_jsonable(m) for m in op.empty_sweep],
            "species_accept": [_jsonable(m) for m in op.species_sweep],
        },
        "sweeps_fit_partition": sweeps_fit,
    }
    (report_dir / "metrics.json").write_text(json.dumps(_round(result), indent=2) + "\n")
    (report_dir / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    (rule.model / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    (rule.model / "calibration.json").write_text(json.dumps(calibration_artifact, indent=2) + "\n")
    from wildinbox.evaluation.calibration_report import write_report

    write_report(report_dir, result)
    return result


def _pv(cfg: Any) -> str:
    from wildinbox.policy.conservative import POLICY_NAME

    return f"{POLICY_NAME}+{cfg.fingerprint()}"
