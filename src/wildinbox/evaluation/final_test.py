"""`wildinbox final-test`: measure the frozen system once on the locked final test.

This module is the only code path that reads the final-test partition, and it
does so only after every artifact pinned by the protocol
(configs/experiments/final_test.yaml) matches its recorded hash. Nothing is
chosen here: thresholds, models, and rules are already frozen.

The first run records `reports/final_test/opened.json`. A later run is allowed
only with the same protocol and must reproduce the recorded results exactly;
otherwise it refuses to write.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict

from wildinbox.class_map import EMPTY_CLASS
from wildinbox.datasets.spec import Partition
from wildinbox.evaluation.data import EventRow, ImageRow, _jsonl, _opt, box_areas
from wildinbox.evaluation.metrics import ScoredEvent, event_metrics, image_metrics, wilson
from wildinbox.policy.conservative import Frame, PolicyConfig, Thresholds, decide

INF = math.inf


class FinalTestError(RuntimeError):
    pass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelPin(_Strict):
    dir: Path
    weights_sha256: str
    release_id: str


class PolicyPin(_Strict):
    path: Path
    sha256: str
    artifact_version: str
    calibration_version: str
    released_policy_version: str
    rule_policy_version: str


class BaselinePin(_Strict):
    dir: Path
    classifier_sha256: str


class UnfamiliarPin(_Strict):
    rule: Path
    rule_sha256: str
    artifact_version: str
    distance_threshold: float
    confidence_threshold: float


class Protocol(_Strict):
    split_version: str
    split_lock_sha256: str
    model: ModelPin
    policy_artifact: PolicyPin
    baseline: BaselinePin
    unfamiliar: UnfamiliarPin
    report: list[str]
    bootstrap_resamples: int
    bootstrap_seed: int


def load_protocol(path: Path) -> Protocol:
    return Protocol.model_validate(yaml.safe_load(path.read_text()))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(protocol: Protocol, lock_path: Path) -> dict[str, str]:
    """Every pinned artifact must match its hash before the test is opened."""
    checks = {
        "split lock": (lock_path, protocol.split_lock_sha256),
        "E3 weights": (protocol.model.dir / "model.pt", protocol.model.weights_sha256),
        "policy artifact": (protocol.policy_artifact.path, protocol.policy_artifact.sha256),
        "baseline classifier": (
            protocol.baseline.dir / "classifier.npz",
            protocol.baseline.classifier_sha256,
        ),
        "unfamiliar rule": (protocol.unfamiliar.rule, protocol.unfamiliar.rule_sha256),
    }
    bad = [name for name, (path, want) in checks.items() if _sha(path) != want]
    if bad:
        raise FinalTestError(f"frozen artifacts changed since the protocol was written: {bad}")
    policy = json.loads(protocol.policy_artifact.path.read_text())
    pa = protocol.policy_artifact
    pinned = (
        pa.artifact_version,
        pa.calibration_version,
        pa.released_policy_version,
        pa.rule_policy_version,
    )
    found = (
        policy["artifact_version"],
        policy["calibration"]["version"],
        policy["released"]["policy_version"],
        policy["rule"]["policy_version"],
    )
    if pinned != found:
        raise FinalTestError(f"policy artifact versions {found} differ from the protocol {pinned}")
    if json.loads(lock_path.read_text()).get("split_version") != protocol.split_version:
        raise FinalTestError("split version differs from the protocol")
    return {name: want for name, (_, want) in checks.items()}


def open_final_test(
    split_dir: Path, boxes: dict[str, float | None]
) -> tuple[list[ImageRow], dict[str, EventRow]]:
    """The locked partition. Call only after `verify` has passed."""
    ft = Partition.FINAL_TEST.value
    images = [
        ImageRow(
            source_id=str(r["source_id"]),
            event_id=str(r["event_id"]),
            partition=Partition.FINAL_TEST,
            camera_id=str(r["camera_id"]),
            storage_path=str(r["storage_path"]),
            image_label=_opt(r["image_label"]),
            event_label=_opt(r["event_label"]),
            event_role=str(r["event_role"]),
            use_for_fit=bool(r["use_for_fit"]),
            max_box_area=boxes.get(str(r["source_id"])),
        )
        for r in _jsonl(split_dir / "images.jsonl.gz")
        if r["partition"] == ft
    ]
    events = {
        str(r["event_id"]): EventRow(
            event_id=str(r["event_id"]),
            partition=Partition.FINAL_TEST,
            camera_id=str(r["camera_id"]),
            role=str(r["role"]),
            label=_opt(r["label"]),
            animal_present=bool(r["animal_present"]),
            image_ids=tuple(r["image_ids"]),
        )
        for r in _jsonl(split_dir / "events.jsonl.gz")
        if r["partition"] == ft
    }
    if any(r.use_for_fit for r in images):
        raise FinalTestError("final-test images are marked as fit examples")
    return images, events


# ------------------------------------------------------------------ metrics


def bootstrap_macro_f1(
    y_true: list[str], y_pred: list[str], groups: list[str], classes: list[str], n: int, seed: int
) -> tuple[float, float]:
    """95% interval for macro-F1, resampling whole capture events."""
    idx = {c: i for i, c in enumerate(classes)}
    k = len(classes)
    group_ids = {g: i for i, g in enumerate(dict.fromkeys(groups))}
    per_group = np.zeros((len(group_ids), k * k))
    for t, p, g in zip(y_true, y_pred, groups, strict=True):
        per_group[group_ids[g], idx[t] * k + idx[p]] += 1
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        w = rng.multinomial(len(group_ids), np.full(len(group_ids), 1 / len(group_ids)))
        cm = (w @ per_group).reshape(k, k)
        tp = np.diag(cm)
        support, predicted = cm.sum(axis=1), cm.sum(axis=0)
        present = support > 0
        prec = np.divide(tp, predicted, out=np.zeros(k), where=predicted > 0)
        rec = np.divide(tp, support, out=np.zeros(k), where=support > 0)
        f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=(prec + rec) > 0)
        out.append(f1[present].mean())
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def _auroc(known: np.ndarray, unknown: np.ndarray) -> float | None:
    from wildinbox.evaluation.unfamiliar import auroc

    return auroc(known, unknown) if len(known) and len(unknown) else None


def _rate(flags: np.ndarray) -> dict[str, Any]:
    k, n = int(flags.sum()), len(flags)
    return {"flagged": k, "n": n, "rate": k / n if n else None, "ci95": wilson(k, n)}


def _policy_config(d: dict[str, Any]) -> PolicyConfig:
    species = d.get("accept_species")
    return PolicyConfig(
        d["empty_threshold"],
        d["species_threshold"],
        d["auto_filter_enabled"],
        d["auto_accept_enabled"],
        tuple(species) if species is not None else None,
    )


def _grid(spec: dict[str, Any]) -> list[float]:
    n = round((spec["stop"] - spec["start"]) / spec["step"])
    return sorted(
        {round(spec["start"] + i * spec["step"], 6) for i in range(n + 1)} | set(spec["extra"])
    )


def run(protocol_path: Path, config_path: Path, report_dir: Path) -> dict[str, Any]:
    from wildinbox.evaluation.predictors import FinetunedPredictor, predictor_for
    from wildinbox.evaluation.run import _events, _round, _score
    from wildinbox.inference.calibration import apply_temperature
    from wildinbox.inference.unfamiliar import knn_cosine_distance, normalize
    from wildinbox.settings import Settings
    from wildinbox.training.run import git_state, load_context

    protocol = load_protocol(protocol_path)
    protocol_sha = _sha(protocol_path)
    opened_path = report_dir / "opened.json"
    if opened_path.exists():
        opened = json.loads(opened_path.read_text())
        if opened["protocol_sha256"] != protocol_sha:
            raise FinalTestError("the final test was opened under a different protocol")
    ctx = load_context(config_path, Settings().data_dir)
    lock_path = Path("manifests") / "cct20-splits-v1.lock.json"
    artifacts = verify(protocol, lock_path)
    code = git_state()
    policy = json.loads(protocol.policy_artifact.path.read_text())
    temperature = float(policy["calibration"]["temperature"])
    classes = list(policy["classes"])
    unf_rule = yaml.safe_load(protocol.unfamiliar.rule.read_text())
    tuning, held_out = set(unf_rule["species"]["tuning"]), set(unf_rule["species"]["held_out"])

    rows, events = open_final_test(ctx.split_dir, box_areas(ctx.inventory_db))
    e3 = FinetunedPredictor(ctx, protocol.model.dir, "mps")
    if e3.weights_digest != protocol.model.weights_sha256[:12]:
        raise FinalTestError("loaded E3 weights differ from the protocol")
    scored = _score(e3, rows)
    raw = np.array([[s.probs[c] for c in classes] for s in scored])
    cal = apply_temperature(raw, temperature)
    cal_scored = [
        replace(s, probs={c: float(v) for c, v in zip(classes, p, strict=True)})
        for s, p in zip(scored, cal, strict=True)
    ]
    baseline_meta = json.loads((protocol.baseline.dir / "meta.json").read_text())
    base = predictor_for(ctx, protocol.baseline.dir, baseline_meta.get("device", "mps"))
    base_scored = _score(base, rows)

    def image_block(items: list[Any]) -> dict[str, Any]:
        sup = [s for s in items if s.supported]
        m = image_metrics([s.row.image_label or "" for s in sup], [s.pred for s in sup], classes)
        m["macro_f1_ci95"] = bootstrap_macro_f1(
            [s.row.image_label or "" for s in sup],
            [s.pred for s in sup],
            [s.row.event_id for s in sup],
            classes,
            protocol.bootstrap_resamples,
            protocol.bootstrap_seed,
        )
        return m

    images = {"e3": image_block(scored), "baseline": image_block(base_scored)}

    # Event decisions with calibrated scores (unfamiliar score not adopted: no flags).
    ev = _events(cal_scored, events)
    released_cfg, rule_cfg = _policy_config(policy["released"]), _policy_config(policy["rule"])
    dispositions: dict[str, Counter[str]] = {"released": Counter(), "rule": Counter()}
    reasons: Counter[str] = Counter()
    for e in ev:
        frames = [Frame(p) for p in e.frames]
        o_rel = decide(frames, released_cfg)
        dispositions["released"][o_rel.disposition.value] += 1
        dispositions["rule"][decide(frames, rule_cfg).disposition.value] += 1
        reasons.update(r.value for r in o_rel.reasons)
    rule_t = Thresholds(rule_cfg.empty_threshold or INF, rule_cfg.species_threshold or INF)
    at_rule = event_metrics(ev, rule_t, rule_cfg.accept_species)[0]
    empty_grid = _grid({"start": 0.5, "stop": 0.99, "step": 0.01, "extra": [0.995, 0.999]})
    sweeps = {
        "empty_filter": [event_metrics(ev, Thresholds(t, INF))[0] for t in empty_grid],
        "species_accept": [event_metrics(ev, Thresholds(rule_t.empty, t))[0] for t in empty_grid],
    }

    # Unsupported inputs, by species group.
    by_species: dict[str, list[ScoredEvent]] = defaultdict(list)
    for e in ev:
        if e.role == "unsupported_animal":
            by_species[e.label or "?"].append(e)
    unsupported = {}
    for sp, evs in sorted(by_species.items()):
        imgs = [
            s
            for s in scored
            if s.row.event_role == "unsupported_animal" and s.row.image_label == sp
        ]
        accepted_at = {
            f"{t:g}": event_metrics(evs, Thresholds(rule_t.empty, t))[0]["accepted"]
            for t in (0.5, 0.7, 0.9)
        }
        m_rule = event_metrics(evs, rule_t, rule_cfg.accept_species)[0]
        unsupported[sp] = {
            "group": "tuning" if sp in tuning else ("held_out" if sp in held_out else "other"),
            "events": len(evs),
            "images": len(imgs),
            "predicted": dict(Counter(s.pred for s in imgs).most_common()),
            "filtered_as_empty_at_rule": m_rule["false_empty"],
            "accepted_as_known_released": 0,
            "accepted_as_known_if_species_threshold": accepted_at,
        }

    # Unfamiliar-input detection at the frozen thresholds (image level).
    with np.load(protocol.model.dir / "unfamiliar-reference.npz") as z:
        reference = z["features"]
    feats = normalize(e3.features("final_test", rows).embeddings)
    dist = knn_cosine_distance(feats, reference, int(unf_rule["score"]["k"]))
    conf = 1.0 - cal.max(axis=1)
    known = np.array([s.supported and s.row.image_label != EMPTY_CLASS for s in scored])
    groups = {
        "tuning": np.array(
            [
                s.row.event_role == "unsupported_animal" and s.row.image_label in tuning
                for s in scored
            ]
        ),
        "held_out": np.array(
            [
                s.row.event_role == "unsupported_animal" and s.row.image_label in held_out
                for s in scored
            ]
        ),
    }
    detection = {}
    for name, score, thr in (
        ("distance", dist, protocol.unfamiliar.distance_threshold),
        ("confidence", conf, protocol.unfamiliar.confidence_threshold),
    ):
        detection[name] = {
            "threshold": thr,
            "false_flags_on_known": _rate(score[known] > thr),
            **{
                g: {
                    "detection": _rate(score[mask] > thr),
                    "auroc": _auroc(score[known], score[mask]),
                }
                for g, mask in groups.items()
            },
        }

    # Per camera and day/night at the rule's operating point.
    def slice_block(items: list[Any], evs: list[ScoredEvent]) -> dict[str, Any]:
        sup = [s for s in items if s.supported]
        im = image_metrics([s.row.image_label or "" for s in sup], [s.pred for s in sup], classes)
        em = event_metrics(evs, rule_t, rule_cfg.accept_species)[0]
        return {
            "images": len(sup),
            "macro_f1": im["macro_f1"],
            "animal_image_recall": im["animal_image_recall"],
            "events": em["events"],
            "animal_events": em["animal_events"],
            "false_empty_at_rule": em["false_empty"],
            "false_empty_rate_at_rule": em["false_empty_rate"],
            "filtered_at_rule": em["filtered"],
        }

    ev_by_id = {e.event_id: e for e in ev}
    per_camera = {}
    for cam in sorted({s.row.camera_id for s in cal_scored}, key=lambda c: (len(c), c)):
        items = [s for s in cal_scored if s.row.camera_id == cam]
        ids = {s.row.event_id for s in items}
        per_camera[cam] = slice_block(items, [ev_by_id[i] for i in ids if i in ev_by_id])
    flags: dict[str, list[bool]] = defaultdict(list)
    for s in cal_scored:
        flags[s.row.event_id].append(s.night)
    night_of = {eid: sum(f) * 2 >= len(f) for eid, f in flags.items()}
    day_night = {}
    for cond in ("day", "night"):
        items = [s for s in cal_scored if night_of[s.row.event_id] == (cond == "night")]
        ids = {s.row.event_id for s in items}
        day_night[cond] = slice_block(items, [ev_by_id[i] for i in ids if i in ev_by_id])
    small = [
        s
        for s in scored
        if s.supported and s.row.image_label != EMPTY_CLASS and (s.row.max_box_area or 1) < 0.01
    ]
    small_recall = sum(s.pred == s.row.image_label for s in small) / len(small) if small else None

    cams_f1 = [c["macro_f1"] for c in per_camera.values()]
    cams_fe = [c["false_empty_rate_at_rule"] for c in per_camera.values() if c["animal_events"]]

    # Random-image vs unseen-camera: development reports of the same models.
    dev_e3 = json.loads(
        Path("reports/experiments/finetune-e3-deep-balanced/metrics.json").read_text()
    )
    dev_base = json.loads(Path("reports/baseline/metrics.json").read_text())

    def dev(m: dict[str, Any], group: str) -> float:
        return float(m["groups"][group]["image"]["macro_f1"])

    comparison = {
        "e3": {
            "seen_cameras_random_sequences": dev(dev_e3, "seen_camera_diagnostic"),
            "unseen_development_cameras": dev(dev_e3, "unseen_cameras"),
            "final_test_unseen_cameras": images["e3"]["macro_f1"],
        },
        "baseline": {
            "seen_cameras_random_sequences": dev(dev_base, "seen_camera_diagnostic"),
            "unseen_development_cameras": dev(dev_base, "unseen_cameras"),
            "final_test_unseen_cameras": images["baseline"]["macro_f1"],
        },
    }

    n_images = len({s.row.source_id for s in scored})
    n_events = len(ev)
    released_review = dispositions["released"]["needs_review"]
    results = {
        "split_version": protocol.split_version,
        "cameras": sorted(per_camera, key=lambda c: (len(c), c)),
        "images": n_images,
        "events": n_events,
        "event_roles": dict(Counter(e.role for e in ev).most_common()),
        "image_level": images,
        "decisions": {
            "released_policy": policy["released"]["policy_version"],
            "rule_policy": policy["rule"]["policy_version"],
            "dispositions": {k: dict(sorted(v.items())) for k, v in dispositions.items()},
            "released_reasons": dict(reasons.most_common()),
            "rule_operating_point": at_rule,
        },
        "sweeps": sweeps,
        "unsupported": unsupported,
        "unfamiliar_detection": detection,
        "review_reduction": {
            "images": n_images,
            "events_after_grouping": n_events,
            "grouping_factor": n_images / n_events,
            "released_events_to_review": released_review,
            "released_reduction_vs_grouped": 1 - released_review / n_events,
            "rule_events_to_review": at_rule["needs_review"],
            "rule_reduction_vs_grouped": 1 - at_rule["needs_review"] / n_events,
        },
        "per_camera": per_camera,
        "camera_spread": {
            "macro_f1": {
                "min": min(cams_f1),
                "median": float(np.median(cams_f1)),
                "max": max(cams_f1),
            },
            "false_empty_rate_at_rule": {
                "min": min(cams_fe),
                "median": float(np.median(cams_fe)),
                "max": max(cams_fe),
            },
        },
        "day_night": day_night,
        "small_animals": {"images": len(small), "recall": small_recall},
        "random_vs_unseen": comparison,
    }
    results = _round(results)

    report_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = report_dir / "metrics.json"
    if opened_path.exists() and metrics_path.exists():
        recorded = json.loads(metrics_path.read_text())["results"]
        if recorded != json.loads(json.dumps(results)):
            raise FinalTestError("re-run does not reproduce the recorded final-test results")
    else:
        opened_path.write_text(
            json.dumps(
                {
                    "protocol": str(protocol_path),
                    "protocol_sha256": protocol_sha,
                    "artifacts": artifacts,
                    "code": code,
                    "opened_at": datetime.now(UTC).isoformat(),
                },
                indent=2,
            )
            + "\n"
        )
    out = {
        "protocol": str(protocol_path),
        "protocol_sha256": protocol_sha,
        "release_id": protocol.model.release_id,
        "code": code,
        "results": results,
    }
    metrics_path.write_text(json.dumps(out, indent=2) + "\n")
    with gzip.open(report_dir / "decisions.jsonl.gz", "wt") as f:
        for e in ev:
            frames = [Frame(p) for p in e.frames]
            f.write(
                json.dumps(
                    {
                        "event_id": e.event_id,
                        "camera_id": events[e.event_id].camera_id,
                        "role": e.role,
                        "label": e.label,
                        "released": decide(frames, released_cfg).disposition.value,
                        "rule": decide(frames, rule_cfg).disposition.value,
                    }
                )
                + "\n"
            )
    from wildinbox.evaluation.final_test_report import write_report

    write_report(report_dir, out, policy)
    return out
