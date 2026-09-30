"""`wildinbox adaptation fresh-test`: measure the frozen v2 operating point once
on the locked fresh-test cameras (configs/experiments/fresh_test.yaml).

This module is the only code path that reads the fresh-test partition, and it
does so only after the protocol is committed and unchanged and every artifact
it pins matches its hash. Nothing is chosen here on the fresh cameras: the
method, N, configuration, and species threshold are frozen, and the
comparisons' thresholds come from the development cameras.

`--dev-check` runs the same procedure on the adaptation-development cameras
only and must reproduce the recorded development result
(reports/adaptation/development/metrics.json) and the rule's choice of the
frozen threshold. It never opens the fresh test.

The first fresh run records `opened.json` and `metrics.json`. A later run is
allowed only with the same protocol, never rewrites them, and must reproduce
the recorded results exactly (on another device, within
CROSS_DEVICE_TOLERANCE).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict

from wildinbox.adaptation.data import (
    CameraEvent,
    FrameOutputs,
    camera_events_from,
    load_camera_events,
    score_frames,
)
from wildinbox.adaptation.evaluate import (
    AUDIT_RATE,
    Scored,
    add,
    choose,
    outcome,
    score_camera,
    summarize,
)
from wildinbox.adaptation.methods import CameraHead, Method, Release
from wildinbox.datasets.spec import Partition
from wildinbox.evaluation.data import ImageRow, _jsonl, _opt
from wildinbox.evaluation.final_test import CROSS_DEVICE_TOLERANCE, compare_results
from wildinbox.evaluation.metrics import wilson

DEV_SPLIT = "cct-fresh-dev-v1"
DEV_IMAGES = "cct_fresh_dev"
DEV_METRICS = Path("reports/adaptation/development/metrics.json")


class FreshTestError(RuntimeError):
    pass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Expected(_Strict):
    sequences: int
    images: int


class CamerasPin(_Strict):
    rule: Path
    rule_sha256: str
    manifest: Path
    manifest_sha256: str
    locations: list[str]
    expected: Expected


class ModelPin(_Strict):
    dir: Path
    weights_sha256: str
    policy_sha256: str


class Thresholds(_Strict):
    empty: float | None
    species: float


class MethodSpec(_Strict):
    name: str
    base: str
    c: float
    camera_share: float
    n_reviewed: int
    reviews: str
    other_animal_accepted: bool
    thresholds: Thresholds


class Bootstrap(_Strict):
    unit: str
    resamples: int
    seed: int


class PassIf(_Strict):
    min_accepted: int
    precision_point: float
    precision_wilson_lower: float


class Target(_Strict):
    claim: str
    pass_if: PassIf
    also_reported_not_gating: dict[str, str]
    expectation: str


class Protocol(_Strict):
    name: str
    cameras: CamerasPin
    model: ModelPin
    method: MethodSpec
    report: list[str]
    bootstrap: Bootstrap
    target: Target


def load_protocol(path: Path) -> Protocol:
    return Protocol.model_validate(yaml.safe_load(path.read_text()))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _committed_and_clean(path: Path) -> bool:
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path)], capture_output=True
        )
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", str(path)])
    except OSError:
        return False
    return tracked.returncode == 0 and not dirty.strip()


def verify(protocol: Protocol) -> dict[str, str]:
    """Every pinned artifact must match its hash; the method must be the one
    this module implements."""
    checks = {
        "camera rule": (protocol.cameras.rule, protocol.cameras.rule_sha256),
        "camera manifest": (protocol.cameras.manifest, protocol.cameras.manifest_sha256),
        "model weights": (protocol.model.dir / "model.pt", protocol.model.weights_sha256),
        "policy artifact": (protocol.model.dir / "policy.json", protocol.model.policy_sha256),
    }
    bad = [name for name, (path, want) in checks.items() if _sha(path) != want]
    if bad:
        raise FreshTestError(f"pinned artifacts changed since the protocol was written: {bad}")
    manifest = json.loads(protocol.cameras.manifest.read_text())
    listed = [c["location"] for c in manifest["fresh_test"]]
    if listed != protocol.cameras.locations:
        raise FreshTestError(f"manifest fresh-test cameras {listed} differ from the protocol")
    m = protocol.method
    if (
        m.name != "camera_head_other"
        or m.other_animal_accepted
        or m.thresholds.empty is not None
        or protocol.bootstrap.unit != "camera"
    ):
        raise FreshTestError("this command implements only the frozen species-only head")
    return {name: want for name, (_, want) in checks.items()}


def open_fresh_test(split_dir: Path) -> tuple[dict[str, list[CameraEvent]], list[ImageRow]]:
    """The locked partition. Call only after `verify` has passed."""
    ft = Partition.FRESH_TEST.value
    rows = [
        ImageRow(
            source_id=str(r["source_id"]),
            event_id=str(r["event_id"]),
            partition=Partition.FRESH_TEST,
            camera_id=str(r["camera_id"]),
            storage_path=str(r["storage_path"]),
            image_label=_opt(r["image_label"]),
            event_label=_opt(r["event_label"]),
            event_role=str(r["event_role"]),
            use_for_fit=bool(r["use_for_fit"]),
        )
        for r in _jsonl(split_dir / "images.jsonl.gz")
        if r["partition"] == ft
    ]
    if any(r.use_for_fit for r in rows):
        raise FreshTestError("fresh-test images are marked as fit examples")
    return camera_events_from(split_dir, {ft}, rows), rows


# ------------------------------------------------------------------ measures


def pooled(
    per_camera: dict[str, tuple[list[Scored], int]], t_species: float | None
) -> dict[str, Any]:
    """Species labels only (no empty filtering) at a fixed threshold, per camera
    and pooled from counts."""
    cams: dict[str, Any] = {}
    total: dict[str, Any] = {}
    for cam, (scored, n_events) in per_camera.items():
        o = outcome(scored, n_events, None, t_species)
        cams[cam] = summarize(o)
        total = add(total, o)
    return {"t_species": t_species, "pooled": summarize(total), "cameras": cams}


def camera_bootstrap(
    per_camera_outcomes: dict[str, dict[str, Any]], resamples: int, seed: int
) -> dict[str, Any]:
    """Percentile intervals from resampling cameras with replacement. A
    resample that accepts nothing has no precision and is left out of that
    interval (counted)."""
    rng = np.random.default_rng(seed)
    cams = sorted(per_camera_outcomes)
    keys = ("automated", "total_events", "accepted", "accepted_correct")
    counts = np.array([[per_camera_outcomes[c][k] for k in keys] for c in cams], dtype=float)
    precision, reduction = [], []
    for _ in range(resamples):
        s = counts[rng.integers(0, len(cams), len(cams))].sum(axis=0)
        reduction.append(s[0] * (1 - AUDIT_RATE) / s[1])
        if s[2]:
            precision.append(s[3] / s[2])

    def interval(xs: list[float]) -> list[float] | None:
        return [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))] if xs else None

    return {
        "resamples": resamples,
        "seed": seed,
        "precision": interval(precision),
        "precision_undefined_resamples": resamples - len(precision),
        "review_reduction": interval(reduction),
    }


def accepted_errors(
    per_camera: dict[str, tuple[list[Scored], int]], t_species: float
) -> dict[str, int]:
    """Wrongly accepted labels by what the event really was."""
    c: Counter[str] = Counter()
    for scored, _ in per_camera.values():
        for s in scored:
            if s.accepted(float("inf"), t_species) and not s.correct:
                if s.role == "supported_species":
                    c[f"{s.label} called {s.suggestion}"] += 1
                else:
                    c[f"{s.role} called {s.suggestion}"] += 1
    return dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))


def judge(pooled_result: dict[str, Any], rule: PassIf) -> str:
    """pass / fail / inconclusive under the protocol's target."""
    acc = pooled_result["accepted"]
    if acc < rule.min_accepted:
        return "inconclusive"
    ok = pooled_result["accepted_correct"]
    lower = wilson(ok, acc)[0]
    return (
        "pass"
        if ok / acc >= rule.precision_point and lower >= rule.precision_wilson_lower
        else "fail"
    )


# ------------------------------------------------------------------ run


def _methods(
    protocol: Protocol, temperature: float, x_base: np.ndarray, y_base: tuple[str, ...]
) -> dict[str, Method]:
    m = protocol.method
    return {
        "adapted": CameraHead(
            x_base, y_base, with_other=True, camera_share=m.camera_share, c=m.c, name=m.name
        ),
        "release": Release(temperature),
        # the same head without the camera's reviews (ablation)
        "no_camera_reviews": CameraHead(
            x_base, y_base, with_other=True, camera_share=0.0, c=m.c, name="unseen_head"
        ),
    }


def _score_all(
    methods: dict[str, Method], cams: dict[str, list[CameraEvent]], n: int, out: FrameOutputs
) -> dict[str, dict[str, tuple[list[Scored], int]]]:
    return {
        name: {cam: score_camera(m, ev, n, out) for cam, ev in sorted(cams.items())}
        for name, m in methods.items()
    }


def _species_rule(per_camera: dict[str, tuple[list[Scored], int]]) -> float | None:
    """The release rule's species threshold on all these cameras at once."""
    return choose([s for ss, _ in per_camera.values() for s in ss], filter_empty=False)[1]


def dev_check(
    protocol: Protocol, dev: dict[str, dict[str, tuple[list[Scored], int]]]
) -> dict[str, Any]:
    """The frozen procedure on the development cameras must reproduce the
    recorded development result, and the rule must choose the frozen threshold."""
    m = protocol.method
    problems = []
    chosen = _species_rule(dev["adapted"])
    if chosen != m.thresholds.species:
        problems.append(
            f"the rule chooses {chosen} on development, protocol has {m.thresholds.species}"
        )
    got = pooled(dev["adapted"], m.thresholds.species)["pooled"]
    recorded = json.loads(DEV_METRICS.read_text())["results"][
        f"camera_head_other_species_only@{m.n_reviewed}"
    ]["per_config_in_sample"][f"C={m.c},share={m.camera_share}"]
    for k in (
        "total_events",
        "later_events",
        "accepted",
        "accepted_correct",
        "filtered",
        "automated",
    ):
        if got[k] != recorded[k]:
            problems.append(f"{k}: {got[k]} vs recorded {recorded[k]}")
    if problems:
        raise FreshTestError("development check failed: " + "; ".join(problems))
    return {"threshold_chosen_by_rule": chosen, "in_sample": got}


def run(
    protocol_path: Path,
    config_path: Path,
    data_dir: Path,
    split_name: str,
    images_name: str,
    report_dir: Path,
    *,
    device: str,
    dev_only: bool,
) -> dict[str, Any]:
    from wildinbox.adaptation.run import unseen_base
    from wildinbox.training.run import git_state

    protocol = load_protocol(protocol_path)
    artifacts = verify(protocol)
    m = protocol.method
    policy = json.loads((protocol.model.dir / "policy.json").read_text())
    temperature = float(policy["calibration"]["temperature"])

    x_base, y_base = unseen_base(config_path, protocol.model.dir, data_dir, device)
    methods = _methods(protocol, temperature, x_base, y_base)

    dev_split = data_dir / "splits" / DEV_SPLIT
    dev_cams, dev_rows = load_camera_events(dev_split, [Partition.ADAPTATION_DEVELOPMENT])
    dev_out = score_frames(
        config_path,
        protocol.model.dir,
        dev_split,
        data_dir / "raw" / DEV_IMAGES / "images",
        dev_rows,
        "fresh-dev-v1",
        device,
    )
    dev = _score_all(methods, dev_cams, m.n_reviewed, dev_out)
    check = dev_check(protocol, dev)
    if dev_only:
        return {"dev_check": check}

    # ---- the locked fresh test: only past this point
    protocol_sha = _sha(protocol_path)
    if not _committed_and_clean(protocol_path):
        raise FreshTestError("the fresh test is opened only under a committed, unchanged protocol")
    code = git_state()
    if code["dirty"]:
        raise FreshTestError("uncommitted changes under src/ or configs/: commit before opening")
    opened_path = report_dir / "opened.json"
    metrics_path = report_dir / "metrics.json"
    if (
        opened_path.exists()
        and json.loads(opened_path.read_text())["protocol_sha256"] != protocol_sha
    ):
        raise FreshTestError("the fresh test was opened under a different protocol")

    split_dir = data_dir / "splits" / split_name
    cams, rows = open_fresh_test(split_dir)
    if sorted(cams) != sorted(protocol.cameras.locations):
        raise FreshTestError(f"fresh-test cameras {sorted(cams)} differ from the protocol")
    short = [c for c, ev in cams.items() if len(ev) <= m.n_reviewed]
    if short:
        raise FreshTestError(f"cameras with no events after the reviewed ones: {short}")
    out = score_frames(
        config_path,
        protocol.model.dir,
        split_dir,
        data_dir / "raw" / images_name / "images",
        rows,
        "fresh-test-v1",
        device,
    )
    if not opened_path.exists():
        report_dir.mkdir(parents=True, exist_ok=True)
        opened_path.write_text(
            json.dumps(
                {
                    "opened_at": datetime.now(UTC).isoformat(),
                    "protocol": str(protocol_path),
                    "protocol_sha256": protocol_sha,
                    "code": code,
                    "device": device,
                },
                indent=2,
            )
            + "\n"
        )
    fresh = _score_all(methods, cams, m.n_reviewed, out)

    t = m.thresholds.species
    main = pooled(fresh["adapted"], t)
    comparisons = {
        name: {
            "threshold_by_rule_on_development": pooled(fresh[name], _species_rule(dev[name])),
            "at_frozen_threshold": pooled(fresh[name], t),
        }
        for name in ("release", "no_camera_reviews")
    }
    dev_loco = json.loads(DEV_METRICS.read_text())["results"][
        f"camera_head_other_species_only@{m.n_reviewed}"
    ]["leave_one_camera_out"]["pooled"]
    results = {
        "data": {
            "cameras": {c: len(ev) for c, ev in sorted(cams.items())},
            "events": sum(len(ev) for ev in cams.values()),
            "images": len(rows),
            "expected": protocol.cameras.expected.model_dump(),
            "roles": dict(sorted(Counter(e.role for ev in cams.values() for e in ev).items())),
        },
        "adapted": main,
        "bootstrap": camera_bootstrap(
            main["cameras"], protocol.bootstrap.resamples, protocol.bootstrap.seed
        ),
        "accepted_errors": accepted_errors(fresh["adapted"], t),
        "comparisons": comparisons,
        "development_estimate": dev_loco,
        "target": {
            "pass_if": protocol.target.pass_if.model_dump(),
            "outcome": judge(main["pooled"], protocol.target.pass_if),
        },
        "original_target": {
            "review_reduction": 0.5,
            "met": False,
            "note": "not met by v1 and not addressed by this operating point",
        },
    }
    results = json.loads(json.dumps(results, default=float))  # the form it is stored in

    if metrics_path.exists():
        record: dict[str, Any] = json.loads(metrics_path.read_text())
        tolerance = 0.0 if device == record["device"] else CROSS_DEVICE_TOLERANCE
        problems = compare_results(record["results"], results, tolerance)
        if problems:
            raise FreshTestError(
                "rerun differs from the recorded fresh test: " + "; ".join(problems[:10])
            )
        return record
    payload = {
        "protocol": str(protocol_path),
        "protocol_sha256": protocol_sha,
        "artifacts": artifacts,
        "code": code,
        "device": device,
        "dev_check": check,
        "results": results,
    }
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload
