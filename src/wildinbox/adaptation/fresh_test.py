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

The fresh split is bound to its committed lock (manifests/<split>.lock.json)
and the ingest lock it was built from (manifests/<images>.lock.json): before
any fresh data is read, their totals must reconcile with the protocol's
expected counts, the only allowed difference being files the ingest declared
rejected. Once read, the split files must reproduce the lock's content digest
and every locked event and image must be loaded or excluded with a recorded
reason.

`opened.json` is created, once and atomically, before the first fresh data is
read, so a failure during inference leaves the opening recorded. A later run is
allowed only with the same protocol and inputs, never rewrites `opened.json`
or `metrics.json`, and must reproduce the recorded results exactly (on another
device, within CROSS_DEVICE_TOLERANCE).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
from collections import Counter, defaultdict
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
MANIFESTS = Path("manifests")


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


# ------------------------------------------------------------------ inputs


def split_inputs(protocol: Protocol, split_name: str, images_name: str) -> dict[str, Any]:
    """The finalized fresh split's identity, from committed locks only (no
    fresh data is read). The split lock must be this partition with the
    protocol's cameras, built from the pinned ingest, with its leakage checks
    passed; its totals must reconcile with the protocol's expected counts,
    where only files the ingest declared rejected may be missing."""
    split_lock = MANIFESTS / f"{split_name}.lock.json"
    ingest_lock = MANIFESTS / f"{images_name}.lock.json"
    for path in (split_lock, ingest_lock):
        if not path.exists():
            raise FreshTestError(f"{path} not found: the fresh split must be locked and ingested")
    lock = json.loads(split_lock.read_text())
    ingest = json.loads(ingest_lock.read_text())
    counts, ingest_version = ingest["counts"], ingest["manifest_version"]
    events = sum(c["events"] for c in lock["cameras"].values())
    images = sum(c["images"] for c in lock["cameras"].values())
    rejected = counts["by_status"]["quarantined"] + counts["by_status"]["excluded"]
    expected = protocol.cameras.expected
    problems = []
    if lock["partition"] != Partition.FRESH_TEST.value:
        problems.append(f"split lock is partition {lock['partition']!r}")
    if not lock["split_version"].startswith(f"{split_name}-"):
        problems.append(f"split lock version {lock['split_version']} is not split {split_name}")
    if sorted(lock["cameras"]) != sorted(protocol.cameras.locations):
        problems.append(f"split lock cameras {sorted(lock['cameras'])} differ from the protocol")
    failed = sorted(k for k, ok in lock["checks"].items() if not ok)
    if failed:
        problems.append(f"split lock checks failed: {failed}")
    if lock["inventory_manifest_version"] != ingest_version:
        problems.append(
            f"split built from {lock['inventory_manifest_version']}; ingest is {ingest_version}"
        )
    if counts["source_records"] != expected.images:
        problems.append(
            f"ingest saw {counts['source_records']} files, protocol expects {expected.images}"
        )
    if counts["by_status"]["accepted"] != images or images + rejected != expected.images:
        problems.append(
            f"split has {images} images; ingest accepted {counts['by_status']['accepted']} "
            f"and rejected {rejected} of the {expected.images} expected"
        )
    if events != expected.sequences:  # a rejected file excludes its event, never removes it
        problems.append(f"split has {events} events, protocol expects {expected.sequences}")
    if problems:
        raise FreshTestError("fresh split does not match its data contract: " + "; ".join(problems))
    return {
        "split_version": lock["split_version"],
        "split_lock": str(split_lock),
        "split_lock_sha256": _sha(split_lock),
        "ingest_lock": str(ingest_lock),
        "ingest_lock_sha256": _sha(ingest_lock),
        "events": events,
        "images": images,
        "ingest_rejected": rejected,
    }


def split_version(split_dir: Path) -> str:
    """The split's content digest, as the build computes it: every event row,
    then every image row, uncompressed."""
    digest = hashlib.sha256()
    for name in ("events", "images"):
        with gzip.open(split_dir / f"{name}.jsonl.gz", "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
    return f"{split_dir.name}-{digest.hexdigest()[:12]}"


def reconcile(
    split_dir: Path,
    inputs: dict[str, Any],
    cams: dict[str, list[CameraEvent]],
    rows: list[ImageRow],
) -> dict[str, int]:
    """Every locked event and image is loaded, or excluded by the build with a
    recorded reason; returns the exclusions."""
    lock = json.loads(Path(inputs["split_lock"]).read_text())["cameras"]
    ft = Partition.FRESH_TEST.value
    found: dict[str, Counter[str]] = defaultdict(Counter)
    for r in _jsonl(split_dir / "events.jsonl.gz"):
        c = found[str(r["camera_id"])]
        c["events"] += 1
        c["images"] += len(r["image_ids"])
        if r["excluded_reason"] or not r["image_ids"]:
            c["excluded_events"] += 1
            c["excluded_images"] += len(r["image_ids"])
        elif r["partition"] != ft:
            c["other_partition"] += 1
    loaded_images = Counter(r.camera_id for r in rows)
    problems = []
    for cam in sorted(set(found) | set(lock)):
        c, want = found[cam], lock.get(cam, {"events": 0, "images": 0})
        if (c["events"], c["images"]) != (want["events"], want["images"]):
            problems.append(
                f"camera {cam}: {c['events']} events/{c['images']} images, "
                f"locked {want['events']}/{want['images']}"
            )
        if c["other_partition"]:
            problems.append(f"camera {cam}: {c['other_partition']} events in another partition")
        if len(cams.get(cam, [])) != c["events"] - c["excluded_events"]:
            problems.append(f"camera {cam}: {len(cams.get(cam, []))} events loaded")
        if loaded_images[cam] != c["images"] - c["excluded_images"]:
            problems.append(f"camera {cam}: {loaded_images[cam]} images loaded")
    if problems:
        raise FreshTestError("loaded fresh test differs from its lock: " + "; ".join(problems))
    return {
        "excluded_events": sum(c["excluded_events"] for c in found.values()),
        "excluded_images": sum(c["excluded_images"] for c in found.values()),
    }


def _sha_at(commit: str, path: str) -> str | None:
    try:
        blob = subprocess.check_output(
            ["git", "show", f"{commit}:{path}"], stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return hashlib.sha256(blob).hexdigest()


def check_opened(opened: dict[str, Any], protocol_sha: str, inputs: dict[str, Any]) -> None:
    """A retry must use the protocol and inputs the test was opened with. The
    recorded v2 opening predates the inputs field; its inputs are the locks
    as committed at the code commit it records."""
    if opened["protocol_sha256"] != protocol_sha:
        raise FreshTestError("the fresh test was opened under a different protocol")
    if "inputs" in opened:
        if opened["inputs"] != inputs:
            raise FreshTestError("the fresh test was opened with different inputs")
        return
    for key in ("split_lock", "ingest_lock"):
        if _sha_at(opened["code"]["commit"], inputs[key]) != inputs[f"{key}_sha256"]:
            raise FreshTestError(f"{inputs[key]} changed since the fresh test was opened")


def create_once(path: Path, payload: dict[str, Any]) -> None:
    """Write `path` durably, all at once, and only if it does not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps(payload, indent=2) + "\n")
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(tmp, path)  # fails if path exists: never overwrites
    except FileExistsError:
        raise FreshTestError(f"{path} appeared while opening the fresh test") from None
    finally:
        tmp.unlink()


def open_locked(
    protocol: Protocol,
    protocol_path: Path,
    split_dir: Path,
    images_name: str,
    opened_path: Path,
    opening: dict[str, Any],
) -> tuple[dict[str, list[CameraEvent]], list[ImageRow], dict[str, Any]]:
    """Record the opening (or check it against the recorded one), then read
    the fresh split and verify it is the locked one."""
    inputs = split_inputs(protocol, split_dir.name, images_name)
    unlocked = [
        inputs[k]
        for k in ("split_lock", "ingest_lock")
        if not _committed_and_clean(Path(inputs[k]))
    ]
    if unlocked:
        raise FreshTestError(f"the fresh test is opened only with committed locks: {unlocked}")
    protocol_sha = _sha(protocol_path)
    if opened_path.exists():
        check_opened(json.loads(opened_path.read_text()), protocol_sha, inputs)
    else:
        create_once(
            opened_path,
            {
                "opened_at": datetime.now(UTC).isoformat(),
                "protocol": str(protocol_path),
                "protocol_sha256": protocol_sha,
                **opening,
                "inputs": inputs,
            },
        )

    # ---- fresh data is read only past this point
    actual = split_version(split_dir)
    if actual != inputs["split_version"]:
        raise FreshTestError(f"fresh split is {actual}, its lock is {inputs['split_version']}")
    cams, rows = open_fresh_test(split_dir)
    exclusions = reconcile(split_dir, inputs, cams, rows)
    if sorted(cams) != sorted(protocol.cameras.locations):
        raise FreshTestError(f"fresh-test cameras {sorted(cams)} differ from the protocol")
    return cams, rows, inputs | exclusions


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
    metrics_path = report_dir / "metrics.json"
    split_dir = data_dir / "splits" / split_name
    cams, rows, inputs = open_locked(
        protocol,
        protocol_path,
        split_dir,
        images_name,
        report_dir / "opened.json",
        {"code": code, "device": device},
    )
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
        "inputs": inputs,
        "code": code,
        "device": device,
        "dev_check": check,
        "results": results,
    }
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload
