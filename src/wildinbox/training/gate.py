"""`wildinbox update gate`: compare a candidate with the deployed model under
configs/experiments/update_cycle.yaml and decide whether it may be released.

- Calibration: the candidate's temperature is fit on the calibration partition
  (cameras it never trained on). Its policy artifact keeps the deployed policy
  settings (automation off); only the calibration changes.
- Holdout: reviewed deployment events after each camera's cutoff (never
  trained on). The metric is event-level macro-F1 of the policy's suggested
  label against the reviewed label.
- Regressions: image-level macro-F1 on the calibration cameras and on the
  seen-camera diagnostic partition, where neither model trained.

Development data only. The gate never reads the final test, and refuses to run
if the snapshot's training or holdout frames overlap the final test or the
regression-check partitions (that would make the checks meaningless). Every
run is appended to `comparisons.jsonl` next to the report, so repeated
comparisons against the same holdout are counted; a protocol can cap them
(`gate.max_comparisons_per_holdout`), after which a fresh holdout is needed.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from wildinbox.class_map import EMPTY_CLASS
from wildinbox.datasets.spec import Partition
from wildinbox.evaluation.data import ImageRow, load_rows
from wildinbox.evaluation.metrics import image_metrics, wilson
from wildinbox.inference.calibration import apply_temperature
from wildinbox.policy.conservative import POLICY_NAME, Frame, PolicyConfig, decide
from wildinbox.training.snapshot import (
    SnapshotError,
    load_summary,
    model_training_frames,
    snapshot_train_frames,
)


class GateError(RuntimeError):
    """The gate cannot run a comparison it could vouch for."""


def _version(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class DeployedRelease:
    """The exact artifacts behind a release id, verified against each other."""

    id: str
    model_dir: Path
    policy_path: Path
    policy: dict[str, Any]
    weights_sha256: str


def resolve_release(
    release_id: str, models_root: Path = Path("models"), policy_path: Path | None = None
) -> DeployedRelease:
    """Local artifacts of `release_id` (`<model name>@<policy artifact version>`):
    `models_root/<name>` and the policy artifact of that version (`policy_path`,
    else the model directory's `policy.json`). Refuses anything that would not
    register as exactly this release: a policy of another version or edited
    after it was versioned, or weights, classes, calibration, or preprocessing
    that disagree (the checks `wildinbox release register` applies)."""
    from wildinbox.inference.releases import ReleaseError, build_release

    name, sep, version = release_id.rpartition("@")
    if not sep or not name or not version:
        raise GateError(f"release id {release_id!r} is not <model>@<policy artifact version>")
    model_dir = models_root / name
    if not (model_dir / "meta.json").is_file() or not (model_dir / "model.pt").is_file():
        raise GateError(f"release {release_id}: no model at {model_dir}")
    path = policy_path or model_dir / "policy.json"
    if not path.is_file():
        raise GateError(f"release {release_id}: no policy artifact at {path}")
    policy = json.loads(path.read_text())
    if policy.get("artifact_version") != version:
        raise GateError(
            f"release {release_id}: {path} is policy version "
            f"{policy.get('artifact_version')!r}, not {version!r}"
        )
    if _version({k: v for k, v in policy.items() if k != "artifact_version"}) != version:
        raise GateError(f"release {release_id}: {path} was changed after it was versioned")
    try:
        row, _ = build_release(model_dir, path)
    except ReleaseError as e:
        raise GateError(f"release {release_id}: {e}") from e
    if row["id"] != release_id:
        raise GateError(f"release {release_id}: artifacts resolve to {row['id']}")
    return DeployedRelease(release_id, model_dir, path, policy, row["weights_sha256"])


GUARDED = (Partition.FINAL_TEST, Partition.CALIBRATION, Partition.SEEN_CAMERA_DIAGNOSTIC)


def partition_hashes(split_dir: Path, partitions: tuple[Partition, ...]) -> dict[str, str]:
    """sha256 -> partition, for the images of these partitions."""
    wanted = {p.value for p in partitions}
    out: dict[str, str] = {}
    with gzip.open(split_dir / "images.jsonl.gz", "rt") as fh:
        for line in fh:
            row = json.loads(line)
            if row["partition"] in wanted:
                out[row["sha256"]] = row["partition"]
    return out


def earlier_holdout_hashes(protocol: dict[str, Any], root: Path = Path(".")) -> set[str]:
    """Frame SHA-256s of the earlier snapshot holdouts the protocol protects."""
    out: set[str] = set()
    for path in (protocol.get("protected") or {}).get("snapshot_holdouts", []):
        for line in (root / path).read_text().splitlines():
            out.update(f["sha256"] for f in json.loads(line).get("frames", []))
    return out


def leakage(
    snapshot_dir: Path,
    guarded: dict[str, str],
    training: set[str] | frozenset[str] = frozenset(),
    earlier_holdouts: set[str] | frozenset[str] = frozenset(),
    fitted: dict[str, set[str]] | None = None,
) -> dict[str, int]:
    """Content-separation violations, counted by frame, checked from the snapshot
    files alone (not trusting the builder's exclusions):

    - `<partition>`: snapshot frames (train or holdout) of a guarded partition;
    - `train_frame_in_holdout`: training frames whose bytes are also in the holdout;
    - `event_in_train_and_holdout`: an event on both sides (counted per event);
    - `holdout_frame_in_training_partition`: holdout frames any model trained on;
    - `holdout_frame_in_<model>_training`: holdout frames a compared model was fit
      on through an update snapshot (`fitted`: model -> frames, e.g. the
      deployed release trained in an earlier cycle);
    - `earlier_snapshot_holdout`: snapshot frames of an earlier protected holdout.
    """
    found: dict[str, int] = {}

    def add(key: str) -> None:
        found[key] = found.get(key, 0) + 1

    train = [json.loads(x) for x in (snapshot_dir / "train.jsonl").read_text().splitlines()]
    hold = [json.loads(x) for x in (snapshot_dir / "holdout.jsonl").read_text().splitlines()]
    train_shas = [r["sha256"] for r in train]
    hold_shas = [f["sha256"] for e in hold for f in e["frames"]]
    for sha in train_shas + hold_shas:
        if sha in guarded:
            add(guarded[sha])
        if sha in earlier_holdouts:
            add("earlier_snapshot_holdout")
    in_holdout = set(hold_shas)
    for sha in train_shas:
        if sha in in_holdout:
            add("train_frame_in_holdout")
    for sha in hold_shas:
        if sha in training:
            add("holdout_frame_in_training_partition")
        for model, frames in (fitted or {}).items():
            if sha in frames:
                add(f"holdout_frame_in_{model}_training")
    for _ in {r["event_id"] for r in train} & {e["event_id"] for e in hold}:
        add("event_in_train_and_holdout")
    return found


def log_comparison(log: Path, entry: dict[str, Any]) -> int:
    """Append a comparison; return how many were made on this holdout, this one included."""
    previous = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
    n: int = 1 + sum(p["holdout_version"] == entry["holdout_version"] for p in previous)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as fh:
        fh.write(json.dumps({**entry, "number_on_this_holdout": n}, sort_keys=True) + "\n")
    return n


def holdout_rows(snapshot_dir: Path) -> tuple[list[dict[str, Any]], list[ImageRow]]:
    events = [
        json.loads(line) for line in (snapshot_dir / "holdout.jsonl").read_text().splitlines()
    ]
    rows = [
        ImageRow(
            source_id=f"snapshot:{f['sha256']}",
            event_id=e["event_id"],
            partition=Partition.POLICY_VALIDATION,  # origin: the deployment cameras
            camera_id=e["camera_id"],
            # Absolute path: the evaluation loader joins it onto the dataset root,
            # which leaves an absolute path unchanged.
            storage_path=str((snapshot_dir / "images" / f"{f['sha256']}.jpg").resolve()),
            image_label=e["label"],
            event_label=e["label"],
            event_role=e["kind"],
            use_for_fit=False,
        )
        for e in events
        for f in e["frames"]
    ]
    return events, rows


def event_block(
    events: list[dict[str, Any]],
    probs: dict[str, dict[str, float]],
    cfg: PolicyConfig,
    classes: list[str],
) -> dict[str, Any]:
    """Suggested label (the policy's) vs reviewed label, per event."""
    y_true, y_pred, lost, animals = [], [], 0, 0
    per_camera: dict[str, list[tuple[str, str]]] = {}
    for e in events:
        frames = [Frame(probs[f"snapshot:{f['sha256']}"]) for f in e["frames"]]
        suggestion = decide(frames, cfg).label or EMPTY_CLASS
        if e["kind"] in ("supported", "unsupported") and e["label"] != EMPTY_CLASS:
            animals += 1
            lost += suggestion == EMPTY_CLASS
        if e["kind"] == "supported":
            y_true.append(e["label"])
            y_pred.append(suggestion)
            per_camera.setdefault(e["camera_id"], []).append((e["label"], suggestion))
    m = image_metrics(y_true, y_pred, classes)
    return {
        "events_scored": len(y_true),
        "macro_f1": m["macro_f1"],
        "per_class": m["per_class"],
        "animal_events": animals,
        "animal_events_suggested_empty": lost,
        "suggested_empty_ci95": wilson(lost, animals),
        "per_camera": {
            cam: image_metrics([t for t, _ in v], [p for _, p in v], classes)["macro_f1"]
            for cam, v in sorted(per_camera.items())
        },
    }


def run(
    protocol_path: Path, candidate_dir: Path, config_path: Path, report_dir: Path
) -> dict[str, Any]:
    from wildinbox.evaluation.calibration import fit_temperature
    from wildinbox.evaluation.predictors import FinetunedPredictor
    from wildinbox.evaluation.run import _round, _score
    from wildinbox.settings import Settings
    from wildinbox.training.run import git_state, load_context

    protocol = yaml.safe_load(protocol_path.read_text())
    gate = protocol["gate"]
    # The baseline is the protocol's deployed release, resolved and verified
    # before any data is read; never whichever policy happens to be current.
    deployed = resolve_release(
        protocol["deployed_release"],
        Path(protocol.get("models_root", "models")),
        Path(protocol["deployed_policy"]) if protocol.get("deployed_policy") else None,
    )
    deployed_policy = deployed.policy
    cand_meta = json.loads((candidate_dir / "meta.json").read_text())
    if list(cand_meta["classes"]) != list(deployed_policy["classes"]):
        raise GateError("candidate and deployed release disagree on the class order")
    ctx = load_context(config_path, Settings().data_dir)
    snap = cand_meta["trained_on"]["snapshot"]
    if snap is None:
        raise RuntimeError("the candidate was not trained on a snapshot")
    snapshot_dir = Path(snap["path"])
    if load_summary(snapshot_dir)["version"] != snap["version"]:
        raise RuntimeError(f"{snapshot_dir} is no longer snapshot {snap['version']}")
    # What each compared model was fit on beyond the training partition: the
    # deployed release's update snapshot (an earlier cycle) and the candidate's
    # recorded lineage, which must be the snapshot being gated.
    try:
        deployed_meta = json.loads((deployed.model_dir / "meta.json").read_text())
        fitted = {
            "deployed": model_training_frames(deployed_meta, deployed.id),
            "candidate": model_training_frames(cand_meta, cand_meta["name"]),
        }
    except SnapshotError as e:
        raise GateError(str(e)) from e
    if fitted["candidate"] != snapshot_train_frames(snapshot_dir):
        raise GateError(f"the candidate's recorded training frames are not {snapshot_dir}'s")
    leaked = leakage(
        snapshot_dir,
        partition_hashes(ctx.split_dir, GUARDED),
        training=set(partition_hashes(ctx.split_dir, (Partition.TRAIN,))),
        earlier_holdouts=earlier_holdout_hashes(protocol),
        fitted=fitted,
    )
    if leaked:
        raise GateError(
            f"snapshot {snap['version']} breaks content separation {leaked}; "
            "rebuild it with `wildinbox snapshot build` (which excludes these events)"
        )
    classes = list(deployed_policy["classes"])
    released = {k: v for k, v in deployed_policy["released"].items() if k != "policy_version"}
    cfg = PolicyConfig(
        released["empty_threshold"],
        released["species_threshold"],
        released["auto_filter_enabled"],
        released["auto_accept_enabled"],
        tuple(released["accept_species"]) if released["accept_species"] is not None else None,
    )

    models = {
        "deployed": FinetunedPredictor(ctx, deployed.model_dir, "mps"),
        "candidate": FinetunedPredictor(ctx, candidate_dir, "mps"),
    }
    if models["deployed"].weights_digest != deployed.weights_sha256[:12]:
        raise GateError(f"{deployed.model_dir}/model.pt changed while the gate was loading it")
    cal_rows, _ = load_rows(ctx.split_dir, [Partition.CALIBRATION])
    diag_rows, _ = load_rows(ctx.split_dir, [Partition.SEEN_CAMERA_DIAGNOSTIC])
    events, hold_rows = holdout_rows(snapshot_dir)

    temperatures = {"deployed": float(deployed_policy["calibration"]["temperature"])}
    cand_cal = _score(models["candidate"], cal_rows)
    sup = [s for s in cand_cal if s.supported]
    temperatures["candidate"] = fit_temperature(
        np.array([[s.probs[c] for c in classes] for s in sup]),
        np.array([classes.index(s.row.image_label or "") for s in sup]),
    )

    results: dict[str, Any] = {}
    for name, pred in models.items():
        t = temperatures[name]
        hold = pred.score(f"update-holdout-{snap['version']}", hold_rows)
        cal = apply_temperature(hold.embeddings, t)
        probs = {
            r.source_id: {c: float(v) for c, v in zip(classes, p, strict=True)}
            for r, p in zip(hold_rows, cal, strict=True)
        }
        regress = {}
        for part, rows in (
            ("calibration_cameras", cal_rows),
            ("seen_camera_diagnostic", diag_rows),
        ):
            scored = [s for s in _score(pred, rows) if s.supported]
            regress[part] = image_metrics(
                [s.row.image_label or "" for s in scored], [s.pred for s in scored], classes
            )["macro_f1"]
        results[name] = {
            "temperature": t,
            "holdout": event_block(events, probs, cfg, classes),
            "regression_macro_f1": regress,
        }

    d, c = results["deployed"], results["candidate"]
    gain = c["holdout"]["macro_f1"] - d["holdout"]["macro_f1"]
    regressions = {
        part: d["regression_macro_f1"][part] - c["regression_macro_f1"][part]
        for part in d["regression_macro_f1"]
    }
    lost_d = d["holdout"]["animal_events_suggested_empty"]
    lost_c = c["holdout"]["animal_events_suggested_empty"]
    lost_limit = lost_d * (1 + gate["max_false_empty_suggestion_increase"])
    checks = {
        "holdout_gain": {
            "value": gain,
            "required": gate["min_holdout_gain"],
            "pass": gain >= gate["min_holdout_gain"],
        },
        **{
            f"no_regression_{part}": {
                "value": -drop,
                "allowed": -gate["max_regression"],
                "pass": drop <= gate["max_regression"],
            }
            for part, drop in regressions.items()
        },
        "false_empty_suggestions": {
            "value": lost_c,
            "limit": lost_limit,
            "pass": lost_c <= lost_limit,
        },
    }
    budget = gate.get("max_comparisons_per_holdout")
    number = log_comparison(
        report_dir.parent / "comparisons.jsonl",
        {
            "at": datetime.now(UTC).isoformat(),
            "candidate": cand_meta["name"],
            "candidate_weights": models["candidate"].weights_digest,
            "deployed_release": deployed.id,
            "deployed_weights": models["deployed"].weights_digest,
            "holdout_version": snap["version"],
            "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            "holdout_gain": gain,
            "checks_passed": all(v["pass"] for v in checks.values()),
        },
    )
    if budget is not None:
        checks["comparison_budget"] = {
            "value": number,
            "limit": budget,
            "pass": number <= budget,
        }
    promote = all(v["pass"] for v in checks.values())

    # The candidate's policy artifact: deployed policy settings, its own calibration.
    calibration = {
        "method": "temperature",
        "temperature": temperatures["candidate"],
        "fit_partition": Partition.CALIBRATION.value,
        "weights_digest": models["candidate"].weights_digest,
    }
    calibration["version"] = _version(calibration)
    policy = {
        "policy": POLICY_NAME,
        "model": cand_meta["name"],
        "classes": classes,
        "calibration": calibration,
        "unfamiliar": None,
        "released": deployed_policy["released"],
        "rule": deployed_policy["rule"],
    }
    policy["artifact_version"] = _version(policy)

    out = {
        "protocol": str(protocol_path),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "deployed_release": deployed.id,
        "deployed": {
            "model_dir": str(deployed.model_dir),
            "policy": str(deployed.policy_path),
            "weights_sha256": deployed.weights_sha256,
            "calibration_version": deployed_policy["calibration"]["version"],
        },
        "candidate": cand_meta["name"],
        "candidate_release": f"{cand_meta['name']}@{policy['artifact_version']}",
        "snapshot": snap,
        "code": git_state(),
        "results": results,
        "checks": checks,
        "promote": promote,
        "development_data_only": {
            "evaluated_on": ["snapshot holdout", "calibration", "seen_camera_diagnostic"],
            "final_test_read": False,
            "protected_frames_in_snapshot": 0,
            "comparison_number_on_this_holdout": number,
            "comparison_budget": budget,
        },
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "metrics.json").write_text(json.dumps(_round(out), indent=2) + "\n")
    (report_dir / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    (candidate_dir / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    from wildinbox.training.gate_report import write_report

    write_report(report_dir, out)
    return out
