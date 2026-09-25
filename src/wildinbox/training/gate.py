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


def _version(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


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


def leakage(snapshot_dir: Path, guarded: dict[str, str]) -> dict[str, int]:
    """Snapshot frames (train and holdout) that belong to guarded partitions."""
    found: dict[str, int] = {}
    train = [json.loads(x) for x in (snapshot_dir / "train.jsonl").read_text().splitlines()]
    hold = [json.loads(x) for x in (snapshot_dir / "holdout.jsonl").read_text().splitlines()]
    shas = [r["sha256"] for r in train] + [f["sha256"] for e in hold for f in e["frames"]]
    for sha in shas:
        if sha in guarded:
            found[guarded[sha]] = found.get(guarded[sha], 0) + 1
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
    ctx = load_context(config_path, Settings().data_dir)
    cand_meta = json.loads((candidate_dir / "meta.json").read_text())
    snap = cand_meta["trained_on"]["snapshot"]
    if snap is None:
        raise RuntimeError("the candidate was not trained on a snapshot")
    snapshot_dir = Path(snap["path"])
    leaked = leakage(snapshot_dir, partition_hashes(ctx.split_dir, GUARDED))
    if leaked:
        raise RuntimeError(
            f"snapshot {snap['version']} contains protected evaluation frames {leaked}; "
            "rebuild it with `wildinbox snapshot build` (which excludes them)"
        )
    deployed_policy = json.loads(Path("reports/calibration/policy.json").read_text())
    deployed_dir = Path("models") / deployed_policy["model"]
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
        "deployed": FinetunedPredictor(ctx, deployed_dir, "mps"),
        "candidate": FinetunedPredictor(ctx, candidate_dir, "mps"),
    }
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
        "deployed_release": protocol["deployed_release"],
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
