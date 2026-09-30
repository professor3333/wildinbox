"""`wildinbox adaptation develop`: compare adaptation methods on the
adaptation-development cameras and write reports/adaptation/development."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

from wildinbox.adaptation.data import (
    CameraEvent,
    FrameOutputs,
    load_camera_events,
    score_frames,
)
from wildinbox.adaptation.evaluate import (
    AUDIT_RATE,
    FALSE_EMPTY_LIMIT,
    PRECISION_TARGET,
    Scored,
    in_sample,
    leave_one_camera_out,
    nested_leave_one_camera_out,
    score_camera,
)
from wildinbox.adaptation.methods import (
    LABELLED_ROLES,
    CameraHead,
    Method,
    PriorShift,
    Release,
    reviewed_frames,
)
from wildinbox.datasets.spec import Partition
from wildinbox.evaluation.data import ImageRow

SPLIT = "cct-fresh-dev-v1"
IMAGES = "cct_fresh_dev"
HEAD_PER_CLASS = 250
SEED = 20260928
HEAD_C = (0.001, 0.003, 0.01, 0.03)
HEAD_SHARE = (0.1, 0.25, 0.5)


def unseen_base(
    config: Path, model_dir: Path, data_dir: Path, device: str
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Frames from the CCT20 cameras the network never trained on (calibration,
    policy validation) with their labels, unsupported animals as OTHER."""
    split = data_dir / "splits" / "cct20-splits-v1"
    cams, rows = load_camera_events(split, [Partition.CALIBRATION, Partition.POLICY_VALIDATION])
    out = score_frames(
        config,
        model_dir,
        split,
        data_dir / "raw" / "cct20" / "images",
        rows,
        "cct20-unseen-dev",
        device,
    )
    ids, labels = reviewed_frames([e for ev in cams.values() for e in ev], with_other=True)
    return out.features[out.rows(ids)], tuple(labels)


def training_sample(
    model_dir: Path, cct20_split: Path, classes: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """A class-balanced, seeded sample of the release's training-frame features
    (unfamiliar-reference.npz) with their labels."""
    ref = np.load(model_dir / "unfamiliar-reference.npz")
    with gzip.open(cct20_split / "images.jsonl.gz", "rt") as f:
        label = {
            r["source_id"]: r["image_label"]
            for r in map(json.loads, f)
            if r["partition"] == "train" and r["use_for_fit"]
        }
    rng = np.random.default_rng(SEED)
    cls = {c: i for i, c in enumerate(classes)}
    rows: list[int] = []
    y: list[int] = []
    for c in classes:
        idx = [i for i, sid in enumerate(ref["ids"]) if label.get(str(sid)) == c]
        pick = rng.choice(idx, size=min(HEAD_PER_CLASS, len(idx)), replace=False)
        rows += list(pick)
        y += [cls[c]] * len(pick)
    return ref["features"][np.array(rows)], np.array(y)


FINETUNED_N = 50


def finetuned_dir(models_dir: Path, camera: str, n: int = FINETUNED_N) -> Path:
    """configs/experiments/adaptation/finetune-e3-cam<camera>-n<n>.yaml"""
    return models_dir / f"finetune-e3-adapt-cam{camera}-n{n}"


def refit_temperature(config: Path, model_dir: Path, data_dir: Path, device: str) -> float:
    """The candidate's temperature on the CCT20 calibration cameras, as the
    update gate refits it."""
    from wildinbox.evaluation.calibration import fit_temperature
    from wildinbox.evaluation.data import load_rows

    split = data_dir / "splits" / "cct20-splits-v1"
    rows, _ = load_rows(split, [Partition.CALIBRATION])
    out = score_frames(
        config, model_dir, split, data_dir / "raw" / "cct20" / "images", rows, "calibration", device
    )
    keep = [
        i
        for i, r in enumerate(rows)
        if r.image_label in out.classes and r.event_role in LABELLED_ROLES
    ]
    labels = np.array([out.classes.index(rows[i].image_label or "") for i in keep])
    return fit_temperature(np.exp(out.log_probs[keep]), labels)


def finetuned_results(
    config: Path,
    models_dir: Path,
    data_dir: Path,
    device: str,
    cams: dict[str, list[CameraEvent]],
    rows: list[ImageRow],
) -> dict[str, Any]:
    """Per-camera fine-tuned models (each trained with its camera's first
    FINETUNED_N reviewed events), alone and with the other-animal head on
    their features. Each camera is scored only by its own model."""
    split = data_dir / "splits" / SPLIT
    per_camera: dict[str, tuple[list[Scored], int]] = {}
    per_config: dict[str, dict[str, tuple[list[Scored], int]]] = {}
    temperatures: dict[str, float] = {}
    for cam, events in cams.items():
        mdir = finetuned_dir(models_dir, cam)
        cam_rows = [r for r in rows if r.camera_id == cam]
        out = score_frames(
            config,
            mdir,
            split,
            data_dir / "raw" / IMAGES / "images",
            cam_rows,
            f"fresh-dev-v1-cam{cam}",
            device,
        )
        temperatures[cam] = refit_temperature(config, mdir, data_dir, device)
        per_camera[cam] = score_camera(
            Release(temperatures[cam], name="finetuned"), events, FINETUNED_N, out
        )
        x_base, y_base = unseen_base(config, mdir, data_dir, device)
        for c in HEAD_C:
            for share in HEAD_SHARE:
                head = CameraHead(x_base, y_base, with_other=True, camera_share=share, c=c)
                per_config.setdefault(f"C={c},share={share}", {})[cam] = score_camera(
                    head, events, FINETUNED_N, out
                )
    results = {
        f"finetuned@{FINETUNED_N}": {
            "method": "finetuned",
            "n": FINETUNED_N,
            "temperatures": temperatures,
            "leave_one_camera_out": leave_one_camera_out(per_camera),
            "in_sample": in_sample(per_camera),
        },
        f"finetuned_head_other@{FINETUNED_N}": {
            "method": "finetuned_head_other",
            "n": FINETUNED_N,
            "leave_one_camera_out": nested_leave_one_camera_out(per_config),
            "per_config_in_sample": {cfg: in_sample(pc) for cfg, pc in per_config.items()},
        },
    }
    for key, r in results.items():
        print(f"{key}: {_line(r)}", flush=True)
    return results


def run(
    config: Path, model_dir: Path, data_dir: Path, report_dir: Path, device: str, ns: list[int]
) -> dict[str, Any]:
    split = data_dir / "splits" / SPLIT
    cams, rows = load_camera_events(split, [Partition.ADAPTATION_DEVELOPMENT])
    out: FrameOutputs = score_frames(
        config, model_dir, split, data_dir / "raw" / IMAGES / "images", rows, "fresh-dev-v1", device
    )
    policy = json.loads((model_dir / "policy.json").read_text())
    t = float(policy["calibration"]["temperature"])
    x_train, y_train = training_sample(
        model_dir, data_dir / "splits" / "cct20-splits-v1", out.classes
    )
    x_base, y_base = unseen_base(config, model_dir, data_dir, device)
    methods: list[Method] = [
        Release(t),
        PriorShift(t),
        CameraHead(x_train, tuple(out.classes[i] for i in y_train)),
    ]
    # Each family's configuration is chosen by nested leave-one-camera-out.
    # `unseen_head` is the ablation: the same head without the camera's reviews.
    families = {
        family: {
            f"C={c},share={share}": CameraHead(
                x_base, y_base, with_other=True, camera_share=share, c=c, name=family
            )
            for c in HEAD_C
            for share in shares
        }
        for family, shares in (("camera_head_other", HEAD_SHARE), ("unseen_head", (0.0,)))
    }
    results: dict[str, Any] = {}
    for n in ns:
        for m in methods:
            per_camera = {cam: score_camera(m, ev, n, out) for cam, ev in cams.items()}
            results[f"{m.name}@{n}"] = {
                "method": m.name,
                "n": n,
                "leave_one_camera_out": leave_one_camera_out(per_camera),
                "in_sample": in_sample(per_camera),
            }
            print(f"{m.name}@{n}: {_line(results[f'{m.name}@{n}'])}", flush=True)
        for family, heads in families.items():
            per_config = {
                cfg: {cam: score_camera(h, ev, n, out) for cam, ev in cams.items()}
                for cfg, h in heads.items()
            }
            key = f"{family}@{n}"
            results[key] = {
                "method": family,
                "n": n,
                # configuration and thresholds both chosen without the held-out camera
                "leave_one_camera_out": nested_leave_one_camera_out(per_config),
                "per_config_in_sample": {cfg: in_sample(pc) for cfg, pc in per_config.items()},
            }
            print(f"{key}: {_line(results[key])}", flush=True)
            if family == "camera_head_other":
                # the same heads with empty filtering off: species labels only
                key = f"camera_head_other_species_only@{n}"
                results[key] = {
                    "method": "camera_head_other_species_only",
                    "n": n,
                    "leave_one_camera_out": nested_leave_one_camera_out(
                        per_config, filter_empty=False
                    ),
                    "per_config_in_sample": {
                        cfg: in_sample(pc, filter_empty=False) for cfg, pc in per_config.items()
                    },
                }
                print(f"{key}: {_line(results[key])}", flush=True)
    models_dir = model_dir.parent
    if all((finetuned_dir(models_dir, cam) / "model.pt").exists() for cam in cams):
        results |= finetuned_results(config, models_dir, data_dir, device, cams, rows)
    else:
        print("per-camera fine-tuned models not all trained; skipped", flush=True)
    payload = {
        "split": SPLIT,
        "release": {"model": model_dir.name, "temperature": t},
        "cameras": {c: len(e) for c, e in sorted(cams.items(), key=lambda kv: (len(kv[0]), kv[0]))},
        "rule": {
            "precision_lower_bound": PRECISION_TARGET,
            "false_empty_upper_bound": FALSE_EMPTY_LIMIT,
            "audit_rate": AUDIT_RATE,
        },
        "head": {"per_class": HEAD_PER_CLASS, "seed": SEED, "camera_share": 0.5, "c": 0.1},
        "head_other": {
            "base": "CCT20 calibration and policy-validation cameras (never trained on)",
            "base_frames": len(y_base),
            "c_grid": HEAD_C,
            "camera_share_grid": HEAD_SHARE,
        },
        "results": results,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "metrics.json").write_text(json.dumps(payload, indent=2, default=float) + "\n")
    return payload


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def _line(r: dict[str, Any]) -> str:
    p = r["leave_one_camera_out"]["pooled"]
    cams = r["leave_one_camera_out"]["cameras"].values()
    worst = min((c["retention"] for c in cams if c["retention"] is not None), default=None)
    return (
        f"LOCO reduction {_pct(p['review_reduction'])}, precision {_pct(p['precision'])} "
        f"({p['accepted_correct']}/{p['accepted']}), retention {_pct(p['retention'])} "
        f"(worst camera {_pct(worst)})"
    )
