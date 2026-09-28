"""`wildinbox adaptation develop`: compare adaptation methods on the
adaptation-development cameras and write reports/adaptation/development."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

from wildinbox.adaptation.data import FrameOutputs, load_camera_events, score_frames
from wildinbox.adaptation.evaluate import (
    AUDIT_RATE,
    FALSE_EMPTY_LIMIT,
    PRECISION_TARGET,
    in_sample,
    leave_one_camera_out,
    score_camera,
)
from wildinbox.adaptation.methods import CameraHead, Method, PriorShift, Release
from wildinbox.datasets.spec import Partition

SPLIT = "cct-fresh-dev-v1"
IMAGES = "cct_fresh_dev"
HEAD_PER_CLASS = 250
SEED = 20260928


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
    methods: list[Method] = [
        Release(t),
        PriorShift(t),
        CameraHead(x_train, y_train),
    ]
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
        "results": results,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "metrics.json").write_text(json.dumps(payload, indent=2, default=float) + "\n")
    return payload


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def _line(r: dict[str, Any]) -> str:
    p = r["leave_one_camera_out"]["pooled"]
    return (
        f"LOCO reduction {_pct(p['review_reduction'])}, precision {_pct(p['precision'])} "
        f"({p['accepted_correct']}/{p['accepted']}), retention {_pct(p['retention'])}"
    )
