"""Replay event decisions from saved predictions and a versioned policy.

Acceptance gate: every event disposition can be reproduced from its saved
frame predictions and the policy artifact (calibration, unfamiliar-input
threshold, and policy configuration). `wildinbox replay` and CI run this.
"""

from __future__ import annotations

import gzip
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from wildinbox.inference.calibration import apply_temperature
from wildinbox.policy.conservative import POLICY_NAME, Frame, PolicyConfig, decide
from wildinbox.schemas import FrameStatus

CONFIG_NAMES = ("released", "rule")


def policy_config(d: dict[str, Any]) -> PolicyConfig:
    species = d.get("accept_species")
    return PolicyConfig(
        d["empty_threshold"],
        d["species_threshold"],
        d["auto_filter_enabled"],
        d["auto_accept_enabled"],
        tuple(species) if species is not None else None,
    )


def frames_from_saved(row: dict[str, Any], policy: dict[str, Any]) -> list[Frame]:
    classes = policy["classes"]
    temperature = policy["calibration"]["temperature"]
    unf = policy.get("unfamiliar")
    threshold = unf["threshold"] if unf and unf["adopted"] else None
    frames = []
    for f in row["frames"]:
        status = FrameStatus(f["status"])
        probs = None
        if status is FrameStatus.COMPLETED:
            cal = apply_temperature(np.array([f["raw_probs"]]), temperature)[0]
            probs = {c: float(v) for c, v in zip(classes, cal, strict=True)}
        d = f.get("unfamiliar_distance")
        flagged = threshold is not None and d is not None and d > threshold
        frames.append(Frame(probs, status, flagged))
    return frames


def replay(decisions_path: Path, policy: dict[str, Any]) -> dict[str, Any]:
    if policy["policy"] != POLICY_NAME:
        raise ValueError(f"artifact is for {policy['policy']}, this code is {POLICY_NAME}")
    configs = {name: policy_config(policy[name]) for name in CONFIG_NAMES}
    problems = [
        f"{name}: policy_version {policy[name]['policy_version']} does not match its settings"
        for name, cfg in configs.items()
        if policy[name]["policy_version"] != f"{POLICY_NAME}+{cfg.fingerprint()}"
    ]
    counts: dict[str, Counter[str]] = {name: Counter() for name in CONFIG_NAMES}
    n = 0
    with gzip.open(decisions_path, "rt") as f:
        for line in f:
            row = json.loads(line)
            n += 1
            frames = frames_from_saved(row, policy)
            for name, cfg in configs.items():
                o = decide(frames, cfg)
                saved = row[name]
                got = (o.disposition.value, o.label, [r.value for r in o.reasons])
                want = (saved["disposition"], saved["label"], saved["reasons"])
                same_conf = (o.confidence is None and saved["confidence"] is None) or (
                    o.confidence is not None
                    and saved["confidence"] is not None
                    and math.isclose(o.confidence, saved["confidence"], abs_tol=1e-9)
                )
                if got != want or not same_conf:
                    problems.append(f"{name} {row['event_id']}: saved {want}, replayed {got}")
                counts[name][o.disposition.value] += 1
    return {
        "events": n,
        "problems": problems,
        "dispositions": {k: dict(sorted(v.items())) for k, v in counts.items()},
    }
