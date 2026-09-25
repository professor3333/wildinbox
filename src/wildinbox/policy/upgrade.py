"""`wildinbox policy upgrade`: the same calibration and thresholds under a newer
policy version, with its own saved decisions so replay covers it.

The new artifact reuses the source artifact's calibration and configurations
unchanged; only the policy name (and therefore the policy versions and the
artifact version) changes. Decisions are recomputed from the source's saved
frame predictions, so the new file replays exactly.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from wildinbox.policy.conservative import POLICIES, decide
from wildinbox.policy.replay import CONFIG_NAMES, frames_from_saved, policy_config


def _outcome(o: Any) -> dict[str, Any]:
    return {
        "disposition": o.disposition.value,
        "label": o.label,
        "confidence": o.confidence,
        "reasons": [r.value for r in o.reasons],
    }


def upgrade(source: Path, decisions: Path, policy_name: str, out_dir: Path) -> dict[str, Any]:
    if policy_name not in POLICIES:
        raise ValueError(f"unknown policy {policy_name!r}")
    old = json.loads(source.read_text())
    new = {k: v for k, v in old.items() if k != "artifact_version"}
    new["policy"] = policy_name
    for name in CONFIG_NAMES:
        cfg = policy_config(old[name])
        new[name] = {**old[name], "policy_version": cfg.version(policy_name)}
    new["derived_from"] = {
        "artifact_version": old["artifact_version"],
        "policy": old["policy"],
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    new["artifact_version"] = hashlib.sha256(json.dumps(new, sort_keys=True).encode()).hexdigest()[
        :12
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    configs = {name: policy_config(new[name]) for name in CONFIG_NAMES}
    changed = 0
    with gzip.open(decisions, "rt") as src, gzip.open(out_dir / "decisions.jsonl.gz", "wt") as dst:
        for line in src:
            row = json.loads(line)
            frames = frames_from_saved(row, new)
            for name, cfg in configs.items():
                o = _outcome(decide(frames, cfg, policy_name))
                changed += o != row[name]
                row[name] = o
            dst.write(json.dumps(row) + "\n")
    (out_dir / "policy.json").write_text(json.dumps(new, indent=2) + "\n")
    return {"artifact_version": new["artifact_version"], "decisions_changed": changed}
