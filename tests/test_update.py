from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from wildinbox.policy.conservative import PolicyConfig
from wildinbox.training.finetune import snapshot_rows
from wildinbox.training.gate import event_block, holdout_rows
from wildinbox.training.snapshot import review_label, split_by_time

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["empty", "bobcat", "cat", "coyote", "dog", "opossum", "rabbit", "raccoon"]


def _ev(start: str | None, outcome: str = "confirmed", label: str | None = "cat") -> dict[str, Any]:
    return {"start_at": start, "latest_review": {"outcome": outcome, "confirmed_label": label}}


def test_time_split_never_puts_later_events_in_training() -> None:
    events = [_ev(f"2012-01-0{d}T10:00:00") for d in range(1, 8)] + [_ev(None)]
    before, after, cutoff = split_by_time(events)
    assert cutoff == "2012-01-04T10:00:00"
    assert [e["start_at"] for e in before] == [f"2012-01-0{d}T10:00:00" for d in (1, 2, 3)]
    assert all(e["start_at"] >= cutoff for e in after) and len(after) == 4
    assert max(e["start_at"] for e in before) < min(e["start_at"] for e in after)


def test_review_labels() -> None:
    assert review_label(_ev("x", "corrected", "skunk")) == ("skunk", "reviewed")
    assert review_label(_ev("x", "unresolved", None)) == (None, "unresolved")


def _snapshot(tmp_path: Path) -> Path:
    d = tmp_path / "snap"
    (d / "images").mkdir(parents=True)
    (d / "snapshot.json").write_text(json.dumps({"version": "abc123", "train": {}}))
    (d / "train.jsonl").write_text(
        json.dumps({"sha256": "aa", "label": "empty", "event_id": "e1", "camera_id": "cct-90"})
        + "\n"
        + json.dumps({"sha256": "bb", "label": "cat", "event_id": "e2", "camera_id": "cct-90"})
        + "\n"
    )
    holdout = [
        {
            "event_id": "h1",
            "camera_id": "cct-90",
            "label": "cat",
            "kind": "supported",
            "frames": [{"sha256": "c1"}, {"sha256": "c2"}],
        },
        {
            "event_id": "h2",
            "camera_id": "cct-90",
            "label": "coyote",
            "kind": "supported",
            "frames": [{"sha256": "d1"}],
        },
        {
            "event_id": "h3",
            "camera_id": "cct-125",
            "label": "skunk",
            "kind": "unsupported",
            "frames": [{"sha256": "e1"}],
        },
        {
            "event_id": "h4",
            "camera_id": "cct-125",
            "label": None,
            "kind": "unresolved",
            "frames": [{"sha256": "f1"}],
        },
    ]
    (d / "holdout.jsonl").write_text("".join(json.dumps(h) + "\n" for h in holdout))
    return d


def test_snapshot_rows_are_fit_rows_with_reviewed_labels(tmp_path: Path) -> None:
    rows, sources, summary = snapshot_rows(_snapshot(tmp_path))
    assert [(r.image_label, r.event_role, r.use_for_fit) for r in rows] == [
        ("empty", "empty", True),
        ("cat", "supported_species", True),
    ]
    assert all(r.source_id.startswith("snapshot:") for r in rows)
    assert sources[rows[0].storage_path].name == "aa.jpg" and summary["version"] == "abc123"


def test_gate_scores_events_by_the_policy_suggestion(tmp_path: Path) -> None:
    events, rows = holdout_rows(_snapshot(tmp_path))
    assert len(rows) == 5 and all(Path(r.storage_path).is_absolute() for r in rows)

    def p(best: str) -> dict[str, float]:
        return {c: (0.9 if c == best else 0.1 / 7) for c in CLASSES}

    probs = {
        "snapshot:c1": p("cat"),
        "snapshot:c2": p("cat"),
        "snapshot:d1": p("empty"),  # a coyote suggested as empty
        "snapshot:e1": p("rabbit"),
        "snapshot:f1": p("empty"),
    }
    cfg = PolicyConfig(0.65, None, False, False, ())
    m = event_block(events, probs, cfg, CLASSES)
    assert m["events_scored"] == 2  # supported labels only
    assert m["animal_events"] == 3 and m["animal_events_suggested_empty"] == 1
    assert m["per_class"]["cat"]["recall"] == 1.0 and m["per_class"]["coyote"]["recall"] == 0.0


def test_committed_protocol_has_the_gate_the_code_reads() -> None:
    protocol = yaml.safe_load((REPO_ROOT / "configs/experiments/update_cycle.yaml").read_text())
    gate = protocol["gate"]
    assert {"min_holdout_gain", "max_regression", "max_false_empty_suggestion_increase"} <= set(
        gate
    )
    assert protocol["deployment"]["reviewer"] == "simulated-ground-truth"
