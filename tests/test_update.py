from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from wildinbox.policy.conservative import PolicyConfig
from wildinbox.settings import Settings
from wildinbox.training.finetune import snapshot_rows
from wildinbox.training.gate import event_block, holdout_rows
from wildinbox.training.snapshot import review_label, split_by_time

from .test_app import database_url, settings  # noqa: F401
from .test_uploads import jpeg

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


def test_snapshot_uses_approved_labels_and_excludes_protected_records_with_their_corrections(
    settings: Settings,  # noqa: F811
    tmp_path: Path,
) -> None:
    """Corrections -> snapshot, through the API: approved reviewers only; frames of
    protected evaluation partitions (by hash or source file name) and earlier
    holdout events are excluded together with every review on them; the rest
    keeps full label provenance."""
    import gzip
    import hashlib

    from fastapi.testclient import TestClient

    from wildinbox.api.app import create_app
    from wildinbox.training.snapshot import build

    photos = {f"p{i}.jpg": jpeg(100 + i) for i in range(7)}
    photos["0a1b-final-test-frame.jpg"] = jpeg(200)  # protected by file name
    final_sha = hashlib.sha256(photos["p6.jpg"]).hexdigest()  # protected by hash
    splits = tmp_path / "images.jsonl.gz"
    with gzip.open(splits, "wt") as fh:
        for row in (
            {"partition": "final_test", "sha256": final_sha, "source_id": "x"},
            {"partition": "calibration", "sha256": "0" * 64, "source_id": "0a1b-final-test-frame"},
            {"partition": "train", "sha256": "1" * 64, "source_id": "p0"},  # not protected
        ):
            fh.write(json.dumps(row) + "\n")
    meta = {
        "files": {
            name: {
                "camera_id": "cct-90",
                "captured_at": f"2012-01-0{i + 1}T10:00:00",
                "sequence_id": f"seq-{i}",
            }
            for i, name in enumerate(sorted(photos))
        }
    }
    with TestClient(create_app(settings)) as c:
        batch = c.post(
            "/batches",
            files=[("files", (n, d, "image/jpeg")) for n, d in sorted(photos.items())],
            data={"metadata": json.dumps(meta)},
        ).json()
        events = c.get("/events", params={"batch_id": batch["id"], "limit": 50}).json()["events"]
        assert len(events) == 8
        by_file = {c.get(f"/events/{e['id']}").json()["images"][0]["filename"]: e for e in events}

        def review(name: str, who: str, label: str) -> None:
            e = by_file[name]
            r = c.post(
                f"/events/{e['id']}/reviews",
                json={"reviewer": who, "outcome": "corrected", "confirmed_label": label},
            )
            assert r.status_code == 201, r.text

        for name in ("p0.jpg", "p1.jpg", "p2.jpg", "p3.jpg", "p4.jpg"):
            review(name, "ranger", "cat")
        review("p5.jpg", "visitor", "cat")  # not an approved reviewer
        review("p6.jpg", "ranger", "cat")  # final-test frame, corrected twice
        review("p6.jpg", "ranger", "bobcat")
        review("0a1b-final-test-frame.jpg", "ranger", "coyote")
        earlier_holdout = tmp_path / "earlier-holdout.jsonl"
        earlier_holdout.write_text(json.dumps({"event_id": by_file["p4.jpg"]["id"]}) + "\n")

        protocol = tmp_path / "protocol.yaml"
        protocol.write_text(
            yaml.safe_dump(
                {
                    "name": "update2",
                    "deployed_release": "test-predictor-v0",
                    "deployment": {"cameras": ["90"], "reviewer": "unused"},
                    "approval": {"reviewers": ["ranger"]},
                    "protected": {
                        "splits": "images.jsonl.gz",
                        "partitions": ["final_test", "calibration"],
                        "snapshot_holdouts": ["earlier-holdout.jsonl"],
                    },
                }
            )
        )
        out = build(protocol, "http://testserver", tmp_path / "snaps", client=c, root=tmp_path)

    summary = json.loads((out / "snapshot.json").read_text())
    assert out.name == f"update2-{summary['version']}"
    assert summary["reviews_not_approved"] == 1
    excluded = {x["event_id"]: x for x in summary["excluded"]["records"]}
    assert summary["excluded"]["by_reason"] == {
        "protected_partition_sha256": 1,
        "protected_partition_filename": 1,
        "earlier_snapshot_holdout": 1,
    }
    # Both reviews of the corrected final-test frame are excluded with it.
    assert len(excluded[by_file["p6.jpg"]["id"]]["review_ids"]) == 2
    train = [json.loads(x) for x in (out / "train.jsonl").read_text().splitlines()]
    hold = [json.loads(x) for x in (out / "holdout.jsonl").read_text().splitlines()]
    used = {r["event_id"] for r in train} | {h["event_id"] for h in hold}
    assert used == {by_file[n]["id"] for n in ("p0.jpg", "p1.jpg", "p2.jpg", "p3.jpg")}
    assert final_sha not in {r["sha256"] for r in train}
    labels = {
        r["event_id"]: r for r in map(json.loads, (out / "labels.jsonl").read_text().splitlines())
    }
    p0 = labels[by_file["p0.jpg"]["id"]]
    assert p0["use"] == "train" and p0["reviewer"] == "ranger" and p0["outcome"] == "corrected"
    assert p0["suggested_by_release"] == "test-predictor-v0" and len(p0["review_chain"]) == 1
    assert labels[by_file["p6.jpg"]["id"]]["use"] == "excluded:protected_partition_sha256"
    assert summary["provenance"]["events"] == len(labels) == 7


def test_gate_refuses_leaked_snapshots_and_counts_repeated_comparisons(tmp_path: Path) -> None:
    from wildinbox.training.gate import leakage, log_comparison

    snap = _snapshot(tmp_path)
    assert leakage(snap, {"zz": "final_test"}) == {}
    assert leakage(snap, {"bb": "calibration", "c2": "final_test"}) == {
        "calibration": 1,
        "final_test": 1,
    }
    log = tmp_path / "comparisons.jsonl"
    assert log_comparison(log, {"holdout_version": "h1", "candidate": "a"}) == 1
    assert log_comparison(log, {"holdout_version": "h1", "candidate": "b"}) == 2
    assert log_comparison(log, {"holdout_version": "h2", "candidate": "b"}) == 1
    assert len(log.read_text().splitlines()) == 3
