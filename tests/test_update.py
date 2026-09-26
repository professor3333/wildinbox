from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from wildinbox.policy.conservative import PolicyConfig
from wildinbox.settings import Settings
from wildinbox.training.finetune import snapshot_record, snapshot_rows
from wildinbox.training.gate import event_block, holdout_rows
from wildinbox.training.snapshot import (
    SNAPSHOT_SCHEMA,
    SnapshotError,
    load_summary,
    review_label,
    split_by_time,
)

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


FRAME_A, FRAME_B = jpeg(1), jpeg(2)
AA, BB = hashlib.sha256(FRAME_A).hexdigest(), hashlib.sha256(FRAME_B).hexdigest()


def _snapshot(tmp_path: Path) -> Path:
    d = tmp_path / "snap"
    (d / "images").mkdir(parents=True)
    (d / "images" / f"{AA}.jpg").write_bytes(FRAME_A)
    (d / "images" / f"{BB}.jpg").write_bytes(FRAME_B)
    (d / "labels.jsonl").write_text('{"event_id": "e1"}\n')
    summary = {
        "schema": SNAPSHOT_SCHEMA,
        "version": "abc123",
        "approved_reviewers": ["ranger"],
        "train": {"images": 2, "events": 2},
        "provenance": {
            "file": "labels.jsonl",
            "sha256": hashlib.sha256((d / "labels.jsonl").read_bytes()).hexdigest(),
        },
    }
    (d / "snapshot.json").write_text(json.dumps(summary))
    (d / "train.jsonl").write_text(
        json.dumps({"sha256": AA, "label": "empty", "event_id": "e1", "camera_id": "cct-90"})
        + "\n"
        + json.dumps({"sha256": BB, "label": "cat", "event_id": "e2", "camera_id": "cct-90"})
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
    assert sources[rows[0].storage_path].name == f"{AA}.jpg" and summary["version"] == "abc123"


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
    # What training reads back from a freshly built snapshot, down to the
    # model metadata (a stale field here crashed training after the epoch).
    assert summary["schema"] == SNAPSHOT_SCHEMA
    rows, _, read = snapshot_rows(out)
    assert len(rows) == summary["train"]["images"] > 0
    assert snapshot_record(out, read) == {
        "version": summary["version"],
        "path": str(out),
        "images": summary["train"]["images"],
        "events": summary["train"]["events"],
        "schema": SNAPSHOT_SCHEMA,
        "approved_reviewers": ["ranger"],
        "provenance_sha256": summary["provenance"]["sha256"],
    }
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
    assert leakage(snap, {BB: "calibration", "c2": "final_test"}) == {
        "calibration": 1,
        "final_test": 1,
    }
    log = tmp_path / "comparisons.jsonl"
    assert log_comparison(log, {"holdout_version": "h1", "candidate": "a"}) == 1
    assert log_comparison(log, {"holdout_version": "h1", "candidate": "b"}) == 2
    assert log_comparison(log, {"holdout_version": "h2", "candidate": "b"}) == 1
    assert len(log.read_text().splitlines()) == 3


def test_legacy_snapshot_is_read_with_its_single_reviewer(tmp_path: Path) -> None:
    d = _snapshot(tmp_path)
    legacy = json.loads((d / "snapshot.json").read_text())
    del legacy["schema"], legacy["approved_reviewers"], legacy["provenance"]
    legacy["reviewer"] = "simulated-ground-truth"
    (d / "snapshot.json").write_text(json.dumps(legacy))
    record = snapshot_record(d, load_summary(d))
    assert record["schema"] == "snapshot/v1" and record["provenance_sha256"] is None
    assert record["approved_reviewers"] == ["simulated-ground-truth"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda s, d: s.update(schema="snapshot/v9"), "unknown snapshot schema"),
        (lambda s, d: s.pop("approved_reviewers"), "approved_reviewers"),
        (lambda s, d: s.update(approved_reviewers=[]), "approved_reviewers"),
        (lambda s, d: s.update(train={"images": 2}), "train.images"),
        (lambda s, d: s.pop("provenance"), "provenance"),
        (lambda s, d: (d / "labels.jsonl").write_text("{}\n"), "SHA-256 differs"),
        (lambda s, d: (d / "train.jsonl").unlink(), "train.jsonl"),
    ],
)
def test_training_refuses_a_snapshot_it_cannot_use(
    tmp_path: Path, change: Any, message: str
) -> None:
    d = _snapshot(tmp_path)
    summary = json.loads((d / "snapshot.json").read_text())
    change(summary, d)
    (d / "snapshot.json").write_text(json.dumps(summary))
    with pytest.raises(SnapshotError, match=message):
        snapshot_rows(d)


def test_snapshot_separates_content_whatever_its_name_camera_or_time(
    settings: Settings,  # noqa: F811
    tmp_path: Path,
) -> None:
    """Identical bytes re-uploaded as other events (another camera, time, or file
    name) never reach training from a holdout, current or earlier; overlaps are
    resolved by whole events, and the gate re-derives the same invariants."""
    import gzip

    from fastapi.testclient import TestClient

    from wildinbox.api.app import create_app
    from wildinbox.training.gate import leakage
    from wildinbox.training.snapshot import build

    def sha(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    dup, seq_dup, fit_img, old = jpeg(301), jpeg(302), jpeg(303), jpeg(304)
    uploads = [  # (file name, bytes, camera, day, sequence)
        ("d-early.jpg", dup, "cct-90", 1, "s1"),
        ("seq-a.jpg", jpeg(305), "cct-90", 2, "s2"),
        ("seq-b.jpg", seq_dup, "cct-90", 2, "s2"),
        ("f1.jpg", jpeg(306), "cct-90", 3, "s3"),
        ("f2.jpg", jpeg(307), "cct-90", 4, "s4"),
        ("renamed-train-image.jpg", fit_img, "cct-90", 5, "s5"),
        ("seq-b-again.jpg", seq_dup, "cct-90", 6, "s6"),
        ("f3.jpg", jpeg(308), "cct-90", 7, "s7"),
        ("renamed-old-holdout.jpg", old, "cct-125", 1, "t1"),
        ("g1.jpg", jpeg(309), "cct-125", 2, "t2"),
        ("g2.jpg", jpeg(310), "cct-125", 3, "t3"),
        ("d-late.jpg", dup, "cct-125", 4, "t4"),
        ("g3.jpg", jpeg(311), "cct-125", 5, "t5"),
    ]
    splits = tmp_path / "images.jsonl.gz"
    with gzip.open(splits, "wt") as fh:
        fh.write(json.dumps({"partition": "train", "sha256": sha(fit_img), "source_id": "t"}))
    earlier = tmp_path / "earlier-holdout.jsonl"
    earlier.write_text(json.dumps({"event_id": "gone", "frames": [{"sha256": sha(old)}]}) + "\n")

    with TestClient(create_app(settings)) as c:
        by_file: dict[str, dict[str, Any]] = {}
        for seq in dict.fromkeys(u[4] for u in uploads):  # one batch per sequence
            files = [u for u in uploads if u[4] == seq]
            meta = {
                "files": {
                    name: {
                        "camera_id": cam,
                        "captured_at": f"2012-01-0{day}T10:00:0{i}",
                        "sequence_id": seq,
                    }
                    for i, (name, _, cam, day, _) in enumerate(files)
                }
            }
            batch = c.post(
                "/batches",
                files=[("files", (n, d, "image/jpeg")) for n, d, *_ in files],
                data={"metadata": json.dumps(meta)},
            ).json()
            (event,) = c.get("/events", params={"batch_id": batch["id"]}).json()["events"]
            by_file.update((n, event) for n, *_ in files)
        for e in {e["id"]: e for e in by_file.values()}.values():
            url = f"/events/{e['id']}/reviews"
            body = {"reviewer": "ranger", "outcome": "corrected", "confirmed_label": "cat"}
            if c.post(url, json=body).status_code == 422:  # suggested cat already
                assert c.post(url, json={**body, "outcome": "confirmed"}).status_code == 201
        protocol = tmp_path / "protocol.yaml"
        protocol.write_text(
            yaml.safe_dump(
                {
                    "name": "update3",
                    "deployed_release": "test-predictor-v0",
                    "deployment": {"cameras": ["90", "125"], "reviewer": "ranger"},
                    "protected": {
                        "splits": "images.jsonl.gz",
                        "partitions": ["final_test"],
                        "snapshot_holdouts": ["earlier-holdout.jsonl"],
                    },
                }
            )
        )
        out = build(protocol, "http://testserver", tmp_path / "snaps", client=c, root=tmp_path)

    summary = json.loads((out / "snapshot.json").read_text())
    ids = {name: e["id"] for name, e in by_file.items()}
    reasons = {x["event_id"]: x["reason"] for x in summary["excluded"]["records"]}
    assert reasons == {
        ids["renamed-old-holdout.jpg"]: "earlier_snapshot_holdout_sha256",
        ids["d-early.jpg"]: "train_frame_in_holdout",  # its bytes are in camera 125's holdout
        ids["seq-a.jpg"]: "train_frame_in_holdout",  # one duplicated frame drops the event
        ids["renamed-train-image.jpg"]: "holdout_frame_in_training_partition",
        # Same-workspace re-uploads are stored as duplicates, not ML inputs, but
        # their content still counts for separation (d-early above).
        ids["d-late.jpg"]: "no_usable_frames",
        ids["seq-b-again.jpg"]: "no_usable_frames",
    }
    train = [json.loads(x) for x in (out / "train.jsonl").read_text().splitlines()]
    hold = [json.loads(x) for x in (out / "holdout.jsonl").read_text().splitlines()]
    assert {r["event_id"] for r in train} == {ids["f1.jpg"], ids["g1.jpg"]}
    train_shas = {r["sha256"] for r in train}
    hold_shas = {f["sha256"] for h in hold for f in h["frames"]}
    assert not train_shas & (hold_shas | {sha(dup), sha(seq_dup), sha(old), sha(fit_img)})
    assert sha(fit_img) not in hold_shas and sha(old) not in hold_shas
    assert leakage(out, {}, training={sha(fit_img)}, earlier_holdouts={sha(old)}) == {}


def test_gate_finds_content_overlap_the_builder_should_have_removed(tmp_path: Path) -> None:
    from wildinbox.training.gate import leakage

    snap = _snapshot(tmp_path)
    train = snap / "train.jsonl"
    train.write_text(
        train.read_text()
        + json.dumps({"sha256": "c2", "label": "cat", "event_id": "h1", "camera_id": "cct-90"})
        + "\n"
    )
    assert leakage(snap, {}, training={"d1"}, earlier_holdouts={AA, "e1"}) == {
        "train_frame_in_holdout": 1,
        "event_in_train_and_holdout": 1,
        "holdout_frame_in_training_partition": 1,
        "earlier_snapshot_holdout": 2,
    }


def test_snapshot_authenticates_with_the_token_convention_and_names_auth_failures(
    settings: Settings,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Against a token-protected API: WILDINBOX_TOKEN builds the snapshot; a
    wrong or missing token is a clear authentication error, never a KeyError
    from reading an error body as data."""
    from fastapi.testclient import TestClient

    from wildinbox import cli
    from wildinbox.api.app import create_app
    from wildinbox.api.auth import new_token, token_hash
    from wildinbox.training import snapshot
    from wildinbox.training.snapshot import SnapshotAPIError, api_client, build

    token = new_token()
    secured = settings.model_copy(
        update={"auth": "tokens", "api_tokens": {"alice": token_hash(token)}}
    )
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text(
        yaml.safe_dump(
            {
                "name": "update4",
                "deployed_release": "test-predictor-v0",
                "deployment": {"cameras": ["90"], "reviewer": "alice"},
                "protected": {"partitions": [], "snapshot_holdouts": []},
            }
        )
    )
    app = create_app(secured)

    def client_as(env_token: str | None) -> TestClient:
        """What `api_client` would send, over the in-process app."""
        if env_token is None:
            monkeypatch.delenv("WILDINBOX_TOKEN", raising=False)
        else:
            monkeypatch.setenv("WILDINBOX_TOKEN", env_token)
        c = TestClient(app)
        c.headers.update(api_client("http://testserver").headers)
        return c

    with client_as(token) as c:
        meta = {"files": {"a.jpg": {"camera_id": "cct-90", "captured_at": "2012-01-01T10:00:00"}}}
        batch = c.post(
            "/batches",
            files=[("files", ("a.jpg", jpeg(401), "image/jpeg"))],
            data={"metadata": json.dumps(meta)},
        ).json()
        (event,) = c.get("/events", params={"batch_id": batch["id"]}).json()["events"]
        res = c.post(f"/events/{event['id']}/reviews", json={"outcome": "unresolved"})
        assert res.status_code == 201 and res.json()["reviewer"] == "alice"
        out = build(protocol, "http://testserver", tmp_path / "snaps", client=c, root=tmp_path)
    summary = json.loads((out / "snapshot.json").read_text())
    assert summary["approved_reviewers"] == ["alice"]
    assert summary["provenance"]["events"] == 1

    for env_token, sent in (("not-the-token", "a token sent"), (None, "no token sent")):
        with (
            client_as(env_token) as c,
            pytest.raises(SnapshotAPIError, match=f"401 Unauthorized \\({sent}\\)"),
        ):
            build(protocol, "http://testserver", tmp_path / "snaps", client=c, root=tmp_path)

    with client_as("not-the-token") as c:
        monkeypatch.setattr(snapshot, "api_client", lambda url: c)
        code = cli.main(["snapshot", "build", "--protocol", str(protocol), "--out", str(tmp_path)])
    err = capsys.readouterr().err
    assert code == 1 and "GET /releases: 401" in err and "WILDINBOX_TOKEN" in err


def test_mixed_valid_and_unusable_members_give_a_valid_snapshot(
    settings: Settings,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rejected, corrupt, and failed-inference members stay in the event and in
    provenance with their reasons; only usable frames become ML inputs."""
    from fastapi.testclient import TestClient

    from wildinbox.api.app import create_app
    from wildinbox.inference.serving import PlumbingScorer
    from wildinbox.training.snapshot import build
    from wildinbox.workers import process

    good_a, good_f, good_b, fails_b, fails_c = (jpeg(600 + i) for i in range(5))
    corrupt = jpeg(605)[:200]  # a truncated JPEG: accepted at upload, fails to decode
    failing = {hashlib.sha256(d).hexdigest() for d in (fails_b, fails_c)}

    class PickyScorer(PlumbingScorer):
        def score(self, images: Any, sha256s: Any) -> list[dict[str, float]]:
            if failing & set(sha256s):
                raise RuntimeError("model rejected this input")
            return super().score(images, sha256s)

    monkeypatch.setattr(process, "scorer_for", lambda r, st, d="cpu": PickyScorer(r))
    uploads = [  # (file name, bytes, day, sequence)
        ("a-good.jpg", good_a, 1, "a"),
        ("a-empty.jpg", b"", 1, "a"),
        ("a-corrupt.jpg", corrupt, 1, "a"),
        ("f-good.jpg", good_f, 2, "f"),
        # Same bytes as a-corrupt (stored as its duplicate): must not tie a to f.
        ("f-corrupt.jpg", corrupt, 2, "f"),
        ("b-good.jpg", good_b, 3, "b"),
        ("b-fails.jpg", fails_b, 3, "b"),
        ("c-fails.jpg", fails_c, 4, "c"),  # decodes, never scored: no usable frame
    ]
    meta = {
        "files": {
            name: {
                "camera_id": "cct-90",
                "captured_at": f"2012-01-0{day}T10:00:0{i}",
                "sequence_id": seq,
            }
            for i, (name, _, day, seq) in enumerate(uploads)
        }
    }
    with TestClient(create_app(settings)) as c:
        batch = c.post(
            "/batches",
            files=[("files", (n, d, "image/jpeg")) for n, d, *_ in uploads],
            data={"metadata": json.dumps(meta)},
        ).json()
        events = c.get("/events", params={"batch_id": batch["id"]}).json()["events"]
        by_seq = {}
        for e in events:
            detail = c.get(f"/events/{e['id']}").json()
            by_seq[detail["images"][0]["sequence_id"]] = e["id"]
            url = f"/events/{e['id']}/reviews"
            body = {"reviewer": "ranger", "outcome": "corrected", "confirmed_label": "cat"}
            if c.post(url, json=body).status_code == 422:  # suggested cat already
                assert c.post(url, json={**body, "outcome": "confirmed"}).status_code == 201
        assert set(by_seq) == {"a", "f", "b", "c"}
        protocol = tmp_path / "protocol.yaml"
        protocol.write_text(
            yaml.safe_dump(
                {
                    "name": "update5",
                    "deployed_release": "test-predictor-v0",
                    "deployment": {"cameras": ["90"], "reviewer": "ranger"},
                    "protected": {"partitions": [], "snapshot_holdouts": []},
                }
            )
        )
        out = build(protocol, "http://testserver", tmp_path / "snaps", client=c, root=tmp_path)

    def sha(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    summary = json.loads((out / "snapshot.json").read_text())
    assert summary["excluded"]["by_reason"] == {"no_usable_frames": 1}
    assert summary["withheld_frames"] == {"duplicate": 1, "invalid": 2, "processing_failed": 1}
    train = [json.loads(x) for x in (out / "train.jsonl").read_text().splitlines()]
    hold = {
        h["event_id"]: h for h in map(json.loads, (out / "holdout.jsonl").read_text().splitlines())
    }
    assert [r["sha256"] for r in train] == [sha(good_a)]
    assert [f["sha256"] for f in hold[by_seq["b"]]["frames"]] == [sha(good_b)]
    assert by_seq["c"] not in hold
    assert sorted(p.name for p in (out / "images").iterdir()) == sorted(
        f"{sha(d)}.jpg" for d in (good_a, good_f, good_b)
    )
    labels = {
        r["event_id"]: r for r in map(json.loads, (out / "labels.jsonl").read_text().splitlines())
    }
    members = {m["filename"]: m for m in labels[by_seq["a"]]["members"]}
    assert [members[n]["status"] for n in ("a-good.jpg", "a-empty.jpg", "a-corrupt.jpg")] == [
        "usable",
        "invalid",
        "invalid",
    ]
    assert members["a-good.jpg"]["reason"] is None
    assert members["a-empty.jpg"]["reason"] and members["a-corrupt.jpg"]["reason"]
    assert labels[by_seq["a"]]["frames"] == [sha(good_a)]
    failed = {m["filename"]: m for m in labels[by_seq["b"]]["members"]}["b-fails.jpg"]
    assert failed["status"] == "processing_failed" and "model rejected" in failed["reason"]
    assert labels[by_seq["c"]]["use"] == "excluded:no_usable_frames"
    rows, _, _ = snapshot_rows(out)  # the loader's own byte check passes
    assert [r.source_id for r in rows] == [f"snapshot:{sha(good_a)}"]


def test_the_training_loader_refuses_bytes_the_snapshot_did_not_record(tmp_path: Path) -> None:
    d = _snapshot(tmp_path)
    (d / "images" / f"{AA}.jpg").write_bytes(b"")
    with pytest.raises(SnapshotError, match="does not match its SHA-256"):
        snapshot_rows(d)


def _release_dir(root: Path, name: str, seed: int, temperature: float = 1.5) -> str:
    """A model directory and its policy artifact that register as one release;
    returns the release id."""
    from wildinbox.class_map import ClassMap
    from wildinbox.config import load_config
    from wildinbox.training.gate import _version

    cfg = load_config(REPO_ROOT / "configs/example.yaml")
    d = root / name
    d.mkdir(parents=True)
    weights = f"weights-{seed}".encode()
    (d / "model.pt").write_bytes(weights)
    (d / "meta.json").write_text(
        json.dumps(
            {
                "name": name,
                "classes": cfg.classes,
                "class_map_fingerprint": ClassMap(cfg.classes).fingerprint(),
                "preprocessing": cfg.preprocessing.model_dump(mode="json"),
                "preprocessing_version": cfg.preprocessing.fingerprint(),
            }
        )
    )
    calibration = {
        "method": "temperature",
        "temperature": temperature,
        "weights_digest": hashlib.sha256(weights).hexdigest()[:12],
    }
    calibration["version"] = _version(calibration)
    policy: dict[str, Any] = {
        "policy": "conservative",
        "model": name,
        "classes": cfg.classes,
        "calibration": calibration,
        "unfamiliar": None,
        "released": {
            "empty_threshold": 0.65,
            "species_threshold": None,
            "auto_filter_enabled": False,
            "auto_accept_enabled": False,
            "accept_species": [],
            "policy_version": f"conservative/v1+{seed}",
        },
        "rule": {},
    }
    policy["artifact_version"] = _version(policy)
    (d / "policy.json").write_text(json.dumps(policy))
    return f"{name}@{policy['artifact_version']}"


def test_the_gate_resolves_exactly_the_deployed_release(tmp_path: Path) -> None:
    from wildinbox.training.gate import GateError, resolve_release

    root = tmp_path / "models"
    v1, v2 = _release_dir(root, "model-v1", 1), _release_dir(root, "model-v2", 2)
    got = resolve_release(v2, root)
    assert (got.id, got.model_dir, got.policy_path) == (
        v2,
        root / "model-v2",
        root / "model-v2/policy.json",
    )
    assert got.weights_sha256 == hashlib.sha256(b"weights-2").hexdigest()
    assert resolve_release(v1, root).model_dir == root / "model-v1"

    def refused(release_id: str, match: str, policy: Path | None = None) -> None:
        with pytest.raises(GateError, match=match):
            resolve_release(release_id, root, policy)

    v1_version = v1.split("@")[1]
    refused(f"model-v2@{v1_version}", "is policy version")  # V2's name, V1's policy
    refused(v2, "is policy version", root / "model-v1/policy.json")  # another policy file
    refused("model-v9@abc", "no model at")
    refused("model-v2", "is not <model>@")

    policy_path = root / "model-v2/policy.json"
    original = policy_path.read_text()
    edited = json.loads(original)
    edited["released"]["empty_threshold"] = 0.2  # loosened, version left as it was
    policy_path.write_text(json.dumps(edited))
    refused(v2, "changed after it was versioned")
    policy_path.write_text(original)

    (root / "model-v2/model.pt").write_bytes(b"other weights")  # retrained in place
    refused(v2, "calibration was fitted on different weights")


def test_a_v3_cycle_against_deployed_v2_loads_v2_and_refuses_mismatches_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gzip
    from types import SimpleNamespace

    from wildinbox.evaluation import predictors
    from wildinbox.training import gate
    from wildinbox.training import run as training_run
    from wildinbox.training.gate import GateError

    root = tmp_path / "models"
    _release_dir(root, "model-v1", 1)
    v2 = _release_dir(root, "model-v2", 2)
    _release_dir(root, "model-v3", 3)
    cand = root / "model-v3"
    meta = json.loads((cand / "meta.json").read_text())
    snap = _snapshot(tmp_path)
    meta["trained_on"] = {"snapshot": {"version": "abc123", "path": str(snap)}}
    (cand / "meta.json").write_text(json.dumps(meta))
    splits = tmp_path / "splits"
    splits.mkdir()
    with gzip.open(splits / "images.jsonl.gz", "wt"):
        pass

    def protocol(deployed: str) -> Path:
        p = tmp_path / "protocol.yaml"
        p.write_text(
            yaml.safe_dump({"deployed_release": deployed, "models_root": str(root), "gate": {}})
        )
        return p

    contexts: list[Path] = []

    def fake_context(config: Path, data_dir: Path) -> SimpleNamespace:
        contexts.append(config)
        return SimpleNamespace(split_dir=splits)

    loaded: list[Path] = []

    class Loaded(Exception):
        pass

    def fake_predictor(ctx: Any, model_dir: Path, device: str) -> None:
        loaded.append(model_dir)
        raise Loaded

    monkeypatch.setattr(training_run, "load_context", fake_context)
    monkeypatch.setattr(predictors, "FinetunedPredictor", fake_predictor)

    wrong = f"model-v2@{v2.split('@')[1][::-1]}"
    with pytest.raises(GateError, match="is policy version"):
        gate.run(protocol(wrong), cand, tmp_path / "cfg.yaml", tmp_path / "report")
    assert contexts == [] and loaded == []  # refused before reading any data

    with pytest.raises(Loaded):
        gate.run(protocol(v2), cand, tmp_path / "cfg.yaml", tmp_path / "report")
    assert loaded == [root / "model-v2"]  # the baseline is V2, not the current policy's model
