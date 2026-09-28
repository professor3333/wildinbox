"""Protocol provenance: the gate report claims only what the record shows."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from wildinbox.training.gate_report import write_report
from wildinbox.training.provenance import protocol_provenance

PROTOCOL = Path("configs/experiments/cycle.yaml")
UNCONDITIONAL = "committed before any review was collected or candidate trained"


def _git(*args: str, at: str | None = None) -> str:
    env = dict(os.environ)
    if at is not None:
        env |= {"GIT_AUTHOR_DATE": at, "GIT_COMMITTER_DATE": at}
    ident = ["-c", "user.name=t", "-c", "user.email=t@example.com"]
    return subprocess.check_output(["git", *ident, *args], text=True, env=env).strip()


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    _git("init", "-q")
    (tmp_path / "README").write_text("start\n")
    _git("add", "README")
    _git("commit", "-q", "-m", "start", at="2026-09-01T00:00:00+00:00")
    return tmp_path


def _commit_protocol(text: str, at: str) -> str:
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(text)
    _git("add", str(PROTOCOL))
    _git("commit", "-q", "-m", "protocol", at=at)
    return _git("rev-parse", "HEAD")


def _snapshot(root: Path, reviewed_at: list[str | None]) -> Path:
    d = root / "snapshot"
    d.mkdir()
    with (d / "labels.jsonl").open("w") as f:
        for t in reviewed_at:
            f.write(json.dumps({"reviewed_at": t}) + "\n")
    return d


def test_protocol_committed_before_reviews_and_clean_training(repo: Path) -> None:
    commit = _commit_protocol("gate: 1\n", "2026-09-25T06:00:00+00:00")
    snap = _snapshot(repo, ["2026-09-25T10:19:19+00:00", "2026-09-25T10:21:00+00:00"])
    prov = protocol_provenance(PROTOCOL, snap, {"commit": commit, "dirty": False})
    assert prov["first_committed"]["commit"] == commit
    assert prov["earliest_review_at"] == "2026-09-25T10:19:19+00:00"
    assert prov["committed_before_reviews"] is True
    assert prov["committed_before_training"] is True


def test_protocol_written_after_reviews_and_training_from_uncommitted_changes(
    repo: Path,
) -> None:
    """The rehearsal's chronology: reviews first, config uncommitted at training."""
    training = _git("rev-parse", "HEAD")
    _commit_protocol("gate: 1\n", "2026-09-27T09:46:00+00:00")
    snap = _snapshot(repo, ["2026-09-25T10:19:19+00:00"])
    prov = protocol_provenance(PROTOCOL, snap, {"commit": training, "dirty": True})
    assert prov["committed_before_reviews"] is False
    assert prov["committed_before_training"] is None


def test_missing_evidence_is_not_established_and_edits_do_not_count(repo: Path) -> None:
    before = _git("rev-parse", "HEAD")  # clean, but the protocol did not exist yet
    _commit_protocol("gate: 1\n", "2026-09-25T06:00:00+00:00")
    later = _commit_protocol("gate: 2\n", "2026-09-26T06:00:00+00:00")
    snap = _snapshot(repo, ["2026-09-25T10:19:19+00:00", None])  # one review time missing
    prov = protocol_provenance(PROTOCOL, snap, {"commit": before, "dirty": False})
    assert prov["first_committed"]["commit"] == later  # this content, not the file
    assert prov["earliest_review_at"] is None
    assert prov["committed_before_reviews"] is None
    assert prov["committed_before_training"] is False


def _gate_record() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    out: dict[str, Any] = json.loads(
        (root / "reports/update/finetune-e3-rehearsal/metrics.json").read_text()
    )
    return out


def _render(tmp_path: Path, out: dict[str, Any]) -> str:
    return write_report(tmp_path, out).read_text()


def test_report_claims_the_order_only_when_both_are_shown(tmp_path: Path) -> None:
    out = _gate_record()
    out["protocol_provenance"].update(
        committed_before_reviews=True,
        committed_before_training=True,
        training_code={"commit": "a" * 40, "dirty": False},
    )
    md = _render(tmp_path, out)
    assert (
        "committed before the snapshot's first review; committed before the candidate "
        "was trained ([evidence](#protocol-provenance))" in md
    )
    assert "| Committed before the reviews | yes |" in md


def test_report_for_the_rehearsal_says_what_the_record_shows(tmp_path: Path) -> None:
    md = _render(tmp_path, _gate_record())
    assert UNCONDITIONAL not in md
    assert "committed after the snapshot's reviews were made" in md
    assert "not shown to predate training" in md
    assert "| Candidate trained at | `1c5de83`, with uncommitted changes |" in md
    assert "| Committed before the reviews | **no** |" in md


def test_report_without_provenance_is_neutral(tmp_path: Path) -> None:
    out = _gate_record()
    del out["protocol_provenance"]
    md = _render(tmp_path, out)
    assert UNCONDITIONAL not in md and "## Protocol provenance" not in md
    assert "was not recorded" in md


def test_the_same_protocol_gets_the_same_provenance_however_it_is_named(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = _commit_protocol("gate: 1\n", "2026-09-25T06:00:00+00:00")
    snap = _snapshot(repo, ["2026-09-25T10:19:19+00:00"])
    training = {"commit": commit, "dirty": False}
    relative = protocol_provenance(PROTOCOL, snap, training)
    assert relative["first_committed"]["commit"] == commit
    assert relative["committed_before_training"] is True
    assert protocol_provenance(repo / PROTOCOL, snap, training) == relative
    assert protocol_provenance((repo / PROTOCOL).resolve(), snap, training) == relative
    # From a subdirectory: `git log` reads paths from there, `git show` from the root.
    monkeypatch.chdir(repo / "configs")
    assert protocol_provenance(Path("experiments/cycle.yaml"), snap, training) == relative


def test_a_protocol_outside_any_repository_has_no_provenance(tmp_path: Path) -> None:
    outside = tmp_path / "protocol.yaml"
    outside.write_text("gate: 1\n")
    snap = _snapshot(tmp_path, ["2026-09-25T10:19:19+00:00"])
    prov = protocol_provenance(outside, snap, {"commit": "abc", "dirty": False})
    assert prov["first_committed"] is None
    assert prov["committed_before_reviews"] is None
    assert prov["committed_before_training"] is None
