"""Was an update protocol fixed before its reviews and its candidate?

The gate report states this only as far as recorded evidence shows it:

- **Before the reviews:** the first commit holding the protocol's exact
  content is older than the earliest `reviewed_at` in the snapshot's labels.
- **Before training:** the candidate was trained from a clean tree (`src` and
  `configs`) at a commit holding the protocol's exact content.

Each answer is True, False, or None when the evidence is missing (git history
unavailable, a snapshot built before review times were recorded, or training
from uncommitted changes, which may have included the protocol).

The protocol is looked up by its path inside its repository, however it was
named (absolute, relative, from any directory), so the same file always gets
the same provenance.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _git(root: Path, *args: str) -> bytes | None:
    try:
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None


def in_repository(path: Path) -> tuple[Path, str] | None:
    """(repository root, `path` relative to it in POSIX form), or None outside
    a git repository. `git log -- <path>` resolves a path against the working
    directory but `git show <commit>:<path>` against the root, so every lookup
    uses this one root-relative spelling."""
    real = path.resolve()
    top = _git(real.parent, "rev-parse", "--show-toplevel")
    if top is None:
        return None
    root = Path(top.decode().strip()).resolve()
    try:
        return root, real.relative_to(root).as_posix()
    except ValueError:
        return None


def _content_at(root: Path, commit: str, rel: str) -> str | None:
    """sha256 of the file at `rel` in `commit`, None if it did not exist there."""
    blob = _git(root, "show", f"{commit}:{rel}")
    return None if blob is None else hashlib.sha256(blob).hexdigest()


def first_commit_with(path: Path, sha256: str) -> dict[str, str] | None:
    """The oldest commit whose copy of `path` has exactly this content."""
    located = in_repository(path)
    if located is None:
        return None
    root, rel = located
    log = _git(root, "log", "--reverse", "--format=%H %cI", "--", rel)
    for line in (log or b"").decode().splitlines():
        commit, committed_at = line.split(" ", 1)
        if _content_at(root, commit, rel) == sha256:
            return {"commit": commit, "committed_at": committed_at}
    return None


def earliest_review(snapshot_dir: Path) -> str | None:
    """The earliest `reviewed_at` among the snapshot's labels; None unless every
    label records one (snapshots before snapshot/v2 did not)."""
    labels = snapshot_dir / "labels.jsonl"
    if not labels.exists():
        return None
    times: list[str | None] = [
        json.loads(line).get("reviewed_at") for line in labels.open() if line.strip()
    ]
    known = [t for t in times if t is not None]
    if not known or len(known) < len(times):
        return None
    return min(known, key=_parse)


def _parse(ts: str) -> datetime:
    t = datetime.fromisoformat(ts)
    return t if t.tzinfo else t.replace(tzinfo=UTC)  # the API records UTC


def protocol_provenance(
    protocol_path: Path, snapshot_dir: Path, training_code: dict[str, Any]
) -> dict[str, Any]:
    sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    first = first_commit_with(protocol_path, sha)
    review = earliest_review(snapshot_dir)
    before_reviews = _parse(first["committed_at"]) < _parse(review) if first and review else None
    commit, dirty = training_code.get("commit"), training_code.get("dirty")
    located = in_repository(protocol_path)
    if commit is None or dirty is None or dirty or located is None:
        before_training = None
    else:
        before_training = _content_at(located[0], commit, located[1]) == sha
    return {
        "protocol_sha256": sha,
        "first_committed": first,
        "earliest_review_at": review,
        "training_code": {"commit": commit, "dirty": dirty},
        "committed_before_reviews": before_reviews,
        "committed_before_training": before_training,
    }
