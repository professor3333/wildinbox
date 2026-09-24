"""End-to-end acquisition: download -> extract -> ingest -> lock -> report."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from wildinbox.ingestion.download import download, extract
from wildinbox.ingestion.inventory import IngestResult, Paths
from wildinbox.ingestion.sources import SourceConfig

log = logging.getLogger(__name__)


class LockMismatchError(RuntimeError):
    pass


def fetch(source: SourceConfig, paths: Paths, *, images: bool = True) -> None:
    ann = download(source.annotations_archive, paths.downloads)
    extract(ann, paths.annotations)
    if images:
        img = download(source.images_archive, paths.downloads)
        log.info("extracting %s (skips files already extracted)", img.name)
        extract(img, paths.images)


def lock_payload(source: SourceConfig, result: IngestResult) -> dict[str, Any]:
    return {
        "manifest_version": result.version,
        "source": source.name,
        "archives": {
            a.filename: {"md5": a.md5, "size": a.size}
            for a in (source.images_archive, source.annotations_archive)
        },
        "counts": result.counts,
    }


def check_or_write_lock(lock_path: Path, payload: dict[str, Any], *, update: bool) -> str:
    """Compare against the committed lock. Returns 'created', 'unchanged', or 'updated'."""
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if not lock_path.exists():
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(text)
        return "created"
    pinned = json.loads(lock_path.read_text())
    if pinned == payload:
        return "unchanged"
    if not update:
        raise LockMismatchError(
            f"inventory {payload['manifest_version']} differs from pinned "
            f"{pinned.get('manifest_version')} in {lock_path}. Inspect the change, then "
            "re-run with --update-lock to accept it."
        )
    lock_path.write_text(text)
    return "updated"
