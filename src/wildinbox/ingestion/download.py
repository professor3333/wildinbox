"""Resumable, checksum-verified downloads and interruption-safe extraction.

A download is written to `<name>.part` and only renamed to its final name once
its size and MD5 match the pinned values, so a partial or corrupt archive is
never mistaken for a complete one.
"""

from __future__ import annotations

import hashlib
import http.client
import logging
import os
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath

from wildinbox.ingestion.sources import Archive

log = logging.getLogger(__name__)

_CHUNK = 1 << 20


class DownloadError(RuntimeError):
    pass


def md5_of(path: Path) -> str:
    h = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _fetch_once(archive: Archive, part: Path, timeout: float) -> None:
    """Append the remaining bytes to `part`. Raises on any network error."""
    offset = part.stat().st_size if part.exists() else 0
    if offset > archive.size:
        log.warning("%s is larger than expected; restarting", part.name)
        part.unlink()
        offset = 0
    if offset == archive.size:
        return
    req = urllib.request.Request(archive.url)
    if offset:
        req.add_header("Range", f"bytes={offset}-")
        log.info("resuming %s from byte %d of %d", archive.filename, offset, archive.size)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if offset and resp.status != 206:
            log.warning("server ignored Range for %s; restarting from 0", archive.filename)
            offset = 0
        mode = "ab" if offset else "wb"
        with part.open(mode) as f:
            while chunk := resp.read(_CHUNK):
                f.write(chunk)
    got = part.stat().st_size
    if got < archive.size:
        raise DownloadError(f"{archive.filename}: connection ended at {got}/{archive.size} bytes")


def download(
    archive: Archive,
    dest_dir: Path,
    *,
    retries: int = 5,
    backoff: float = 2.0,
    timeout: float = 60.0,
) -> Path:
    """Download `archive` into `dest_dir`, resuming any partial file.

    Returns the final path. Idempotent: an already verified file is not fetched again.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    final = dest_dir / archive.filename
    part = dest_dir / f"{archive.filename}.part"
    if final.exists():
        if final.stat().st_size == archive.size:
            return final
        log.warning("%s has the wrong size; downloading again", final.name)
        final.unlink()

    for attempt in range(1, retries + 1):
        try:
            _fetch_once(archive, part, timeout)
            break
        except (OSError, http.client.HTTPException, urllib.error.URLError, DownloadError) as e:
            done = part.stat().st_size if part.exists() else 0
            log.warning(
                "attempt %d/%d for %s failed at %d bytes: %s",
                attempt,
                retries,
                archive.filename,
                done,
                e,
            )
            if attempt == retries:
                raise DownloadError(
                    f"{archive.filename}: gave up after {retries} attempts; "
                    f"{done} bytes kept in {part.name}; re-run to resume"
                ) from e
            time.sleep(backoff * attempt)

    actual = md5_of(part)
    if actual != archive.md5:
        part.unlink()
        raise DownloadError(
            f"{archive.filename}: MD5 mismatch (expected {archive.md5}, got {actual}); "
            "partial file deleted"
        )
    os.replace(part, final)
    return final


def _safe_member_path(name: str) -> PurePosixPath:
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts:
        raise DownloadError(f"refusing unsafe archive path: {name!r}")
    return p


def extract(archive_path: Path, dest_dir: Path) -> int:
    """Extract a .tar.gz, skipping files already fully extracted.

    Each file is written to a temporary name and renamed when complete, so an
    interrupted extraction never leaves a truncated file under a real name.
    Returns the number of files written this call.
    """
    marker = dest_dir / f".extracted-{archive_path.name}"
    if marker.exists() and marker.read_text().strip() == str(archive_path.stat().st_size):
        return 0
    dest_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with tarfile.open(archive_path, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            target = dest_dir / _safe_member_path(member.name)
            if target.exists() and target.stat().st_size == member.size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            if src is None:
                continue
            tmp = target.with_name(target.name + ".tmp")
            with tmp.open("wb") as out:
                while chunk := src.read(_CHUNK):
                    out.write(chunk)
            os.replace(tmp, target)
            written += 1
    marker.write_text(str(archive_path.stat().st_size))
    return written
