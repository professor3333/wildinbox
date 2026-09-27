"""Upload validation that runs inside the request: limits, file types, hashes.

Cheap checks happen here so the caller gets an immediate answer; full image
decoding happens in the worker. Oversized *batches* are rejected whole.
Oversized, empty, or unsupported *files* become individual error records and
the rest of the batch continues.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.types import Message, Receive

from wildinbox.datasets.grouping import DEFAULT_GAP_SECONDS, MAX_GAP_SECONDS, batch_grouping
from wildinbox.settings import Settings

SIGNATURES: list[tuple[bytes, str]] = [
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
]
# Recognised but unsupported, so the error can say what the file actually is.
KNOWN_OTHER: list[tuple[bytes, str]] = [
    (b"GIF8", "GIF"),
    (b"RIFF", "RIFF (WebP/AVI/WAV)"),
    (b"%PDF", "PDF"),
    (b"II*\x00", "TIFF"),
    (b"MM\x00*", "TIFF"),
    (b"BM", "BMP"),
]


class UploadError(Exception):
    """The whole request is rejected (nothing is stored)."""

    def __init__(self, status: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status, self.code, self.detail = status, code, detail


def byte_budget(receive: Receive, limit: int) -> Receive:
    """Wraps an ASGI `receive` so the request body is rejected as soon as more
    than `limit` bytes have arrived, whatever Content-Length says (or whether
    it is sent at all). The multipart parser closes its temporary files when
    this raises."""
    received = 0

    async def limited() -> Message:
        nonlocal received
        message = await receive()
        if message["type"] == "http.request":
            received += len(message.get("body", b""))
            if received > limit:
                raise UploadError(413, "batch_too_large", f"request body exceeds {limit} bytes")
        return message

    return limited


class FileMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    camera_id: str | None = Field(default=None, max_length=200)
    captured_at: datetime | None = None
    sequence_id: str | None = Field(default=None, max_length=200)


class BatchMetadata(BaseModel):
    """Optional JSON sent as the `metadata` form field."""

    model_config = ConfigDict(extra="forbid")
    camera_id: str | None = Field(
        default=None, max_length=200, description="Default for all files."
    )
    files: dict[str, FileMetadata] = Field(default_factory=dict, description="Per filename.")
    gap_seconds: float | None = Field(
        default=None,
        ge=0,
        le=MAX_GAP_SECONDS,
        allow_inf_nan=False,
        description=(
            "Grouping interval G: photos from one camera without a sequence id stay in one "
            f"event while consecutive capture times are at most G seconds apart. Default "
            f"{DEFAULT_GAP_SECONDS:g}. Sequence ids, when supplied, take precedence."
        ),
    )

    def for_file(self, filename: str) -> FileMetadata:
        own = self.files.get(filename, FileMetadata())
        return own if own.camera_id else own.model_copy(update={"camera_id": self.camera_id})


def parse_metadata(raw: str | None) -> BatchMetadata:
    if not raw:
        return BatchMetadata()
    try:
        return BatchMetadata.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as e:
        raise UploadError(422, "invalid_metadata", f"metadata is not valid: {e}") from None


def sniff(data: bytes) -> tuple[str | None, str]:
    """(supported content type or None, human description)."""
    for sig, ctype in SIGNATURES:
        if data.startswith(sig):
            return ctype, ctype
    for sig, name in KNOWN_OTHER:
        if data.startswith(sig):
            return None, name
    if data[4:8] == b"ftyp":
        return None, "HEIC/MP4 container"
    return None, "unrecognised format"


@dataclass
class UploadedFile:
    position: int
    filename: str
    size: int
    data: bytes | None  # None when rejected before reading completely
    sha256: str | None = None
    content_type: str | None = None
    error_code: str | None = None
    error: str | None = None
    metadata: FileMetadata = field(default_factory=FileMetadata)

    @property
    def accepted(self) -> bool:
        return self.error is None


def check_file(
    position: int,
    filename: str,
    data: bytes,
    oversize: bool,
    settings: Settings,
    metadata: FileMetadata,
) -> UploadedFile:
    f = UploadedFile(
        position=position, filename=filename, size=len(data), data=None, metadata=metadata
    )
    if oversize:
        f.error_code = "file_too_large"
        f.error = (
            f"file is larger than the {settings.max_file_bytes // (1024 * 1024)} MiB per-file limit"
        )
        return f
    if not data:
        f.error_code, f.error = "empty_file", "file is empty (0 bytes)"
        return f
    ctype, described = sniff(data)
    if ctype is None:
        f.error_code = "unsupported_type"
        f.error = f"unsupported file type ({described}); only JPEG and PNG are accepted"
        return f
    f.data, f.content_type = data, ctype
    f.sha256 = hashlib.sha256(data).hexdigest()
    return f


def request_fingerprint(files: list[UploadedFile], metadata: BatchMetadata) -> str:
    """Identifies an identical resubmission: same files, names, order, and metadata,
    including the grouping interval. The interval joins the fingerprint only when
    supplied, so requests without one keep the fingerprints they always had."""
    payload: dict[str, Any] = {
        "files": [[f.filename, f.sha256, f.size, f.error_code] for f in files],
        "metadata": metadata.model_dump(mode="json", exclude={"gap_seconds"}),
    }
    if metadata.gap_seconds is not None:
        payload["gap_seconds"] = metadata.gap_seconds
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def manifest(
    files: list[UploadedFile], metadata: BatchMetadata, fingerprint: str, settings: Settings
) -> dict[str, Any]:
    return {
        "fingerprint": fingerprint,
        "files": [
            {
                "position": f.position,
                "filename": f.filename,
                "size": f.size,
                "sha256": f.sha256,
                "content_type": f.content_type,
                "error_code": f.error_code,
            }
            for f in files
        ],
        "metadata": metadata.model_dump(mode="json"),
        "grouping": batch_grouping(metadata.gap_seconds),
        "limits": {
            "max_files_per_batch": settings.max_files_per_batch,
            "max_file_bytes": settings.max_file_bytes,
            "max_batch_bytes": settings.max_batch_bytes,
        },
    }
