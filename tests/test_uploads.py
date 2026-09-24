"""Unit tests that need no database: upload checks, predictor, policy, storage."""

from __future__ import annotations

import io
import uuid
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from wildinbox.api.uploads import (
    BatchMetadata,
    UploadError,
    check_file,
    parse_metadata,
    request_fingerprint,
    sniff,
)
from wildinbox.inference.plumbing import PlumbingPredictor
from wildinbox.policy.review_all import POLICY_VERSION, decide
from wildinbox.schemas import Disposition, ReviewReason
from wildinbox.settings import Settings
from wildinbox.storage.objects import LocalStore, ObjectNotFoundError, original_key
from wildinbox.workers.process import inspect

SETTINGS = Settings(max_file_bytes=1000)


def jpeg(seed: int = 0, exif_time: str | None = None, size: tuple[int, int] = (64, 48)) -> bytes:
    arr = np.random.default_rng(seed).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    if exif_time:
        exif = Image.Exif()
        exif.get_ifd(0x8769)[0x9003] = exif_time
        img.save(buf, "JPEG", exif=exif)
    else:
        img.save(buf, "JPEG")
    return buf.getvalue()


@pytest.mark.parametrize(
    ("data", "ctype", "described"),
    [
        (b"\xff\xd8\xff\xe0rest", "image/jpeg", "image/jpeg"),
        (b"\x89PNG\r\n\x1a\nrest", "image/png", "image/png"),
        (b"GIF89a...", None, "GIF"),
        (b"\x00\x00\x00\x18ftypheic", None, "HEIC/MP4 container"),
        (b"hello", None, "unrecognised format"),
    ],
)
def test_sniff_uses_content_not_extension(data: bytes, ctype: str | None, described: str) -> None:
    assert sniff(data) == (ctype, described)


def test_check_file_errors_are_specific() -> None:
    meta = BatchMetadata().for_file("x")
    assert check_file(0, "a.jpg", b"", False, SETTINGS, meta).error_code == "empty_file"
    big = check_file(0, "a.jpg", b"\xff\xd8\xff" + b"0" * 2000, True, SETTINGS, meta)
    assert big.error_code == "file_too_large" and big.data is None
    gif = check_file(0, "photo.jpg", b"GIF89a", False, SETTINGS, meta)
    assert gif.error_code == "unsupported_type" and "GIF" in (gif.error or "")
    ok = check_file(0, "a.jpg", jpeg(), False, SETTINGS, meta)
    assert ok.accepted and ok.content_type == "image/jpeg" and ok.sha256


def test_metadata_defaults_and_validation() -> None:
    meta = parse_metadata('{"camera_id": "north", "files": {"a.jpg": {"camera_id": "south"}}}')
    assert meta.for_file("a.jpg").camera_id == "south"
    assert meta.for_file("b.jpg").camera_id == "north"
    with pytest.raises(UploadError) as e:
        parse_metadata('{"camera": "typo"}')
    assert e.value.status == 422
    with pytest.raises(UploadError):
        parse_metadata("not json")


def test_request_fingerprint_is_stable_and_sensitive() -> None:
    meta = BatchMetadata()
    files = [check_file(0, "a.jpg", jpeg(1), False, SETTINGS, meta.for_file("a.jpg"))]
    same = [check_file(0, "a.jpg", jpeg(1), False, SETTINGS, meta.for_file("a.jpg"))]
    other = [check_file(0, "a.jpg", jpeg(2), False, SETTINGS, meta.for_file("a.jpg"))]
    assert request_fingerprint(files, meta) == request_fingerprint(same, meta)
    assert request_fingerprint(files, meta) != request_fingerprint(other, meta)
    assert request_fingerprint(files, meta) != request_fingerprint(
        files, BatchMetadata(camera_id="x")
    )


def test_plumbing_predictor_is_labeled_and_deterministic() -> None:
    p = PlumbingPredictor(["empty", "raccoon", "coyote"])
    img = Image.new("RGB", (8, 8))
    a, b = p.predict(img, "ab" * 32), p.predict(img, "ab" * 32)
    assert p.is_test and a == b
    assert abs(sum(a.values()) - 1) < 1e-9 and set(a) == {"empty", "raccoon", "coyote"}
    assert a != p.predict(img, "cd" * 32)


def test_review_all_policy_never_automates() -> None:
    d = decide(
        uuid.uuid4(),
        [uuid.uuid4(), uuid.uuid4()],
        [{"empty": 0.9, "raccoon": 0.1}, {"empty": 0.99, "raccoon": 0.01}],
        model_version="m",
        preprocessing_version="p",
    )
    assert d.disposition is Disposition.NEEDS_REVIEW
    assert d.reasons == [ReviewReason.AUTOMATION_DISABLED]
    assert d.suggested_label == "empty" and d.policy_version == POLICY_VERSION
    assert d.confidence == pytest.approx(0.945)


def test_inspect_reads_exif_and_rejects_corrupt() -> None:
    found = inspect(jpeg(exif_time="2024:05:01 21:03:00"))
    assert found.error is None and found.captured_at is not None
    assert found.captured_at.isoformat() == "2024-05-01T21:03:00"
    bad = inspect(b"\xff\xd8\xff" + b"garbage" * 10)
    assert bad.error and bad.error.startswith("unreadable image") and "0x" not in bad.error


def test_local_store_roundtrip_and_key_safety(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    key = original_key("ab" * 32)
    store.put(key, b"data", "image/jpeg")
    assert store.exists(key) and store.get(key) == b"data"
    with pytest.raises(ObjectNotFoundError):
        store.get(original_key("cd" * 32))
    with pytest.raises(ValueError):
        store.put("../escape", b"x", "text/plain")
