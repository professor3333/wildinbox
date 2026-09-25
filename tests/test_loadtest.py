"""The load test's file variants: new bytes, identical pixels."""

from __future__ import annotations

import hashlib
import importlib.util
import io

import numpy as np
from PIL import Image

from .conftest import REPO_ROOT
from .test_uploads import jpeg

_spec = importlib.util.spec_from_file_location("loadtest", REPO_ROOT / "scripts/loadtest.py")
assert _spec and _spec.loader
loadtest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(loadtest)


def pixels(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def test_variant_changes_the_hash_but_not_the_image() -> None:
    original = jpeg(3, exif_time="2024:01:02 03:04:05")
    a, b = loadtest.variant(original, "run1-a"), loadtest.variant(original, "run1-b")
    assert len({hashlib.sha256(x).hexdigest() for x in (original, a, b)}) == 3
    assert np.array_equal(pixels(original), pixels(a))
    assert np.array_equal(pixels(original), pixels(b))
    exif = Image.open(io.BytesIO(a)).getexif().get_ifd(0x8769)
    assert exif[0x9003] == "2024:01:02 03:04:05"  # capture time survives


def test_memory_units() -> None:
    assert loadtest.MemorySampler._bytes("1.5GiB") == 1.5 * 2**30
    assert loadtest.MemorySampler._bytes("512MiB") == 512 * 2**20
    assert loadtest.MemorySampler._bytes("20kB") == 20e3
