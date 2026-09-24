"""Per-file checks: checksum, full decode, dimensions, near-duplicate signature.

Runs in worker processes, so it only takes and returns plain data.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

# Bump when check_file's outputs change so cached results are recomputed.
CHECK_VERSION = 4

# Camera traps stamp date/temperature bars on the top and bottom edges; they
# are identical across cameras and would dominate a whole-frame comparison.
EDGE_CROP = 0.07
SIGNATURE_SIZE = (32, 24)
SIGNATURE_BLUR = 2.0
# Below this grayscale std the thumbnail is nearly uniform (e.g. a black
# frame) and its signature is not used for near-duplicate search.
LOW_INFORMATION_STD = 2.0


@dataclass(frozen=True)
class FileCheck:
    path: str
    size: int
    mtime_ns: int
    sha256: str | None = None
    width: int | None = None
    height: int | None = None
    signature: bytes | None = None
    low_information: bool = False
    error: str | None = None
    check_version: int = CHECK_VERSION


def signature(img: Image.Image) -> tuple[bytes, float]:
    """High-pass grayscale thumbnail used to find near-duplicates, as int8 bytes.

    Subtracting a blurred copy removes illumination falloff (strong in
    night-time infrared frames), so similarity reflects scene content. Returns
    the signature and the thumbnail's grayscale std.
    """
    crop = int(img.height * EDGE_CROP)
    body = img.crop((0, crop, img.width, img.height - crop)).convert("L")
    small = body.resize(SIGNATURE_SIZE, Image.Resampling.BOX)
    arr = np.asarray(small, dtype=np.float32)
    low = np.asarray(small.filter(ImageFilter.GaussianBlur(SIGNATURE_BLUR)), dtype=np.float32)
    high = (arr - low).ravel()
    high -= high.mean()
    peak = float(np.abs(high).max()) or 1.0
    return (np.round(high / peak * 127).astype(np.int8).tobytes(), float(arr.std()))


def unpack_signatures(blobs: list[bytes]) -> np.ndarray:
    """Stack int8 signatures into unit-norm float32 rows (cosine similarity = dot)."""
    mat = np.frombuffer(b"".join(blobs), dtype=np.int8).reshape(len(blobs), -1)
    out = mat.astype(np.float32)
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-6
    return out


def check_file(path: str) -> FileCheck:
    p = Path(path)
    st = p.stat()
    data = p.read_bytes()
    base = FileCheck(
        path=path,
        size=st.st_size,
        mtime_ns=st.st_mtime_ns,
        sha256=hashlib.sha256(data).hexdigest(),
    )
    if not data:
        return replace(base, error="empty file (0 bytes)")
    try:
        with Image.open(BytesIO(data)) as img:
            img.load()  # full decode: catches truncated and corrupt files
            if img.format not in ("JPEG", "PNG"):
                raise ValueError(f"unsupported format {img.format}")
            sig, std = signature(img)
            return replace(
                base,
                width=img.width,
                height=img.height,
                signature=sig,
                low_information=std < LOW_INFORMATION_STD,
            )
    except Exception as e:
        # Drop object addresses ("<... at 0x10ac36d90>") so errors, and therefore
        # the manifest version, are identical across runs.
        message = re.sub(r"<[^<>]* at 0x[0-9a-f]+>", "<file>", str(e))
        return replace(base, error=f"{type(e).__name__}: {message}")
