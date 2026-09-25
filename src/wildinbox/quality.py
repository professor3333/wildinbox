"""Cheap per-image quality statistics, shared by evaluation slices and the
monitoring of deployments (night share, blur, brightness per camera)."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

# Infrared night frames are near-grayscale: the channels barely differ.
NIGHT_CHANNEL_DIFF = 2.0
_STATS_WIDTH = 256


def image_stats(img: Image.Image) -> tuple[bool, float]:
    """(looks like an infrared night frame, variance-of-Laplacian sharpness)."""
    small = img.resize((_STATS_WIDTH, max(1, round(img.height * _STATS_WIDTH / img.width))))
    arr = np.asarray(small, dtype=np.float32)
    channel_diff = (
        float(np.abs(arr[..., 0] - arr[..., 1]).mean() + np.abs(arr[..., 1] - arr[..., 2]).mean())
        / 2
    )
    gray = arr.mean(axis=2)
    lap = (
        -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    )
    return channel_diff < NIGHT_CHANNEL_DIFF, float(lap.var())


def quality(img: Image.Image) -> dict[str, Any]:
    """What a worker records for each scored image (RGB input)."""
    night, blur = image_stats(img)
    gray = np.asarray(img.convert("L").resize((_STATS_WIDTH, _STATS_WIDTH)), dtype=np.float32)
    return {"night": night, "blur": round(blur, 1), "brightness": round(float(gray.mean()), 1)}
