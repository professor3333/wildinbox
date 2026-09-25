"""Unfamiliar-input score: how far an image's features are from the training
images. Softmax confidence alone does not show that an input belongs to a
supported class; distance from the training data is a separate signal."""

from __future__ import annotations

import numpy as np

CHUNK = 1024


def normalize(x: np.ndarray) -> np.ndarray:
    out: np.ndarray = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return out.astype(np.float32)


def knn_cosine_distance(query: np.ndarray, reference: np.ndarray, k: int) -> np.ndarray:
    """Mean cosine distance from each query row to its k nearest reference rows.
    Both inputs must already be L2-normalised."""
    if k < 1 or k > len(reference):
        raise ValueError(f"k must be in [1, {len(reference)}], got {k}")
    out = np.empty(len(query), dtype=np.float32)
    for start in range(0, len(query), CHUNK):
        sims = query[start : start + CHUNK] @ reference.T
        top = np.partition(sims, -k, axis=1)[:, -k:]
        out[start : start + CHUNK] = 1.0 - top.mean(axis=1)
    return out
