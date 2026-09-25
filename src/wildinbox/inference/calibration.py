"""Score calibration shared by evaluation, replay, and serving."""

from __future__ import annotations

import numpy as np

P_FLOOR = 1e-12  # cached scores are probabilities; log(0) would be -inf


def log_probs(probs: np.ndarray) -> np.ndarray:
    # softmax(log p / T) == softmax(z / T) for the logits z behind p, because
    # log p differs from z by a per-row constant.
    out: np.ndarray = np.log(np.clip(probs, P_FLOOR, 1.0))
    return out


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    z = log_probs(probs) / temperature
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    out: np.ndarray = e / e.sum(axis=1, keepdims=True)
    return out
