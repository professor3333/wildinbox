"""A clearly labeled TEST predictor that exercises the pipeline end to end.

Its scores are pseudo-random numbers derived from each file's SHA-256. They
are deterministic (so reruns are comparable) and look like a probability
distribution, but they carry no information about the image. Any metric
computed from them is meaningless and must never be reported as ML performance.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

TEST_RELEASE_ID = "test-predictor-v0"
TEST_NOTICE = (
    "TEST PREDICTOR: scores are pseudo-random values derived from file hashes, used only "
    "to exercise the pipeline. They are not model output and say nothing about ML performance."
)


class PlumbingPredictor:
    is_test = True

    def __init__(self, class_names: list[str]) -> None:
        if not class_names:
            raise ValueError("class_names must not be empty")
        self.class_names = list(class_names)

    def predict(self, image: Image.Image, sha256: str) -> dict[str, float]:
        if image.width <= 0 or image.height <= 0:
            raise ValueError("image has no pixels")
        rng = np.random.default_rng(int(sha256[:16], 16))
        logits = rng.normal(0.0, 1.5, len(self.class_names))
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        return {name: float(p) for name, p in zip(self.class_names, probs, strict=True)}
