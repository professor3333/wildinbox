"""Scoring images with a release inside a worker.

A worker loads each release once (weights verified against their SHA-256) and
keeps it for the life of the process. Images go through the same `load_image`
and `preprocess` used in training and evaluation, so the model sees identical
tensors in every stage.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from io import BytesIO
from typing import TYPE_CHECKING, Protocol

import numpy as np

from wildinbox.config import PreprocessingConfig
from wildinbox.inference.calibration import apply_temperature
from wildinbox.inference.plumbing import PlumbingPredictor
from wildinbox.preprocessing import load_image, preprocess
from wildinbox.storage.models import ModelRelease
from wildinbox.storage.objects import ObjectStore

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)


class Scorer(Protocol):
    release_id: str
    class_names: list[str]

    def score(self, images: Sequence[bytes], sha256s: Sequence[str]) -> list[dict[str, float]]:
        """Raw class probabilities for each image (original file bytes)."""
        ...


class PlumbingScorer:
    """Wraps the pseudo-random TEST predictor (plumbing only, never ML output)."""

    def __init__(self, release: ModelRelease) -> None:
        self.release_id = release.id
        self.class_names = list(release.class_names)
        self._predictor = PlumbingPredictor(self.class_names)

    def score(self, images: Sequence[bytes], sha256s: Sequence[str]) -> list[dict[str, float]]:
        return [
            self._predictor.predict(load_image(data), sha)
            for data, sha in zip(images, sha256s, strict=True)
        ]


class ReleaseScorer:
    def __init__(self, release: ModelRelease, store: ObjectStore, device: str = "cpu") -> None:
        import torch

        from wildinbox.inference.architecture import build_model

        if not (release.weights_key and release.weights_sha256):
            raise ValueError(f"release {release.id} has no weights")
        data = store.get(release.weights_key)
        if hashlib.sha256(data).hexdigest() != release.weights_sha256:
            raise ValueError(f"weights for release {release.id} fail their SHA-256 check")
        self.preprocessing = PreprocessingConfig.model_validate(release.preprocessing)
        if self.preprocessing.fingerprint() != release.preprocessing_version:
            raise ValueError(f"release {release.id} preprocessing does not match its version")
        self.release_id = release.id
        self.class_names = list(release.class_names)
        self.device = device
        model = build_model(len(self.class_names), None)
        model.load_state_dict(torch.load(BytesIO(data), map_location="cpu", weights_only=True))
        self.model = model.to(device).eval()

    def tensors(self, images: Sequence[bytes]) -> torch.Tensor:
        import torch

        return torch.stack([preprocess(load_image(data), self.preprocessing) for data in images])

    def score(self, images: Sequence[bytes], sha256s: Sequence[str]) -> list[dict[str, float]]:
        import torch

        with torch.inference_mode():
            probs = torch.softmax(self.model(self.tensors(images).to(self.device)), dim=1)
        rows = probs.float().cpu().numpy()
        return [{c: float(p) for c, p in zip(self.class_names, row, strict=True)} for row in rows]


_LOADED: dict[str, Scorer] = {}


def scorer_for(release: ModelRelease, store: ObjectStore, device: str = "cpu") -> Scorer:
    """The scorer for a release, loaded once per process."""
    if release.id not in _LOADED:
        log.info("loading release %s", release.id)
        _LOADED[release.id] = (
            PlumbingScorer(release)
            if release.kind == "test_predictor"
            else ReleaseScorer(release, store, device)
        )
    return _LOADED[release.id]


def calibrated(release: ModelRelease, raw: dict[str, float]) -> dict[str, float]:
    """Apply the release's calibration (identity when it has none)."""
    temperature = (release.calibration or {}).get("temperature")
    if temperature is None:
        return dict(raw)
    classes = list(release.class_names)
    row = apply_temperature(np.array([[raw[c] for c in classes]]), float(temperature))[0]
    return {c: float(v) for c, v in zip(classes, row, strict=True)}
