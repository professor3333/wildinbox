"""Model adapters for the evaluation harness. Every model is scored through the
same interface, so metrics, slices, and benchmarks are directly comparable."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from wildinbox.config import ConfigError
from wildinbox.evaluation.data import ImageRow
from wildinbox.training.classifier import LinearClassifier
from wildinbox.training.embeddings import (
    Extraction,
    cache_identity,
    extract,
    load_cache,
    save_cache,
)
from wildinbox.training.run import Context, LazyBackbone, embed


def check_preprocessing(ctx: Context, model_dir: Path) -> None:
    """Evaluation must preprocess exactly as the model artifact records;
    anything else would score a different model than the one being reported."""
    meta = json.loads((model_dir / "meta.json").read_text())
    expected, configured = meta.get("preprocessing_version"), ctx.run.preprocessing.fingerprint()
    if expected != configured:
        raise ConfigError(
            f"{model_dir} was trained with preprocessing {expected} "
            f"({meta.get('preprocessing')}), but the evaluation config gives {configured} "
            f"({ctx.run.preprocessing.model_dump(mode='json')})"
        )


class Predictor(Protocol):
    classes: tuple[str, ...]
    device: str

    def score(self, name: str, rows: list[ImageRow]) -> Extraction:
        """Class probabilities (in `classes` order) plus night/blur statistics."""
        ...

    def forward(self, x: torch.Tensor) -> np.ndarray:
        """Probabilities for a preprocessed batch already on `device`."""
        ...


class LinearProbePredictor:
    def __init__(self, ctx: Context, model_dir: Path, device: str) -> None:
        check_preprocessing(ctx, model_dir)
        self.ctx, self.device = ctx, device
        self.clf = LinearClassifier.load(model_dir / "classifier.npz")
        self.classes = self.clf.classes
        self.backbone = LazyBackbone(ctx)

    def score(self, name: str, rows: list[ImageRow]) -> Extraction:
        ext = embed(self.ctx, name, rows, self.backbone, device=self.device)
        return Extraction(
            self.clf.predict_proba(ext.embeddings),
            ext.night,
            ext.blur,
            ext.seconds,
            ext.images_per_second,
            ext.peak_rss_mb,
        )

    def forward(self, x: torch.Tensor) -> np.ndarray:
        model = self.backbone.get().to(self.device)
        return self.clf.predict_proba(model(x).float().cpu().numpy())


class FinetunedPredictor:
    def __init__(self, ctx: Context, model_dir: Path, device: str) -> None:
        from wildinbox.training.finetune import load_finetuned

        check_preprocessing(ctx, model_dir)
        self.ctx, self.device, self.model_dir = ctx, device, model_dir
        self.model, meta = load_finetuned(model_dir, device)
        self.classes = tuple(meta["classes"])
        # Cached predictions are keyed by the weights, so a model retrained into
        # the same directory can never be scored from the previous model's cache.
        self.weights_digest = hashlib.sha256((model_dir / "model.pt").read_bytes()).hexdigest()[:12]

    def _probs(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model(x), dim=1)

    def _cached(
        self, name: str, rows: list[ImageRow], fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> Extraction:
        """`fn` over `rows`, cached under the weights, preprocessing, and input
        files; a change to any of them recomputes."""
        ids = [r.source_id for r in rows]
        files = [self.ctx.images_root / r.storage_path for r in rows]
        identity = cache_identity(
            f"finetuned:{self.weights_digest}", self.ctx.run.preprocessing, files
        )
        path = self.model_dir / "eval-cache" / self.weights_digest / f"{name}.npz"
        cached = load_cache(path, ids, identity=identity)
        if cached is not None:
            return cached
        ext = extract(
            fn,
            files,
            self.ctx.run.preprocessing,
            device=self.device,
            batch_size=64,
            num_workers=self.ctx.cfg.extraction.num_workers,
        )
        save_cache(path, ids, ext, identity=identity)
        return ext

    def score(self, name: str, rows: list[ImageRow]) -> Extraction:
        return self._cached(name, rows, self._probs)

    def forward(self, x: torch.Tensor) -> np.ndarray:
        out: np.ndarray = self._probs(x).float().cpu().numpy()
        return out

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled penultimate features (the classifier's input)."""
        m = self.model
        return torch.flatten(m.avgpool(m.features(x)), 1)  # type: ignore[operator]

    def features(self, name: str, rows: list[ImageRow]) -> Extraction:
        """Penultimate features for `rows`, cached like `score`."""
        return self._cached(f"features-{name}", rows, self._features)


def predictor_for(ctx: Context, model_dir: Path, device: str) -> Predictor:
    kind = json.loads((model_dir / "meta.json").read_text()).get("kind", "linear_probe")
    if kind == "finetuned":
        return FinetunedPredictor(ctx, model_dir, device)
    return LinearProbePredictor(ctx, model_dir, device)
