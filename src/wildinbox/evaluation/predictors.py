"""Model adapters for the evaluation harness. Every model is scored through the
same interface, so metrics, slices, and benchmarks are directly comparable."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from wildinbox.evaluation.data import ImageRow
from wildinbox.training.classifier import LinearClassifier
from wildinbox.training.embeddings import Extraction, extract, load_cache, save_cache
from wildinbox.training.run import Context, LazyBackbone, embed


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

        self.ctx, self.device, self.model_dir = ctx, device, model_dir
        self.model, meta = load_finetuned(model_dir, device)
        self.classes = tuple(meta["classes"])
        # Cached predictions are keyed by the weights, so a model retrained into
        # the same directory can never be scored from the previous model's cache.
        self.weights_digest = hashlib.sha256((model_dir / "model.pt").read_bytes()).hexdigest()[:12]

    def _probs(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model(x), dim=1)

    def score(self, name: str, rows: list[ImageRow]) -> Extraction:
        ids = [r.source_id for r in rows]
        path = self.model_dir / "eval-cache" / self.weights_digest / f"{name}.npz"
        cached = load_cache(path, ids)
        if cached is not None:
            return cached
        ext = extract(
            self._probs,
            [self.ctx.images_root / r.storage_path for r in rows],
            self.ctx.run.preprocessing,
            device=self.device,
            batch_size=64,
            num_workers=self.ctx.cfg.extraction.num_workers,
        )
        save_cache(path, ids, ext)
        return ext

    def forward(self, x: torch.Tensor) -> np.ndarray:
        out: np.ndarray = self._probs(x).float().cpu().numpy()
        return out


def predictor_for(ctx: Context, model_dir: Path, device: str) -> Predictor:
    kind = json.loads((model_dir / "meta.json").read_text()).get("kind", "linear_probe")
    if kind == "finetuned":
        return FinetunedPredictor(ctx, model_dir, device)
    return LinearProbePredictor(ctx, model_dir, device)
