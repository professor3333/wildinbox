"""Frozen-backbone embedding extraction, plus cheap per-image statistics
(night/infrared, blur) used for evaluation slices.

Images go through `wildinbox.preprocessing`, the same code serving uses.
"""

from __future__ import annotations

import hashlib
import logging
import resource
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from wildinbox.config import PreprocessingConfig
from wildinbox.preprocessing import build_eval_transform, load_image
from wildinbox.training.spec import BackboneSpec

log = logging.getLogger(__name__)

# Infrared night frames are grayscale: channels differ by less than this on average.
NIGHT_CHANNEL_DIFF = 2.0
_STATS_WIDTH = 256


class Backbone(nn.Module):
    """EfficientNet-B0 up to global average pooling: 1280-d embeddings."""

    def __init__(self, spec: BackboneSpec) -> None:
        super().__init__()
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

        model = efficientnet_b0(weights=EfficientNet_B0_Weights[spec.weights])
        self.features, self.pool = model.features, model.avgpool
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.pool(self.features(x)), 1)


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


class _Images(Dataset[tuple[torch.Tensor, int, bool, float]]):
    def __init__(self, paths: list[Path], preprocessing: PreprocessingConfig) -> None:
        self.paths = paths
        self.transform = build_eval_transform(preprocessing)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, bool, float]:
        img = load_image(self.paths[i])
        night, blur = image_stats(img)
        return self.transform(img), i, night, blur


def _single_thread(_: int) -> None:
    # Decoding workers must not each start a full PyTorch thread pool.
    torch.set_num_threads(1)


@dataclass
class Extraction:
    embeddings: np.ndarray  # (N, D) float32
    night: np.ndarray  # (N,) bool
    blur: np.ndarray  # (N,) float32
    seconds: float
    images_per_second: float
    peak_rss_mb: float


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


def extract(
    model: nn.Module | Callable[[torch.Tensor], torch.Tensor],
    paths: list[Path],
    preprocessing: PreprocessingConfig,
    *,
    device: str = "cpu",
    batch_size: int = 64,
    num_workers: int = 0,
) -> Extraction:
    if isinstance(model, nn.Module):
        model = model.to(device).eval()
    loader = DataLoader(
        _Images(paths, preprocessing), batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    out: np.ndarray | None = None
    night = np.zeros(len(paths), dtype=bool)
    blur = np.zeros(len(paths), dtype=np.float32)
    start = time.perf_counter()
    done = 0
    with torch.inference_mode():
        for x, idx, n, b in loader:
            emb = model(x.to(device)).float().cpu().numpy()
            if out is None:
                out = np.zeros((len(paths), emb.shape[1]), dtype=np.float32)
            out[idx.numpy()] = emb
            night[idx.numpy()], blur[idx.numpy()] = n.numpy(), b.numpy()
            done += len(idx)
            if device == "mps" and done % (batch_size * 10) < batch_size:
                torch.mps.empty_cache()  # MPS keeps freed buffers; on 8 GB this starves RAM
            if done % (batch_size * 20) < batch_size:
                log.info("embedded %d/%d images", done, len(paths))
    seconds = time.perf_counter() - start
    assert out is not None
    return Extraction(out, night, blur, seconds, len(paths) / max(seconds, 1e-9), peak_rss_mb())


def cache_key(spec: BackboneSpec, preprocessing: PreprocessingConfig, split_version: str) -> str:
    raw = f"{spec.model_dump_json()}|{preprocessing.fingerprint()}|{split_version}"
    return f"{spec.architecture}-{spec.weights}-{hashlib.sha256(raw.encode()).hexdigest()[:12]}"


def save_cache(path: Path, ids: list[str], ext: Extraction) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        ids=np.array(ids),
        embeddings=ext.embeddings,
        night=ext.night,
        blur=ext.blur,
        seconds=ext.seconds,
        images_per_second=ext.images_per_second,
        peak_rss_mb=ext.peak_rss_mb,
    )
    tmp.replace(path)


def concat(parts: list[Extraction]) -> Extraction:
    seconds = sum(p.seconds for p in parts)
    n = sum(len(p.embeddings) for p in parts)
    return Extraction(
        np.concatenate([p.embeddings for p in parts]),
        np.concatenate([p.night for p in parts]),
        np.concatenate([p.blur for p in parts]),
        seconds,
        n / max(seconds, 1e-9),
        max(p.peak_rss_mb for p in parts),
    )


def load_cache(path: Path, ids: list[str]) -> Extraction | None:
    """Cached embeddings, or None if missing or for a different set of images."""
    if not path.exists():
        return None
    with np.load(path) as z:
        if z["ids"].tolist() != ids:
            log.warning("embedding cache %s is for different images; recomputing", path)
            return None
        return Extraction(
            z["embeddings"],
            z["night"],
            z["blur"],
            float(z["seconds"]),
            float(z["images_per_second"]),
            float(z["peak_rss_mb"]),
        )
