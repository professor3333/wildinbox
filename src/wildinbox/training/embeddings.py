"""Frozen-backbone embedding extraction, plus cheap per-image statistics
(night/infrared, blur) used for evaluation slices.

Images go through `wildinbox.preprocessing`, the same code serving uses.
"""

from __future__ import annotations

import hashlib
import json
import logging
import resource
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from wildinbox.config import PreprocessingConfig
from wildinbox.preprocessing import build_eval_transform, load_image
from wildinbox.quality import image_stats
from wildinbox.training.spec import BackboneSpec

log = logging.getLogger(__name__)


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


def cache_identity(model: str, preprocessing: PreprocessingConfig, files: list[Path]) -> str:
    """What a cached extraction depends on besides the image ids: the model,
    the preprocessing, and the input files (path, size, modification time).
    A change to any of them makes the cache miss instead of returning stale
    outputs."""
    digest = hashlib.sha256()
    for f in files:
        st = f.stat()
        digest.update(f"{f}\t{st.st_size}\t{st.st_mtime_ns}\n".encode())
    return json.dumps(
        {
            "model": model,
            "preprocessing": preprocessing.fingerprint(),
            "inputs": digest.hexdigest(),
        },
        sort_keys=True,
    )


def save_cache(path: Path, ids: list[str], ext: Extraction, *, identity: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        identity=np.array(identity),
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


def load_cache(path: Path, ids: list[str], *, identity: str) -> Extraction | None:
    """Cached outputs, or None if missing, for a different set of images, or made
    under a different `cache_identity` (model, preprocessing, or input files);
    caches written before identities were recorded never match."""
    if not path.exists():
        return None
    with np.load(path) as z:
        if z["ids"].tolist() != ids:
            log.warning("embedding cache %s is for different images; recomputing", path)
            return None
        if "identity" not in z.files or str(z["identity"]) != identity:
            log.warning(
                "cache %s was made with another model, preprocessing, or input files; recomputing",
                path,
            )
            return None
        return Extraction(
            z["embeddings"],
            z["night"],
            z["blur"],
            float(z["seconds"]),
            float(z["images_per_second"]),
            float(z["peak_rss_mb"]),
        )
