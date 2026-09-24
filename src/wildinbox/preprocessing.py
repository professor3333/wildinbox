"""The only image preprocessing implementation. Training and serving both call
into this module; no other module may build its own transforms.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision.transforms import InterpolationMode, v2

from wildinbox.config import PreprocessingConfig

_INTERPOLATION = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
}


def load_image(source: str | Path | bytes) -> Image.Image:
    """Decode an image, apply its EXIF orientation, and convert to 3-channel RGB.

    Night-time infrared frames are often grayscale and PNGs may carry alpha;
    both become RGB so the model always sees the same input layout.
    """
    fp = BytesIO(source) if isinstance(source, bytes) else source
    with Image.open(fp) as img:
        img.load()
        return ImageOps.exif_transpose(img).convert("RGB")


def _to_normalized_tensor(cfg: PreprocessingConfig) -> list[v2.Transform]:
    return [
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=list(cfg.mean), std=list(cfg.std)),
    ]


def build_eval_transform(cfg: PreprocessingConfig) -> v2.Compose:
    """Deterministic transform for validation, test, and serving."""
    interpolation = _INTERPOLATION[cfg.interpolation]
    return v2.Compose(
        [
            v2.Resize(cfg.resize_size, interpolation=interpolation, antialias=True),
            v2.CenterCrop(cfg.crop_size),
            *_to_normalized_tensor(cfg),
        ]
    )


def build_train_transform(cfg: PreprocessingConfig) -> v2.Compose:
    """Training transform: random augmentation, then the same tensor conversion
    and normalization as build_eval_transform."""
    interpolation = _INTERPOLATION[cfg.interpolation]
    return v2.Compose(
        [
            v2.RandomResizedCrop(cfg.crop_size, interpolation=interpolation, antialias=True),
            v2.RandomHorizontalFlip(),
            *_to_normalized_tensor(cfg),
        ]
    )


def preprocess(image: Image.Image, cfg: PreprocessingConfig) -> torch.Tensor:
    """Eval-mode preprocessing of one image to a (3, crop, crop) float tensor."""
    out: torch.Tensor = build_eval_transform(cfg)(image)
    return out
