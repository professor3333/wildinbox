"""The only image preprocessing implementation. Training and serving both call
into this module; no other module may build its own transforms.
"""

from __future__ import annotations

import math
import random
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


# --------------------------------------------------------------------------
# Training-only augmentation. Serving never calls anything below.

Box = tuple[float, float, float, float]  # x, y, w, h as fractions of the image


def resize_shorter_side(img: Image.Image, size: int) -> Image.Image:
    """Downscale so the shorter side is `size` (training input cache only)."""
    scale = size / min(img.size)
    if scale >= 1:
        return img
    new = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    return img.resize(new, Image.Resampling.BICUBIC, reducing_gap=3.0)


def _kept_fraction(box: Box, crop: tuple[float, float, float, float]) -> float:
    bx, by, bw, bh = box
    cx, cy, cw, ch = crop
    ix = max(0.0, min(bx + bw, cx + cw) - max(bx, cx))
    iy = max(0.0, min(by + bh, cy + ch) - max(by, cy))
    return (ix * iy) / (bw * bh) if bw * bh > 0 else 1.0


def safe_crop_window(
    boxes: list[Box],
    rng: random.Random,
    scale: tuple[float, float],
    min_box_kept: float,
    aspect: float,
    ratio: tuple[float, float] = (3 / 4, 4 / 3),
    tries: int = 20,
) -> tuple[float, float, float, float]:
    """A random crop (fractions of the image) that keeps >= `min_box_kept` of
    every annotated animal box. Falls back to the full image, so a crop can
    never remove the animal while the label still says it is there."""
    for _ in range(tries):
        area = rng.uniform(*scale)
        r = math.exp(rng.uniform(math.log(ratio[0]), math.log(ratio[1])))
        w = math.sqrt(area * r / aspect)
        h = math.sqrt(area * aspect / r)
        if w > 1 or h > 1:
            continue
        crop = (rng.uniform(0, 1 - w), rng.uniform(0, 1 - h), w, h)
        if all(_kept_fraction(b, crop) >= min_box_kept for b in boxes):
            return crop
    return (0.0, 0.0, 1.0, 1.0)


class TrainAugmentation:
    """Box-aware crop + flip (+ optional photometric changes), then the same
    tensor conversion and normalization as `build_eval_transform`."""

    def __init__(
        self,
        cfg: PreprocessingConfig,
        *,
        crop_scale: tuple[float, float],
        unboxed_crop_scale: tuple[float, float],
        min_box_kept: float,
        photometric: bool,
    ) -> None:
        self.cfg = cfg
        self.crop_scale, self.unboxed_crop_scale = crop_scale, unboxed_crop_scale
        self.min_box_kept = min_box_kept
        self.resize = v2.Resize(
            (cfg.crop_size, cfg.crop_size),
            interpolation=_INTERPOLATION[cfg.interpolation],
            antialias=True,
        )
        self.photometric = (
            v2.Compose(
                [
                    v2.RandomApply(
                        [v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)], p=0.8
                    ),
                    v2.RandomGrayscale(p=0.2),  # daytime frames look like infrared ones
                    v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.1),  # motion blur
                ]
            )
            if photometric
            else None
        )
        self.to_tensor = v2.Compose(_to_normalized_tensor(cfg))

    def crop(
        self, img: Image.Image, boxes: list[Box] | None, rng: random.Random
    ) -> tuple[Image.Image, list[Box]]:
        """Returns the cropped (and maybe flipped) image and the boxes in its frame."""
        scale = self.crop_scale if boxes else self.unboxed_crop_scale
        x, y, w, h = safe_crop_window(
            boxes or [], rng, scale, self.min_box_kept, aspect=img.height / img.width
        )
        W, H = img.size
        out = img.crop((round(x * W), round(y * H), round((x + w) * W), round((y + h) * H)))
        moved = [((bx - x) / w, (by - y) / h, bw / w, bh / h) for bx, by, bw, bh in boxes or []]
        if rng.random() < 0.5:
            out = out.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            moved = [(1 - bx - bw, by, bw, bh) for bx, by, bw, bh in moved]
        return out, moved

    def render(
        self, img: Image.Image, boxes: list[Box] | None, seed: int
    ) -> tuple[Image.Image, list[Box]]:
        """The augmented image before tensor conversion, and where the boxes went."""
        out, moved = self.crop(img, boxes, random.Random(seed))
        out = self.resize(out)
        if self.photometric is not None:
            torch.manual_seed(seed)
            out = self.photometric(out)
        return out, moved

    def __call__(self, img: Image.Image, boxes: list[Box] | None, seed: int) -> torch.Tensor:
        """`seed` identifies (run, epoch, sample), so augmentation is reproducible
        whatever the number of data-loader workers."""
        out, _ = self.render(img, boxes, seed)
        tensor: torch.Tensor = self.to_tensor(out)
        return tensor
