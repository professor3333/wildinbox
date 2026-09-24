from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.models import EfficientNet_B0_Weights

from wildinbox.config import load_config
from wildinbox.preprocessing import build_train_transform, load_image, preprocess

from .conftest import EXAMPLE_CONFIG

CFG = load_config(EXAMPLE_CONFIG).preprocessing


def _random_image(w: int = 640, h: int = 480, mode: str = "RGB") -> Image.Image:
    rng = np.random.default_rng(0)
    channels = {"RGB": 3, "RGBA": 4}.get(mode)
    shape = (h, w, channels) if channels else (h, w)
    return Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8), mode=mode)


def _encode(img: Image.Image, fmt: str, **kwargs: object) -> bytes:
    buf = BytesIO()
    img.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


def test_matches_torchvision_reference_for_pretrained_weights() -> None:
    """Our eval transform must reproduce what the pretrained weights expect."""
    img = _random_image()
    reference = EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()(img)
    ours = preprocess(img, CFG)
    assert ours.shape == (3, 224, 224)
    assert torch.allclose(ours, reference, atol=1e-5)


def test_eval_transform_is_deterministic() -> None:
    img = _random_image()
    assert torch.equal(preprocess(img, CFG), preprocess(img, CFG))


def test_train_transform_shape_and_scale_match_eval() -> None:
    torch.manual_seed(0)
    out = build_train_transform(CFG)(_random_image())
    assert out.shape == (3, CFG.crop_size, CFG.crop_size)
    assert out.dtype == torch.float32


@pytest.mark.parametrize("mode", ["L", "RGBA"])
def test_load_image_always_returns_rgb(mode: str) -> None:
    data = _encode(_random_image(mode=mode), "PNG")
    img = load_image(data)
    assert img.mode == "RGB"
    assert preprocess(img, CFG).shape == (3, 224, 224)


def test_load_image_applies_exif_orientation() -> None:
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: rotate 90° CW to display
    data = _encode(_random_image(w=640, h=480), "JPEG", exif=exif)
    assert load_image(data).size == (480, 640)


def test_load_image_rejects_corrupt_bytes() -> None:
    with pytest.raises(OSError):
        load_image(b"not an image")
