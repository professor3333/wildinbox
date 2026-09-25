from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from wildinbox.config import load_config
from wildinbox.datasets.spec import Partition
from wildinbox.preprocessing import TrainAugmentation, safe_crop_window
from wildinbox.training.finetune import (
    build_model,
    freeze_below,
    load_finetune_config,
    training_rows,
)

from .conftest import EXAMPLE_CONFIG, REPO_ROOT

PRE = load_config(EXAMPLE_CONFIG).preprocessing


def _kept(box: tuple[float, float, float, float], crop: tuple[float, float, float, float]) -> float:
    bx, by, bw, bh = box
    cx, cy, cw, ch = crop
    ix = max(0.0, min(bx + bw, cx + cw) - max(bx, cx))
    iy = max(0.0, min(by + bh, cy + ch) - max(by, cy))
    return ix * iy / (bw * bh)


@pytest.mark.parametrize(
    "box", [(0.9, 0.9, 0.05, 0.05), (0.0, 0.4, 0.02, 0.03), (0.4, 0.4, 0.2, 0.2)]
)
def test_safe_crop_never_cuts_an_animal(box: tuple[float, float, float, float]) -> None:
    rng = random.Random(0)
    for _ in range(300):
        crop = safe_crop_window([box], rng, (0.2, 1.0), 0.9, aspect=0.75)
        assert _kept(box, crop) >= 0.9


def test_safe_crop_falls_back_to_full_frame_when_impossible() -> None:
    whole = [(0.0, 0.0, 1.0, 1.0)]  # animal fills the frame: only the full crop keeps it
    assert safe_crop_window(whole, random.Random(1), (0.2, 0.5), 0.9, aspect=0.75) == (
        0.0,
        0.0,
        1.0,
        1.0,
    )


def _aug(photometric: bool = True) -> TrainAugmentation:
    return TrainAugmentation(
        PRE,
        crop_scale=(0.35, 1.0),
        unboxed_crop_scale=(0.8, 1.0),
        min_box_kept=0.9,
        photometric=photometric,
    )


def test_augmentation_is_reproducible_per_seed_and_keeps_boxes() -> None:
    arr = np.random.default_rng(0).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    boxes = [(0.8, 0.7, 0.05, 0.08)]
    aug = _aug()
    a, b, c = aug(img, boxes, 7), aug(img, boxes, 7), aug(img, boxes, 8)
    assert a.shape == (3, PRE.crop_size, PRE.crop_size)
    assert torch.equal(a, b) and not torch.equal(a, c)
    for seed in range(50):
        _, moved = aug.render(img, boxes, seed)
        ((bx, by, bw, bh),) = moved
        inside = (min(bx + bw, 1) - max(bx, 0)) * (min(by + bh, 1) - max(by, 0)) / (bw * bh)
        assert inside >= 0.9 - 1e-6


def test_freeze_below_freezes_only_early_blocks() -> None:
    model = build_model(8, None)
    frozen = freeze_below(model, 6)
    features = model.get_submodule("features")
    assert len(frozen) == 6
    assert not any(p.requires_grad for i in range(6) for p in features[i].parameters())
    assert all(p.requires_grad for i in range(6, len(features)) for p in features[i].parameters())
    assert all(p.requires_grad for p in model.get_submodule("classifier").parameters())


def test_training_rows_come_only_from_the_training_partition(tmp_path: Path) -> None:
    rows = [
        {
            "source_id": "a",
            "event_id": "e1",
            "partition": "train",
            "camera_id": "1",
            "storage_path": "a.jpg",
            "image_label": "raccoon",
            "event_label": "raccoon",
            "event_role": "supported_species",
            "use_for_fit": True,
        },
        {
            "source_id": "b",
            "event_id": "e2",
            "partition": "train",
            "camera_id": "1",
            "storage_path": "b.jpg",
            "image_label": "badger",
            "event_label": "badger",
            "event_role": "unsupported_animal",
            "use_for_fit": False,
        },
        {
            "source_id": "c",
            "event_id": "e3",
            "partition": "calibration",
            "camera_id": "2",
            "storage_path": "c.jpg",
            "image_label": "raccoon",
            "event_label": "raccoon",
            "event_role": "supported_species",
            "use_for_fit": False,
        },
        {
            "source_id": "d",
            "event_id": "e4",
            "partition": "final_test",
            "camera_id": "3",
            "storage_path": "d.jpg",
            "image_label": "raccoon",
            "event_label": "raccoon",
            "event_role": "supported_species",
            "use_for_fit": False,
        },
    ]
    with gzip.open(tmp_path / "images.jsonl.gz", "wt") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    with gzip.open(tmp_path / "events.jsonl.gz", "wt") as f:
        f.write("")
    got = training_rows(tmp_path)
    assert [r.source_id for r in got] == ["a"]
    assert all(r.partition is Partition.TRAIN for r in got)


@pytest.mark.parametrize(
    "path", sorted((REPO_ROOT / "configs/experiments").glob("finetune-*.yaml"))
)
def test_experiment_configs_are_valid(path: Path) -> None:
    cfg = load_finetune_config(path)
    assert cfg.name == path.stem
    assert cfg.augmentation.min_box_kept >= 0.9  # crops may not remove the animal


def test_eval_cache_is_keyed_by_model_weights(tmp_path: Path) -> None:
    from wildinbox.evaluation.predictors import FinetunedPredictor

    (tmp_path / "meta.json").write_text(json.dumps({"classes": ["empty", "cat"]}))
    digests = []
    for seed in (0, 1):  # "retrain" into the same directory
        torch.manual_seed(seed)
        torch.save(build_model(2, None).state_dict(), tmp_path / "model.pt")
        digests.append(FinetunedPredictor(None, tmp_path, "cpu").weights_digest)  # type: ignore[arg-type]
    assert digests[0] != digests[1]
