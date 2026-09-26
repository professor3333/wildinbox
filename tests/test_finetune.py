from __future__ import annotations

import gzip
import json
import random
from pathlib import Path
from typing import Any

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


def test_eval_cache_reuses_unchanged_inputs_and_refuses_other_preprocessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same weights, preprocessing, and files: cached. New weights or a changed
    file: recomputed. Preprocessing other than the model's: a ConfigError."""
    import os
    from types import SimpleNamespace

    from wildinbox.config import ConfigError
    from wildinbox.evaluation import predictors
    from wildinbox.evaluation.data import ImageRow
    from wildinbox.training.embeddings import Extraction

    pre = load_config(REPO_ROOT / "configs/example.yaml").preprocessing
    model_dir, images = tmp_path / "model", tmp_path / "images"
    model_dir.mkdir()
    images.mkdir()
    (model_dir / "meta.json").write_text(
        json.dumps({"classes": ["empty", "cat"], "preprocessing_version": pre.fingerprint()})
    )
    rows = []
    for i in range(2):
        Image.new("RGB", (40, 30), (i * 50, 0, 0)).save(images / f"{i}.jpg")
        rows.append(
            ImageRow(
                f"s{i}",
                f"e{i}",
                Partition.CALIBRATION,
                "c",
                f"{i}.jpg",
                "cat",
                "cat",
                "supported_species",
                False,
            )
        )
    calls: list[int] = []

    def fake_extract(fn: Any, files: list[Path], *a: Any, **k: Any) -> Extraction:
        calls.append(len(files))
        n = len(files)
        return Extraction(np.full((n, 2), 0.5), np.zeros(n, bool), np.ones(n), 1.0, 1.0, 1.0)

    monkeypatch.setattr(predictors, "extract", fake_extract)

    def ctx(preprocessing: Any = pre) -> Any:
        return SimpleNamespace(
            run=SimpleNamespace(preprocessing=preprocessing),
            images_root=images,
            cfg=SimpleNamespace(extraction=SimpleNamespace(num_workers=0)),
        )

    def predictor(seed: int = 0) -> Any:
        torch.manual_seed(seed)
        torch.save(build_model(2, None).state_dict(), model_dir / "model.pt")
        return predictors.FinetunedPredictor(ctx(), model_dir, "cpu")

    first = predictor()
    first.score("cal", rows)
    first.score("cal", rows)
    assert calls == [2]  # unchanged inputs reuse the cache
    st = (images / "1.jpg").stat()
    os.utime(images / "1.jpg", ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    first.score("cal", rows)
    assert calls == [2, 2]  # a changed input file recomputes
    retrained = predictor(seed=1)  # retrained into the same directory
    assert retrained.weights_digest != first.weights_digest
    retrained.score("cal", rows)
    assert calls == [2, 2, 2]
    with pytest.raises(ConfigError, match="trained with preprocessing"):
        predictors.FinetunedPredictor(
            ctx(pre.model_copy(update={"crop_size": 192})), model_dir, "cpu"
        )


def test_update_candidate_uses_the_deployed_recipe_unchanged() -> None:
    e3 = load_finetune_config(REPO_ROOT / "configs/experiments/finetune-e3-deep-balanced.yaml")
    cand = load_finetune_config(REPO_ROOT / "configs/experiments/finetune-e3-update1.yaml")
    differs = {k for k in e3.model_dump() if e3.model_dump()[k] != cand.model_dump()[k]}
    assert differs == {"name", "description", "snapshot"}
    assert cand.snapshot is not None and e3.snapshot is None
