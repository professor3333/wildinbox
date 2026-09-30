"""Events per camera in time order, and the released model's frame outputs.

A camera's first N events (by start time) stand in for the events a person
reviews after installing it; their ground-truth labels play the reviews. Every
later event is what adaptation is judged on.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wildinbox.datasets.spec import Partition
from wildinbox.evaluation.data import ImageRow, assert_development, load_rows


@dataclass(frozen=True)
class CameraEvent:
    event_id: str
    camera_id: str
    start: str | None
    role: str
    label: str | None
    image_ids: tuple[str, ...]
    image_labels: tuple[str | None, ...]


def load_camera_events(
    split_dir: Path, partitions: list[Partition]
) -> tuple[dict[str, list[CameraEvent]], list[ImageRow]]:
    """Usable events per camera, oldest first (events without a start time
    last), and the image rows behind them."""
    wanted = {p.value for p in assert_development(partitions)}
    rows, _ = load_rows(split_dir, partitions)
    return camera_events_from(split_dir, wanted, rows), rows


def camera_events_from(
    split_dir: Path, wanted: set[str], rows: list[ImageRow]
) -> dict[str, list[CameraEvent]]:
    """Events of the `wanted` partitions whose image rows are already loaded.
    Development callers go through `load_camera_events`; the locked fresh test
    only through `wildinbox.adaptation.fresh_test`."""
    image_label = {r.source_id: r.image_label for r in rows}
    cams: dict[str, list[CameraEvent]] = defaultdict(list)
    with gzip.open(split_dir / "events.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["partition"] not in wanted or r["excluded_reason"] or not r["image_ids"]:
                continue
            ids = tuple(r["image_ids"])
            cams[r["camera_id"]].append(
                CameraEvent(
                    event_id=r["event_id"],
                    camera_id=r["camera_id"],
                    start=r["start"],
                    role=r["role"],
                    label=r["label"],
                    image_ids=ids,
                    image_labels=tuple(image_label[i] for i in ids),
                )
            )
    for events in cams.values():
        events.sort(key=lambda e: (e.start is None, e.start or "", e.event_id))
    return dict(cams)


@dataclass(frozen=True)
class FrameOutputs:
    """The released model on every frame: log-probabilities and features."""

    index: dict[str, int]
    log_probs: np.ndarray  # (frames, classes), uncalibrated
    features: np.ndarray  # (frames, 1280), pooled penultimate layer
    classes: tuple[str, ...]

    def rows(self, ids: tuple[str, ...] | list[str]) -> np.ndarray:
        return np.array([self.index[i] for i in ids])


def score_frames(
    config_path: Path,
    model_dir: Path,
    split_dir: Path,
    images_root: Path,
    rows: list[ImageRow],
    name: str,
    device: str,
) -> FrameOutputs:
    """Released-model outputs for `rows`, through the same cached, preprocessing-
    checked scorer as every development evaluation."""
    from wildinbox.evaluation.predictors import FinetunedPredictor
    from wildinbox.settings import Settings
    from wildinbox.training.run import load_context

    ctx = load_context(config_path, Settings().data_dir)
    ctx = dataclasses.replace(ctx, split_dir=split_dir, images_root=images_root)
    predictor = FinetunedPredictor(ctx, model_dir, device)
    probs = predictor.score(name, rows)
    feats = predictor.features(name, rows)
    return FrameOutputs(
        index={r.source_id: i for i, r in enumerate(rows)},
        log_probs=np.log(np.clip(probs.embeddings, 1e-12, 1.0)),
        features=feats.embeddings,
        classes=predictor.classes,
    )
