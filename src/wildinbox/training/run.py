"""`wildinbox baseline train`: embed -> fit -> save artifact -> log to MLflow."""

from __future__ import annotations

import json
import logging
import platform
import random
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wildinbox.config import ConfigError, WildInboxConfig, load_config
from wildinbox.datasets.spec import Partition, load_split_spec
from wildinbox.evaluation.data import ImageRow, load_rows
from wildinbox.training.classifier import fit
from wildinbox.training.embeddings import (
    Backbone,
    Extraction,
    cache_key,
    concat,
    extract,
    load_cache,
    save_cache,
)
from wildinbox.training.spec import BaselineConfig, load_baseline_config

log = logging.getLogger(__name__)
TRAIN_FIT = "train_fit"
MLFLOW_URI = "sqlite:///mlruns/mlflow.db"


@dataclass
class Context:
    cfg: BaselineConfig
    run: WildInboxConfig
    split_name: str
    split_version: str
    split_dir: Path
    images_root: Path
    inventory_db: Path
    cache_dir: Path

    @property
    def classes(self) -> list[str]:
        return list(self.run.classes)


def load_context(config_path: Path, data_dir: Path, repo_root: Path = Path(".")) -> Context:
    cfg = load_baseline_config(config_path)
    run = load_config(repo_root / cfg.run_config)
    spec = load_split_spec(repo_root / cfg.splits)
    lock = json.loads((repo_root / "manifests" / f"{spec.name}.lock.json").read_text())
    if run.classes != lock["supported_classes"]:
        raise ConfigError(
            f"run config classes {run.classes} differ from the split lock "
            f"{lock['supported_classes']}"
        )
    if run.dataset.manifest_version != spec.inventory_manifest_version:
        raise ConfigError("run config and split spec pin different inventory versions")
    dataset = spec.inventory_manifest_version.split("-")[0]
    return Context(
        cfg=cfg,
        run=run,
        split_name=spec.name,
        split_version=lock["split_version"],
        split_dir=data_dir / "splits" / spec.name,
        images_root=data_dir / "raw" / dataset / "images",
        inventory_db=data_dir / "inventory" / f"{dataset}.sqlite",
        cache_dir=data_dir
        / "embeddings"
        / cache_key(cfg.backbone, run.preprocessing, lock["split_version"]),
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


class LazyBackbone:
    """Loads pretrained weights only if some embeddings are not cached."""

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self._model: Backbone | None = None

    def get(self) -> Backbone:
        if self._model is None:
            self._model = Backbone(self.ctx.cfg.backbone)
        return self._model


CACHE_CHUNK = 4096


def embed(
    ctx: Context,
    name: str,
    rows: list[ImageRow],
    backbone: LazyBackbone,
    *,
    device: str,
    use_cache: bool = True,
) -> Extraction:
    """Embeddings for `rows`, cached in chunks so an interrupted run resumes."""
    parts = []
    for k in range(0, len(rows), CACHE_CHUNK):
        chunk = rows[k : k + CACHE_CHUNK]
        ids = [r.source_id for r in chunk]
        path = ctx.cache_dir / f"{name}.{k // CACHE_CHUNK:03d}.npz"
        cached = load_cache(path, ids) if use_cache else None
        if cached is None:
            log.info(
                "%s: embedding images %d-%d of %d on %s", name, k, k + len(chunk), len(rows), device
            )
            cached = extract(
                backbone.get(),
                [ctx.images_root / r.storage_path for r in chunk],
                ctx.run.preprocessing,
                device=device,
                batch_size=ctx.cfg.extraction.batch_size,
                num_workers=ctx.cfg.extraction.num_workers,
            )
            save_cache(path, ids, cached)
        parts.append(cached)
    log.info("%s: %d embeddings ready", name, len(rows))
    return concat(parts)


def git_state(repo_root: Path = Path(".")) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--", "src", "configs"], cwd=repo_root, text=True
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
    }
    try:
        if platform.system() == "Darwin":
            info["cpu"] = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
            info["memory_gb"] = (
                int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)) / 2**30
            )
        else:
            info["cpu"] = platform.processor()
    except (OSError, subprocess.CalledProcessError, ValueError):
        pass
    return info


def train_baseline(
    config_path: Path,
    data_dir: Path,
    models_dir: Path,
    *,
    device: str | None = None,
    use_cache: bool = True,
) -> Path:
    ctx = load_context(config_path, data_dir)
    cfg = ctx.cfg
    device = device or cfg.extraction.device
    seed_everything(ctx.run.seed)
    partitions = [Partition.TRAIN, *cfg.evaluation.partitions]
    rows, _ = load_rows(ctx.split_dir, partitions)
    fit_rows = [r for r in rows if r.partition is Partition.TRAIN and r.use_for_fit]
    labels = [r.image_label or "" for r in fit_rows]
    backbone = LazyBackbone(ctx)

    extractions: dict[str, Extraction] = {
        TRAIN_FIT: embed(ctx, TRAIN_FIT, fit_rows, backbone, device=device, use_cache=use_cache)
    }
    for p in cfg.evaluation.partitions:  # embed dev partitions now; evaluation reuses them
        extractions[p.value] = embed(
            ctx,
            p.value,
            [r for r in rows if r.partition is p],
            backbone,
            device=device,
            use_cache=use_cache,
        )

    start = time.perf_counter()
    clf = fit(extractions[TRAIN_FIT].embeddings, labels, ctx.classes, cfg.classifier, ctx.run.seed)
    fit_seconds = time.perf_counter() - start

    out = models_dir / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    clf.save(out / "classifier.npz")
    meta = {
        "name": cfg.name,
        "config": cfg.model_dump(mode="json"),
        "classes": ctx.classes,
        "class_map_fingerprint": ctx.run.class_map.fingerprint(),
        "preprocessing": ctx.run.preprocessing.model_dump(),
        "preprocessing_version": ctx.run.preprocessing.fingerprint(),
        "seed": ctx.run.seed,
        "inventory_manifest_version": ctx.run.dataset.manifest_version,
        "split_version": ctx.split_version,
        "embedding_cache": ctx.cache_dir.name,
        "train_images_per_class": dict(sorted(Counter(labels).items())),
        "train_images": len(fit_rows),
        "device": device,
        "fit_seconds": round(fit_seconds, 3),
        "extraction": {
            k: {
                "images": len(v.embeddings),
                "seconds": round(v.seconds, 2),
                "images_per_second": round(v.images_per_second, 2),
                "peak_rss_mb": round(v.peak_rss_mb, 1),
            }
            for k, v in extractions.items()
        },
        "code": git_state(),
        "hardware": hardware(),
        "versions": {"sklearn": __import__("sklearn").__version__, "numpy": np.__version__},
    }
    meta["mlflow_run_id"] = _log_mlflow(cfg, meta, out)
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log.info("baseline saved to %s", out)
    return out


def _log_mlflow(cfg: BaselineConfig, meta: dict[str, Any], artifact_dir: Path) -> str | None:
    try:
        import mlflow
    except ImportError:
        return None
    Path("mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("wildinbox-baseline")
    with mlflow.start_run(run_name=cfg.name) as run:
        mlflow.set_tags(
            {
                "split_version": meta["split_version"],
                "inventory_manifest_version": meta["inventory_manifest_version"],
                "git_commit": str(meta["code"]["commit"]),
                "git_dirty": str(meta["code"]["dirty"]),
                "device": meta["device"],
            }
        )
        mlflow.log_params(
            {
                "seed": meta["seed"],
                "classifier_c": cfg.classifier.c,
                "class_weight": str(cfg.classifier.class_weight),
                "backbone": f"{cfg.backbone.architecture}/{cfg.backbone.weights}",
                "preprocessing_version": meta["preprocessing_version"],
                "train_images": meta["train_images"],
            }
        )
        mlflow.log_metrics(
            {
                "fit_seconds": meta["fit_seconds"],
                "train_embed_images_per_second": meta["extraction"][TRAIN_FIT]["images_per_second"],
            }
        )
        mlflow.log_artifact(str(artifact_dir / "classifier.npz"))
        return str(run.info.run_id)
