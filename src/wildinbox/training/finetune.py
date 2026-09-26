"""`wildinbox finetune train`: fine-tune EfficientNet-B0 on the training partition.

Only `train` images marked `use_for_fit` are read; this is asserted. Training
inputs come from a resized cache (shorter side `cache_short_side`) purely for
data-loading speed; evaluation always uses the original files and the serving
transform.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from wildinbox.class_map import EMPTY_CLASS
from wildinbox.config import ConfigError, PreprocessingConfig
from wildinbox.datasets.events import Role
from wildinbox.datasets.spec import Partition
from wildinbox.evaluation.data import ImageRow, box_lists, load_rows
from wildinbox.inference.architecture import build_model
from wildinbox.preprocessing import Box, TrainAugmentation, load_image, resize_shorter_side
from wildinbox.training.run import MLFLOW_URI, git_state, hardware, load_context, seed_everything
from wildinbox.training.snapshot import SnapshotError, load_summary

log = logging.getLogger(__name__)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AugmentSpec(_Strict):
    crop_scale: tuple[float, float] = Field(description="Area fraction for images with boxes.")
    unboxed_crop_scale: tuple[float, float] = Field(
        description="Milder crops when an animal has no box to protect."
    )
    min_box_kept: float = Field(gt=0, le=1)
    photometric: bool


class OptimSpec(_Strict):
    lr: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    warmup_steps: int = Field(ge=0)


class FinetuneConfig(_Strict):
    name: str = Field(pattern=r"^[a-z0-9_.-]+$")
    description: str
    run_config: Path
    splits: Path
    weights: Literal["IMAGENET1K_V1"]
    trainable_from_block: int = Field(ge=0, le=8, description="0 = fine-tune every block.")
    imbalance: Literal["loss_weighting", "balanced_sampling"]
    augmentation: AugmentSpec
    optim: OptimSpec
    device: Literal["cpu", "mps", "cuda"]
    num_workers: int = Field(ge=0)
    cache_short_side: int = Field(gt=0)
    # Reviewed deployment data (wildinbox snapshot build) added to the training
    # partition; None trains on the training partition only.
    snapshot: Path | None = None


def load_finetune_config(path: str | Path) -> FinetuneConfig:
    path = Path(path)
    try:
        return FinetuneConfig.model_validate(yaml.safe_load(path.read_text()))
    except FileNotFoundError:
        raise ConfigError(f"{path}: file not found") from None
    except ValidationError as e:
        raise ConfigError(f"{path}: invalid fine-tune config\n{e}") from None


# ------------------------------------------------------------------- model


def freeze_below(model: nn.Module, block: int) -> list[nn.Module]:
    """Freeze feature blocks [0, block). Returns them so their batch-norm layers
    can be kept in eval mode (their running statistics stay ImageNet's)."""
    features = model.get_submodule("features")
    assert isinstance(features, nn.Sequential)
    frozen: list[nn.Module] = [features[i] for i in range(block)]
    for m in frozen:
        for p in m.parameters():
            p.requires_grad_(False)
    return frozen


def load_finetuned(model_dir: Path, device: str = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    meta = json.loads((model_dir / "meta.json").read_text())
    model = build_model(len(meta["classes"]), None)
    state = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval(), meta


# -------------------------------------------------------------------- data


def _cache_one(args: tuple[str, str, int]) -> None:
    src, dst, size = args
    out = Path(dst)
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    img = resize_shorter_side(load_image(src), size)
    tmp = out.with_suffix(".tmp.jpg")
    img.save(tmp, "JPEG", quality=95)
    tmp.replace(out)


def build_input_cache(
    rows: list[ImageRow],
    images_root: Path,
    cache_root: Path,
    size: int,
    workers: int = 4,
    sources: dict[str, Path] | None = None,
) -> None:
    """`sources` maps a row's storage_path to its file when it is not under
    `images_root` (snapshot images)."""
    sources = sources or {}
    jobs = [
        (
            str(sources.get(r.storage_path, images_root / r.storage_path)),
            str(cache_root / r.storage_path),
            size,
        )
        for r in rows
        if not (cache_root / r.storage_path).exists()
    ]
    if not jobs:
        return
    log.info("building %dpx training input cache for %d images", size, len(jobs))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_cache_one, jobs, chunksize=64))


class TrainImages(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        rows: list[ImageRow],
        cache_root: Path,
        boxes: dict[str, list[Box]],
        classes: list[str],
        aug: TrainAugmentation,
        seed: int,
    ) -> None:
        self.rows, self.cache_root, self.boxes = rows, cache_root, boxes
        self.index = {c: i for i, c in enumerate(classes)}
        self.aug, self.seed, self.epoch = aug, seed, 0

    def __len__(self) -> int:
        return len(self.rows)

    def sample_seed(self, i: int) -> int:
        raw = f"{self.seed}:{self.epoch}:{i}".encode()
        return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        row = self.rows[i]
        from PIL import Image

        with Image.open(self.cache_root / row.storage_path) as raw:
            img = raw.convert("RGB")
        boxes = self.boxes.get(row.source_id) or None
        label = row.image_label or ""
        return self.aug(img, boxes, self.sample_seed(i)), self.index[label]


def _single_thread(_: int) -> None:
    torch.set_num_threads(1)


def make_augmentation(cfg: FinetuneConfig, pre: PreprocessingConfig) -> TrainAugmentation:
    a = cfg.augmentation
    return TrainAugmentation(
        pre,
        crop_scale=a.crop_scale,
        unboxed_crop_scale=a.unboxed_crop_scale,
        min_box_kept=a.min_box_kept,
        photometric=a.photometric,
    )


# ------------------------------------------------------------------- train


def training_rows(split_dir: Path) -> list[ImageRow]:
    """The only images a model may be fitted on: `train` partition, `use_for_fit`."""
    rows, _ = load_rows(split_dir, [Partition.TRAIN])
    fit_rows = [r for r in rows if r.use_for_fit]
    if not all(r.partition is Partition.TRAIN for r in fit_rows):
        raise RuntimeError("non-training rows reached the training set")
    return fit_rows


def snapshot_rows(snapshot_dir: Path) -> tuple[list[ImageRow], dict[str, Path], dict[str, Any]]:
    """Fit rows for a snapshot's training frames, their source files, and its summary.
    Rows are marked as fitting rows (partition `train`); their origin is the
    snapshot, recorded in the model's metadata and in the `snapshot:` id prefix.
    Raises `SnapshotError` for a summary training cannot use."""
    summary = load_summary(snapshot_dir)
    rows, sources = [], {}
    checked: set[str] = set()
    for line in (snapshot_dir / "train.jsonl").read_text().splitlines():
        r = json.loads(line)
        storage = f"snapshot-{summary['version']}/{r['sha256']}.jpg"
        path = snapshot_dir / "images" / f"{r['sha256']}.jpg"
        if r["sha256"] not in checked:
            # Only the bytes the snapshot recorded reach the loader.
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != r["sha256"]:
                raise SnapshotError(f"{path} is missing or does not match its SHA-256")
            checked.add(r["sha256"])
        sources[storage] = path
        rows.append(
            ImageRow(
                source_id=f"snapshot:{r['sha256']}",
                event_id=r["event_id"],
                partition=Partition.TRAIN,
                camera_id=r["camera_id"],
                storage_path=storage,
                image_label=r["label"],
                event_label=r["label"],
                event_role=Role.EMPTY if r["label"] == EMPTY_CLASS else Role.SUPPORTED,
                use_for_fit=True,
            )
        )
    return rows, sources, summary


def snapshot_record(snapshot_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    """The model metadata's `trained_on.snapshot`: which snapshot, and whose reviews."""
    return {
        "version": summary["version"],
        "path": str(snapshot_dir),
        "images": summary["train"]["images"],
        "events": summary["train"]["events"],
        "schema": summary["schema"],
        "approved_reviewers": summary["approved_reviewers"],
        "provenance_sha256": (summary["provenance"] or {}).get("sha256"),
    }


def train(config_path: Path, data_dir: Path, models_dir: Path) -> Path:
    cfg = load_finetune_config(config_path)
    code = git_state()  # the code this run starts from, not whatever HEAD is at the end
    ctx = load_context(Path("configs/experiments/baseline.yaml"), data_dir)
    run = ctx.run
    if (Path(cfg.run_config), Path(cfg.splits)) != (Path(ctx.cfg.run_config), Path(ctx.cfg.splits)):
        raise ConfigError("fine-tune and baseline configs must share run config and splits")
    seed_everything(run.seed)

    fit_rows = training_rows(ctx.split_dir)
    sources: dict[str, Path] = {}
    snapshot: dict[str, Any] | None = None
    if cfg.snapshot is not None:
        extra, sources, snapshot = snapshot_rows(cfg.snapshot)
        fit_rows = fit_rows + extra
    classes = ctx.classes
    labels = [r.image_label or "" for r in fit_rows]
    counts = Counter(labels)

    cache_root = data_dir / "cache" / f"train-s{cfg.cache_short_side}"
    build_input_cache(fit_rows, ctx.images_root, cache_root, cfg.cache_short_side, sources=sources)
    boxes = box_lists(ctx.inventory_db)
    aug = make_augmentation(cfg, run.preprocessing)
    dataset = TrainImages(fit_rows, cache_root, boxes, classes, aug, run.seed)

    gen = torch.Generator().manual_seed(run.seed)
    if cfg.imbalance == "balanced_sampling":
        weights = [1.0 / counts[lab] for lab in labels]
        sampler: Any = WeightedRandomSampler(
            weights, num_samples=len(fit_rows), replacement=True, generator=gen
        )
        class_weights = None
    else:
        sampler = torch.utils.data.RandomSampler(dataset, generator=gen)
        class_weights = torch.tensor(
            [len(labels) / (len(classes) * counts[c]) for c in classes], dtype=torch.float32
        )
    loader = DataLoader(
        dataset,
        batch_size=cfg.optim.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        worker_init_fn=_single_thread,
        persistent_workers=False,
        drop_last=True,
    )

    device = cfg.device
    model = build_model(len(classes), cfg.weights).to(device)
    frozen = freeze_below(model, cfg.trainable_from_block)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    total = cfg.optim.epochs * len(loader)

    def lr_at(step: int) -> float:
        if step < cfg.optim.warmup_steps:
            return (step + 1) / cfg.optim.warmup_steps
        progress = (step - cfg.optim.warmup_steps) / max(1, total - cfg.optim.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )

    out = models_dir / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "checkpoint.pt"
    start_epoch, history = 0, []
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ckpt["config"] == cfg.model_dump(mode="json"):
            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["opt"])
            sched.load_state_dict(ckpt["sched"])
            start_epoch, history = ckpt["epoch"], ckpt["history"]
            log.info("resuming %s from epoch %d", cfg.name, start_epoch)

    started = time.perf_counter()
    for epoch in range(start_epoch, cfg.optim.epochs):
        dataset.epoch = epoch
        model.train()
        for m in frozen:
            m.eval()
        t0, total_loss, seen = time.perf_counter(), 0.0, 0
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total_loss += float(loss) * len(y)
            seen += len(y)
            if device == "mps" and step % 50 == 0:
                torch.mps.empty_cache()
            if step % 100 == 0:
                log.info(
                    "%s epoch %d step %d/%d loss %.4f",
                    cfg.name,
                    epoch + 1,
                    step,
                    len(loader),
                    total_loss / seen,
                )
        secs = time.perf_counter() - t0
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total_loss / seen,
                "seconds": secs,
                "images_per_second": seen / secs,
            }
        )
        torch.save(
            {
                "config": cfg.model_dump(mode="json"),
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "epoch": epoch + 1,
                "history": history,
            },
            ckpt_path,
        )
        log.info(
            "%s epoch %d done: loss %.4f, %.0f s",
            cfg.name,
            epoch + 1,
            history[-1]["train_loss"],
            secs,
        )

    torch.save(model.cpu().state_dict(), out / "model.pt")
    from torchvision.models import EfficientNet_B0_Weights

    ids_digest = hashlib.sha256("\n".join(sorted(r.source_id for r in fit_rows)).encode())
    meta: dict[str, Any] = {
        "name": cfg.name,
        "kind": "finetuned",
        "description": cfg.description,
        "config": cfg.model_dump(mode="json"),
        "classes": classes,
        "class_map_fingerprint": run.class_map.fingerprint(),
        "preprocessing": run.preprocessing.model_dump(),
        "preprocessing_version": run.preprocessing.fingerprint(),
        "seed": run.seed,
        "inventory_manifest_version": run.dataset.manifest_version,
        "split_version": ctx.split_version,
        "pretrained_weights": {
            "source": "torchvision",
            "enum": f"EfficientNet_B0_Weights.{cfg.weights}",
            "url": EfficientNet_B0_Weights[cfg.weights].url,
            "pretraining": "ImageNet-1k",
            "wildlife_specific": False,
        },
        "trained_on": {
            "partitions": [Partition.TRAIN.value],
            "images": len(fit_rows),
            "images_per_class": dict(sorted(counts.items())),
            "source_ids_sha256": ids_digest.hexdigest(),
            "snapshot": None
            if snapshot is None or cfg.snapshot is None
            else snapshot_record(cfg.snapshot, snapshot),
        },
        "device": device,
        "history": history,
        "train_seconds": round(sum(h["seconds"] for h in history), 1),
        "wall_seconds_this_run": round(time.perf_counter() - started, 1),
        "code": code,
        "hardware": hardware(),
    }
    meta["mlflow_run_id"] = _log_mlflow(cfg, meta, out)
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    ckpt_path.unlink(missing_ok=True)
    log.info("fine-tuned model saved to %s", out)
    return out


def _log_mlflow(cfg: FinetuneConfig, meta: dict[str, Any], out: Path) -> str | None:
    import mlflow

    Path("mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("wildinbox-finetune")
    with mlflow.start_run(run_name=cfg.name) as run:
        mlflow.set_tags(
            {
                "split_version": meta["split_version"],
                "inventory_manifest_version": meta["inventory_manifest_version"],
                "git_commit": str(meta["code"]["commit"]),
                "git_dirty": str(meta["code"]["dirty"]),
                "pretrained_weights": meta["pretrained_weights"]["url"],
                "trained_on": ",".join(meta["trained_on"]["partitions"]),
                "train_ids_sha256": meta["trained_on"]["source_ids_sha256"],
            }
        )
        flat = {
            "seed": meta["seed"],
            "trainable_from_block": cfg.trainable_from_block,
            "imbalance": cfg.imbalance,
            "photometric": cfg.augmentation.photometric,
            "crop_scale": str(cfg.augmentation.crop_scale),
            "min_box_kept": cfg.augmentation.min_box_kept,
            "lr": cfg.optim.lr,
            "epochs": cfg.optim.epochs,
            "batch_size": cfg.optim.batch_size,
            "weight_decay": cfg.optim.weight_decay,
            "device": cfg.device,
        }
        mlflow.log_params(flat)
        for h in meta["history"]:
            mlflow.log_metrics(
                {"train_loss": h["train_loss"], "train_images_per_second": h["images_per_second"]},
                step=h["epoch"],
            )
        mlflow.log_dict(meta["config"], "config.json")
        mlflow.log_artifact(str(out / "model.pt"))
        return str(run.info.run_id)


def inspect_augmentation(
    config_path: Path, data_dir: Path, out: Path, n: int = 8, variants: int = 4
) -> Path:
    """Contact sheet: each row is an original training frame (smallest animals
    first) followed by augmented versions, with animal boxes drawn in red."""
    from PIL import Image, ImageDraw

    cfg = load_finetune_config(config_path)
    ctx = load_context(Path("configs/experiments/baseline.yaml"), data_dir)
    rows = [r for r in load_rows(ctx.split_dir, [Partition.TRAIN])[0] if r.use_for_fit]
    boxes = box_lists(ctx.inventory_db)
    boxed = sorted(
        (r for r in rows if r.source_id in boxes),
        key=lambda r: max(b[2] * b[3] for b in boxes[r.source_id]),
    )
    pick = boxed[: n // 2] + boxed[len(boxed) // 2 : len(boxed) // 2 + n - n // 2]
    cache_root = data_dir / "cache" / f"train-s{cfg.cache_short_side}"
    build_input_cache(pick, ctx.images_root, cache_root, cfg.cache_short_side)
    aug = make_augmentation(cfg, ctx.run.preprocessing)
    tile = 200
    sheet = Image.new("RGB", ((variants + 1) * tile, len(pick) * tile), "white")

    def draw(img: Image.Image, bxs: list[Box], x0: int, y0: int) -> None:
        im = img.resize((tile, tile))
        d = ImageDraw.Draw(im)
        for bx, by, bw, bh in bxs:
            d.rectangle(
                [bx * tile, by * tile, (bx + bw) * tile, (by + bh) * tile], outline="red", width=2
            )
        sheet.paste(im, (x0, y0))

    kept = []
    for i, row in enumerate(pick):
        with Image.open(cache_root / row.storage_path) as raw:
            img = raw.convert("RGB")
        bxs = boxes[row.source_id]
        draw(img, bxs, 0, i * tile)
        for v in range(variants):
            aug_img, moved = aug.render(img, bxs, seed=1000 * i + v)
            draw(aug_img, moved, (v + 1) * tile, i * tile)
            kept.append(
                min(
                    max(0.0, min(bx + bw, 1) - max(bx, 0))
                    * max(0.0, min(by + bh, 1) - max(by, 0))
                    / (bw * bh)
                    for bx, by, bw, bh in moved
                )
            )
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{cfg.name}.jpg"
    sheet.save(path, "JPEG", quality=85)
    log.info(
        "augmentation sheet -> %s; minimum box area kept across samples: %.2f", path, min(kept)
    )
    return path
