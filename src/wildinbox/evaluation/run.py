"""`wildinbox evaluate`: score development partitions and write the report.

Outputs:
- reports/.../metrics.json  compact numbers, committed; the reproducibility reference
- reports/.../README.md     human report with error gallery
- <model>/eval/*.jsonl.gz   every image and event with its prediction and error type,
                            so each metric can be traced to the examples behind it
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wildinbox.class_map import EMPTY_CLASS
from wildinbox.datasets.spec import Partition
from wildinbox.evaluation.data import ImageRow, box_areas, load_rows
from wildinbox.evaluation.metrics import ScoredEvent, event_metrics, image_metrics
from wildinbox.evaluation.predictors import Predictor, predictor_for
from wildinbox.policy.conservative import Thresholds
from wildinbox.settings import Settings
from wildinbox.training.embeddings import peak_rss_mb
from wildinbox.training.run import Context, load_context

log = logging.getLogger(__name__)
UNSEEN = "unseen_cameras"


@dataclass
class Scored:
    row: ImageRow
    probs: dict[str, float]
    pred: str
    confidence: float
    night: bool
    blur: float

    @property
    def supported(self) -> bool:
        return self.row.image_label is not None and self.row.image_label in self.probs

    @property
    def is_animal(self) -> bool:
        return self.row.image_label not in (None, EMPTY_CLASS) and self.row.event_role not in (
            "non_animal",
        )

    def error_type(self) -> str | None:
        label = self.row.image_label
        if label is None:
            return None
        if self.supported:
            if self.pred == label:
                return None
            if self.pred == EMPTY_CLASS:
                return "false_empty"
            if label == EMPTY_CLASS:
                return "false_animal"
            return "confused_species"
        if self.row.event_role == "unsupported_animal":
            return (
                "unfamiliar_predicted_empty"
                if self.pred == EMPTY_CLASS
                else ("unfamiliar_predicted_known")
            )
        return None


def _score(predictor: Predictor, rows: list[ImageRow]) -> list[Scored]:
    out: list[Scored] = []
    for part in sorted({r.partition for r in rows}, key=lambda p: p.value):
        prows = [r for r in rows if r.partition is part]
        ext = predictor.score(part.value, prows)
        for i, r in enumerate(prows):
            p = {c: float(v) for c, v in zip(predictor.classes, ext.embeddings[i], strict=True)}
            pred = max(p, key=p.__getitem__)
            out.append(Scored(r, p, pred, p[pred], bool(ext.night[i]), float(ext.blur[i])))
    return out


def _events(scored: list[Scored], events: dict[str, Any]) -> list[ScoredEvent]:
    by_image = {s.row.source_id: s for s in scored}
    out = []
    for ev in events.values():
        frames = [by_image[i].probs for i in ev.image_ids if i in by_image]
        if frames:
            out.append(ScoredEvent(ev.event_id, ev.role, ev.label, ev.animal_present, frames))
    return out


def _group_metrics(
    scored: list[Scored], events: dict[str, Any], classes: list[str], cfg: Any
) -> dict[str, Any]:
    sup = [s for s in scored if s.supported]
    unsup = [s for s in scored if s.row.event_role == "unsupported_animal" and not s.supported]
    ev = _events(scored, events)
    ref = Thresholds(cfg.reference_empty_threshold, cfg.reference_species_threshold)
    empty_sweep = [event_metrics(ev, Thresholds(t, ref.species))[0] for t in cfg.empty_thresholds]
    species_sweep = [event_metrics(ev, Thresholds(ref.empty, t))[0] for t in cfg.species_thresholds]
    return {
        "image": image_metrics(
            [s.row.image_label or "" for s in sup], [s.pred for s in sup], classes
        ),
        "unsupported_images": {
            "n": len(unsup),
            "predicted": dict(Counter(s.pred for s in unsup).most_common()),
            "predicted_known_species_at_reference": sum(
                1 for s in unsup if s.pred != EMPTY_CLASS and s.confidence >= ref.species
            ),
        },
        "event_reference": event_metrics(ev, ref)[0],
        "event_empty_sweep": empty_sweep,
        "event_species_sweep": species_sweep,
    }


def _slices(
    scored: list[Scored], events: dict[str, Any], classes: list[str], ref: Thresholds
) -> dict[str, Any]:
    def summary(items: list[Scored], evs: dict[str, Any]) -> dict[str, Any]:
        sup = [s for s in items if s.supported]
        im = image_metrics([s.row.image_label or "" for s in sup], [s.pred for s in sup], classes)
        em = event_metrics(_events(items, evs), ref)[0]
        return {
            "images": len(sup),
            "macro_f1": im["macro_f1"],
            "animal_image_recall": im["animal_image_recall"],
            "animal_events": em["animal_events"],
            "false_empty": em["false_empty"],
            "false_empty_rate": em["false_empty_rate"],
        }

    by_cam: dict[str, list[Scored]] = defaultdict(list)
    by_night: dict[str, list[Scored]] = defaultdict(list)
    for s in scored:
        by_cam[s.row.camera_id].append(s)
    flags: dict[str, list[bool]] = defaultdict(list)
    for s in scored:
        flags[s.row.event_id].append(s.night)
    night_of_event = {eid: sum(f) * 2 >= len(f) for eid, f in flags.items()}
    for s in scored:
        by_night["night" if night_of_event.get(s.row.event_id) else "day"].append(s)
    return {
        "camera": {
            c: summary(v, {e: events[e] for e in {s.row.event_id for s in v}})
            for c, v in sorted(by_cam.items(), key=lambda kv: kv[0])
        },
        "day_night": {
            k: summary(v, {e: events[e] for e in {s.row.event_id for s in v}})
            for k, v in sorted(by_night.items())
        },
    }


def _benchmark(
    ctx: Context, predictor: Predictor, rows: list[ImageRow], n: int = 32
) -> dict[str, Any]:
    """End-to-end single-image latency: decode + preprocess + model + probabilities."""
    import torch

    from wildinbox.preprocessing import load_image, preprocess

    paths = [ctx.images_root / r.storage_path for r in rows[: n + 3]]
    times = []
    with torch.inference_mode():
        for i, path in enumerate(paths):
            start = time.perf_counter()
            x = preprocess(load_image(path), ctx.run.preprocessing).unsqueeze(0)
            predictor.forward(x.to(predictor.device))
            if i >= 3:  # warm-up
                times.append((time.perf_counter() - start) * 1000)
    times.sort()
    return {
        "device": predictor.device,
        "images": len(times),
        "p50_ms": statistics.median(times),
        "p95_ms": times[int(0.95 * (len(times) - 1))],
        "peak_rss_mb": peak_rss_mb(),
    }


def evaluate(
    model_dir: Path, config_path: Path, report_dir: Path, *, benchmark: bool = True
) -> dict[str, Any]:
    settings = Settings()
    ctx = load_context(config_path, settings.data_dir)
    meta = json.loads((model_dir / "meta.json").read_text())
    device = meta["device"]
    predictor = predictor_for(ctx, model_dir, device)
    if list(predictor.classes) != ctx.classes:
        raise ValueError(f"model classes {predictor.classes} differ from config {ctx.classes}")
    ecfg = ctx.cfg.evaluation
    rows, events = load_rows(ctx.split_dir, ecfg.partitions, box_areas(ctx.inventory_db))
    scored = _score(predictor, rows)
    ref = Thresholds(ecfg.reference_empty_threshold, ecfg.reference_species_threshold)

    groups: dict[str, list[Partition]] = {UNSEEN: list(ecfg.unseen_camera_partitions)}
    groups.update({p.value: [p] for p in ecfg.partitions})
    results: dict[str, Any] = {}
    for name, parts in groups.items():
        items = [s for s in scored if s.row.partition in parts]
        evs = {k: v for k, v in events.items() if v.partition in parts}
        results[name] = _group_metrics(items, evs, ctx.classes, ecfg)
    unseen = [s for s in scored if s.row.partition in groups[UNSEEN]]
    unseen_events = {k: v for k, v in events.items() if v.partition in groups[UNSEEN]}
    results["slices_unseen"] = _slices(unseen, unseen_events, ctx.classes, ref)

    metrics = {
        "model": meta["name"],
        "split_version": meta["split_version"],
        "classes": ctx.classes,
        "reference_thresholds": {"empty_filter": ref.empty, "species_accept": ref.species},
        "groups": results,
        "hardware": meta["hardware"],
        "extraction": meta["extraction"],
        "latency": [
            _benchmark(ctx, predictor_for(ctx, model_dir, d), rows)
            for d in dict.fromkeys([device, "cpu"])
        ]
        if benchmark
        else None,
    }
    _write_examples(model_dir / "eval", scored, events, ref)

    from wildinbox.evaluation.report import write_report

    write_report(report_dir, metrics, meta, scored, events, ctx, ecfg)
    (report_dir / "metrics.json").write_text(json.dumps(_round(metrics), indent=2) + "\n")
    _log_mlflow(meta, metrics, report_dir)
    return metrics


def _write_examples(
    out: Path, scored: list[Scored], events: dict[str, Any], ref: Thresholds
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out / "predictions.jsonl.gz", "wt") as f:
        for s in scored:
            f.write(
                json.dumps(
                    {
                        "source_id": s.row.source_id,
                        "event_id": s.row.event_id,
                        "partition": s.row.partition.value,
                        "camera_id": s.row.camera_id,
                        "image_label": s.row.image_label,
                        "event_role": s.row.event_role,
                        "predicted": s.pred,
                        "confidence": round(s.confidence, 6),
                        "probabilities": {k: round(v, 6) for k, v in s.probs.items()},
                        "error_type": s.error_type(),
                        "night": s.night,
                        "blur": round(s.blur, 2),
                        "max_box_area": s.row.max_box_area,
                    }
                )
                + "\n"
            )
    from wildinbox.policy.conservative import decide_event

    by_image = {s.row.source_id: s for s in scored}
    with gzip.open(out / "events.jsonl.gz", "wt") as f:
        for ev in events.values():
            frames = [by_image[i] for i in ev.image_ids if i in by_image]
            if not frames:
                continue
            o = decide_event([fr.probs for fr in frames], ref)
            f.write(
                json.dumps(
                    {
                        "event_id": ev.event_id,
                        "partition": ev.partition.value,
                        "camera_id": ev.camera_id,
                        "role": ev.role,
                        "label": ev.label,
                        "animal_present": ev.animal_present,
                        "disposition": o.disposition.value,
                        "suggested": o.label,
                        "confidence": o.confidence,
                        "reasons": [r.value for r in o.reasons],
                        "false_empty": ev.animal_present and o.disposition.value == "likely_empty",
                        "frames": [
                            {
                                "source_id": fr.row.source_id,
                                "p_empty": round(fr.probs[EMPTY_CLASS], 6),
                                "predicted": fr.pred,
                            }
                            for fr in frames
                        ],
                    }
                )
                + "\n"
            )


def _round(x: Any) -> Any:
    if isinstance(x, float):
        return round(x, 6)
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_round(v) for v in x]
    return x


def compare(new: dict[str, Any], ref: dict[str, Any], tol: Any) -> list[str]:
    """Differences beyond the documented tolerance (empty list = reproduced)."""
    problems = []
    for group in ref["groups"]:
        if group.startswith("slices"):
            continue
        a, b = new["groups"][group], ref["groups"][group]
        d = abs(a["image"]["macro_f1"] - b["image"]["macro_f1"])
        if d > tol.macro_f1:
            problems.append(f"{group} macro-F1 differs by {d:.4f} (> {tol.macro_f1})")
        for c, pc in b["image"]["per_class"].items():
            d = abs(a["image"]["per_class"][c]["recall"] - pc["recall"])
            if d > tol.per_class_recall:
                problems.append(f"{group} {c} recall differs by {d:.4f} (> {tol.per_class_recall})")
        for x, y in zip(a["event_empty_sweep"], b["event_empty_sweep"], strict=True):
            d = abs(x["false_empty_rate"] - y["false_empty_rate"])
            if d > tol.false_empty_rate:
                problems.append(
                    f"{group} false-empty rate at t={y['thresholds']['empty_filter']} "
                    f"differs by {d:.4f} (> {tol.false_empty_rate})"
                )
    return problems


def _log_mlflow(meta: dict[str, Any], metrics: dict[str, Any], report_dir: Path) -> None:
    run_id = meta.get("mlflow_run_id")
    if not run_id:
        return
    import mlflow

    from wildinbox.training.run import MLFLOW_URI

    mlflow.set_tracking_uri(MLFLOW_URI)
    u = metrics["groups"][UNSEEN]
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(
            {
                "unseen_macro_f1": u["image"]["macro_f1"],
                "unseen_min_species_recall": u["image"]["min_species_recall"],
                "unseen_false_empty_rate_ref": u["event_reference"]["false_empty_rate"],
                "seen_macro_f1": metrics["groups"]["seen_camera_diagnostic"]["image"]["macro_f1"],
            }
        )
        mlflow.log_artifact(str(report_dir / "README.md"))


def evaluate_cli(args: argparse.Namespace) -> int:
    from wildinbox.training.spec import load_baseline_config

    metrics = evaluate(args.model, args.config, args.report_dir, benchmark=not args.no_benchmark)
    u = metrics["groups"][UNSEEN]
    print(
        f"unseen cameras: macro-F1 {u['image']['macro_f1']:.3f}, min species recall "
        f"{u['image']['min_species_recall']:.3f}, false-empty rate "
        f"{u['event_reference']['false_empty_rate']:.4f} at reference thresholds"
    )
    print(f"report -> {args.report_dir / 'README.md'}")
    if args.compare_to:
        ref = json.loads(Path(args.compare_to).read_text())
        tol = load_baseline_config(args.config).reproducibility
        problems = compare(_round(metrics), ref, tol)
        if problems:
            print("NOT reproduced:\n  " + "\n  ".join(problems))
            return 1
        print(f"reproduced {args.compare_to} within tolerance {tol.model_dump()}")
    return 0


def evaluate_partitions(names: Sequence[str]) -> list[Partition]:
    from wildinbox.evaluation.data import assert_development

    return assert_development(names)
