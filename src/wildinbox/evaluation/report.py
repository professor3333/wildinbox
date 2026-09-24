"""Baseline report: headline metrics, trade-offs, slices, and an error gallery."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageDraw, ImageOps

from wildinbox.class_map import EMPTY_CLASS

if TYPE_CHECKING:
    from wildinbox.evaluation.run import Scored
    from wildinbox.training.run import Context

THUMB = (220, 160)
CAPTION = 34


def _t(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _p(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def _ci(ci: Sequence[float] | None) -> str:
    return "" if not ci else f" [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"


def _sheet(items: list[tuple[Path, list[str]]], out: Path, cols: int = 4) -> None:
    w, h = THUMB
    rows = max(1, -(-len(items) // cols))
    sheet = Image.new("RGB", (cols * w, rows * (h + CAPTION)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (path, lines) in enumerate(items):
        x, y = (i % cols) * w, (i // cols) * (h + CAPTION)
        try:
            with Image.open(path) as img:
                sheet.paste(ImageOps.fit(img.convert("RGB"), THUMB), (x, y))
        except OSError:
            draw.rectangle([x, y, x + w - 1, y + h - 1], fill="#ddd")
        for j, line in enumerate(lines[:2]):
            draw.text((x + 3, y + h + 2 + 15 * j), line[:36], fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, "JPEG", quality=82)


def _gallery(
    report_dir: Path, scored: list[Scored], ctx: Context, ecfg: Any, unseen: set[Any]
) -> list[str]:
    pool = [s for s in scored if s.row.partition in unseen]
    n = ecfg.gallery_size
    blur_cut = sorted(s.blur for s in pool)[len(pool) // 10] if pool else 0.0
    groups: list[tuple[str, str, list[Scored]]] = [
        (
            "false_empty",
            "Animals predicted empty (highest P(empty) first). These frames "
            "are what the empty filter would drop.",
            sorted(
                [s for s in pool if s.error_type() == "false_empty"],
                key=lambda s: -s.probs[EMPTY_CLASS],
            ),
        ),
        (
            "confused_species",
            "Supported species predicted as another species (most confident mistakes first).",
            sorted(
                [s for s in pool if s.error_type() == "confused_species"],
                key=lambda s: -s.confidence,
            ),
        ),
        (
            "small_animals",
            f"Misclassified animals whose largest box covers < "
            f"{100 * ecfg.small_animal_area:.0f}% of the frame.",
            sorted(
                [
                    s
                    for s in pool
                    if s.error_type()
                    and s.row.max_box_area is not None
                    and s.row.max_box_area < ecfg.small_animal_area
                    and s.is_animal
                ],
                key=lambda s: s.row.max_box_area or 0,
            ),
        ),
        (
            "blurred",
            f"Misclassified images in the blurriest 10% (variance of Laplacian < {blur_cut:.0f}).",
            sorted(
                [
                    s
                    for s in pool
                    if s.error_type() in ("false_empty", "confused_species", "false_animal")
                    and s.blur < blur_cut
                ],
                key=lambda s: s.blur,
            ),
        ),
        (
            "unfamiliar_inputs",
            "Unsupported animals labeled as a supported species "
            "(most confident first). Confidence alone does not reject them.",
            sorted(
                [s for s in pool if s.error_type() == "unfamiliar_predicted_known"],
                key=lambda s: -s.confidence,
            ),
        ),
    ]
    md = [
        "## Error gallery",
        "",
        "From the unseen-camera development partitions. Captions: true label -> predicted "
        "(confidence), camera, night flag.",
        "",
    ]
    for name, text, items in groups:
        seen_events: set[str] = set()
        pick = []
        for s in items:  # one frame per event, so a single burst can't fill a sheet
            if s.row.event_id not in seen_events:
                seen_events.add(s.row.event_id)
                pick.append(s)
            if len(pick) == n:
                break
        md += [
            f"### {name.replace('_', ' ').capitalize()} ({len(items)} images, "
            f"{len({s.row.event_id for s in items})} events)",
            "",
            text,
            "",
        ]
        if not pick:
            md += ["None.", ""]
            continue
        _sheet(
            [
                (
                    ctx.images_root / s.row.storage_path,
                    [
                        f"{s.row.image_label} -> {s.pred} ({s.confidence:.2f})",
                        f"cam {s.row.camera_id}{' night' if s.night else ''}"
                        + (f" box {100 * s.row.max_box_area:.1f}%" if s.row.max_box_area else ""),
                    ],
                )
                for s in pick
            ],
            report_dir / "gallery" / f"{name}.jpg",
        )
        md += [
            f"![{name}](gallery/{name}.jpg)",
            "",
            "<details><summary>Examples</summary>",
            "",
            _t(
                [
                    "Image",
                    "Event",
                    "Camera",
                    "True",
                    "Predicted",
                    "Conf.",
                    "Night",
                    "Blur",
                    "Box area",
                ],
                [
                    [
                        s.row.source_id[:8],
                        s.row.event_id.split(":")[-1][:8],
                        s.row.camera_id,
                        s.row.image_label,
                        s.pred,
                        f"{s.confidence:.2f}",
                        "yes" if s.night else "",
                        f"{s.blur:.0f}",
                        _p(s.row.max_box_area),
                    ]
                    for s in pick
                ],
            ),
            "",
            "</details>",
            "",
        ]
    return md


def write_report(
    report_dir: Path,
    metrics: dict[str, Any],
    meta: dict[str, Any],
    scored: list[Scored],
    events: dict[str, Any],
    ctx: Context,
    ecfg: Any,
) -> Path:
    from wildinbox.evaluation.run import UNSEEN

    g = metrics["groups"]
    u = g[UNSEEN]
    ref = u["event_reference"]
    classes = metrics["classes"]
    unseen_parts = set(ecfg.unseen_camera_partitions)
    md: list[str] = [
        f"# Baseline report: `{meta['name']}`",
        "",
        "Frozen EfficientNet-B0 (ImageNet) embeddings + class-weighted logistic regression. "
        "Development partitions only; **the locked final test was not opened.**",
        "",
        "## Headline (unseen cameras: calibration + policy validation)",
        "",
        _t(
            ["Metric", "Value"],
            [
                ["Macro-F1, supported classes (image level)", f"{u['image']['macro_f1']:.3f}"],
                ["Lowest per-species recall", f"{u['image']['min_species_recall']:.3f}"],
                [
                    "Animal events wrongly filtered as empty, at reference thresholds",
                    f"{ref['false_empty']} of {ref['animal_events']} "
                    f"({_p(ref['false_empty_rate'], 2)}{_ci(ref['false_empty_rate_ci95'])})",
                ],
                ["Events left for review at reference thresholds", _p(ref["review_fraction"])],
            ],
        ),
        "",
        f"Reference thresholds P(empty) >= {ref['thresholds']['empty_filter']}, species >= "
        f"{ref['thresholds']['species_accept']} are **reference points, not chosen operating "
        "points**; scores are uncalibrated. Thresholds are chosen later on policy validation.",
        "",
        f"Overall accuracy ({_p(u['image']['accuracy_context_only'])}) is **not** a headline "
        "metric: a model can score well by predicting the most common classes while missing "
        "animals. Macro-F1 weights every class equally; the false-empty count measures the "
        "failure that matters most.",
        "",
    ]

    # Seen vs unseen
    md += [
        "## Seen vs unseen cameras",
        "",
        "Same model; the diagnostic partition holds out sequences from *training* cameras.",
        "",
        _t(
            [
                "Partition",
                "Cameras",
                "Images",
                "Macro-F1",
                "Min species recall",
                "Animal image recall",
                "False-empty events (ref)",
            ],
            [
                [
                    name,
                    "unseen"
                    if name == UNSEEN or name in {p.value for p in unseen_parts}
                    else "seen (training)",
                    g[name]["image"]["n"],
                    f"{g[name]['image']['macro_f1']:.3f}",
                    f"{g[name]['image']['min_species_recall']:.3f}",
                    _p(g[name]["image"]["animal_image_recall"]),
                    f"{g[name]['event_reference']['false_empty']} / "
                    f"{g[name]['event_reference']['animal_events']}",
                ]
                for name in [UNSEEN, *[p.value for p in ecfg.partitions]]
            ],
        ),
        "",
    ]

    # Per class + confusion
    pc = u["image"]["per_class"]
    md += [
        "## Image level, unseen cameras",
        "",
        _t(
            ["Class", "Precision", "Recall", "F1", "Support", "Predicted"],
            [
                [
                    c,
                    f"{pc[c]['precision']:.3f}",
                    f"{pc[c]['recall']:.3f}",
                    f"{pc[c]['f1']:.3f}",
                    pc[c]["support"],
                    pc[c]["predicted"],
                ]
                for c in classes
            ],
        ),
        "",
        "Confusion matrix (rows = true, columns = predicted):",
        "",
        _t(
            ["", *classes],
            [[c, *row] for c, row in zip(classes, u["image"]["confusion"]["matrix"], strict=True)],
        ),
        "",
    ]
    un = u["unsupported_images"]
    md += [
        f"**Unsupported animals** ({un['n']} images of species outside the supported set): "
        f"{un['predicted_known_species_at_reference']} were given a supported species with "
        f"confidence >= {ref['thresholds']['species_accept']}. Predicted as: "
        + ", ".join(f"{k} {v}" for k, v in un["predicted"].items())
        + ".",
        "",
    ]

    # Event trade-offs
    md += [
        "## Event-level trade-off, unseen cameras",
        "",
        "Conservative policy: filter only if **every** frame has P(empty) >= threshold; any "
        "animal-looking frame keeps the event; species accepted only if animal frames agree "
        "and their mean probability >= species threshold.",
        "",
        f"Empty threshold sweep (species threshold {ref['thresholds']['species_accept']}):",
        "",
        _t(
            [
                "P(empty) >=",
                "Filtered",
                "True empty filtered",
                "Animal events filtered (95% CI)",
                "Retention",
                "Review",
            ],
            [
                [
                    m["thresholds"]["empty_filter"],
                    f"{m['filtered']} / {m['events']}",
                    f"{m['filtered_true_empty']} / {m['empty_events']}",
                    f"{m['false_empty']} ({_p(m['false_empty_rate'], 2)}"
                    f"{_ci(m['false_empty_rate_ci95'])})",
                    _p(m["animal_event_retention"], 2),
                    _p(m["review_fraction"]),
                ]
                for m in u["event_empty_sweep"]
            ],
        ),
        "",
        f"Species threshold sweep (empty threshold {ref['thresholds']['empty_filter']}):",
        "",
        _t(
            [
                "Species >=",
                "Accepted",
                "Precision (95% CI)",
                "Wrong by true role",
                "Unsupported accepted as known",
                "Review",
            ],
            [
                [
                    m["thresholds"]["species_accept"],
                    f"{m['accepted']} / {m['events']}",
                    f"{_p(m['accepted_precision'])}{_ci(m['accepted_precision_ci95'])}",
                    ", ".join(f"{k} {v}" for k, v in m["accepted_wrong_by_true_role"].items())
                    or "-",
                    m["unsupported_accepted_as_known"],
                    _p(m["review_fraction"]),
                ]
                for m in u["event_species_sweep"]
            ],
        ),
        "",
    ]

    # Slices
    s = g["slices_unseen"]
    md += [
        "## Slices, unseen cameras",
        "",
        "Event false-empty counts at the reference thresholds. Night = most frames are "
        "grayscale infrared.",
        "",
        _t(
            ["Camera", "Images", "Macro-F1", "Animal image recall", "False-empty events"],
            [
                [
                    c,
                    v["images"],
                    f"{v['macro_f1']:.3f}",
                    _p(v["animal_image_recall"]),
                    f"{v['false_empty']} / {v['animal_events']}",
                ]
                for c, v in s["camera"].items()
            ],
        ),
        "",
        _t(
            ["Condition", "Images", "Macro-F1", "Animal image recall", "False-empty events"],
            [
                [
                    c,
                    v["images"],
                    f"{v['macro_f1']:.3f}",
                    _p(v["animal_image_recall"]),
                    f"{v['false_empty']} / {v['animal_events']}",
                ]
                for c, v in s["day_night"].items()
            ],
        ),
        "",
    ]

    # False-empty explanation
    from wildinbox.policy.conservative import Thresholds, decide_event

    t = Thresholds(ref["thresholds"]["empty_filter"], ref["thresholds"]["species_accept"])
    by_image = {x.row.source_id: x for x in scored}
    fe_rows = []
    for ev in events.values():
        if ev.partition not in unseen_parts or not ev.animal_present:
            continue
        frames = [by_image[i] for i in ev.image_ids if i in by_image]
        if (
            frames
            and decide_event([f.probs for f in frames], t).disposition.value == "likely_empty"
        ):
            fe_rows.append(
                [
                    ev.event_id.split(":")[-1][:8],
                    ev.camera_id,
                    ev.label or "mixed",
                    ev.role,
                    ", ".join(f"{f.probs[EMPTY_CLASS]:.3f}" for f in frames),
                ]
            )
    md += [
        "## Why each false-empty event was filtered",
        "",
        "Every animal event the reference policy would drop, with each frame's P(empty); "
        "all frames were above the threshold, which is the only way the conservative "
        "policy filters an event.",
        "",
    ]
    md += [
        _t(["Event", "Camera", "True label", "Role", "Frame P(empty)"], fe_rows[:40])
        if fe_rows
        else "None at the reference thresholds.",
        "",
    ]
    if len(fe_rows) > 40:
        md += [f"...and {len(fe_rows) - 40} more in `eval/events.jsonl.gz`.", ""]

    # Hardware
    hw, ex, lat = metrics["hardware"], metrics["extraction"], metrics["latency"]
    md += [
        "## Inference time and memory",
        "",
        f"Declared hardware: {hw.get('cpu', hw['machine'])}, "
        f"{hw.get('memory_gb', 0):.0f} GB RAM, {hw['platform']}, torch {hw['torch']} "
        f"({hw['torch_threads']} threads), device `{meta['device']}`.",
        "",
        _t(
            ["Stage", "Images", "Seconds", "Images/s", "Peak RSS, main process (MB)"],
            [
                [k, v["images"], v["seconds"], v["images_per_second"], v["peak_rss_mb"]]
                for k, v in ex.items()
            ],
        ),
        "",
    ]
    if lat:
        md += [
            "Single image end to end (decode, preprocess, embed, classify; batch size 1, "
            "after warm-up). The Docker deployment runs on CPU.",
            "",
            _t(
                ["Device", "Images", "p50 ms", "p95 ms", "Peak RSS, main process (MB)"],
                [
                    [
                        x["device"],
                        x["images"],
                        f"{x['p50_ms']:.0f}",
                        f"{x['p95_ms']:.0f}",
                        f"{x['peak_rss_mb']:.0f}",
                    ]
                    for x in lat
                ],
            ),
            "",
        ]

    md += _gallery(report_dir, scored, ctx, ecfg, unseen_parts)

    tr = meta["train_images_per_class"]
    md += [
        "## Setup and reproduction",
        "",
        _t(
            ["", ""],
            [
                [
                    "Training images (use_for_fit)",
                    f"{meta['train_images']}: " + ", ".join(f"{k} {v}" for k, v in tr.items()),
                ],
                [
                    "Backbone",
                    "torchvision EfficientNet-B0, IMAGENET1K_V1, frozen (ImageNet-1k "
                    "pretraining; no wildlife-specific checkpoint, so no overlap with CCT20)",
                ],
                [
                    "Classifier",
                    f"logistic regression, C={meta['config']['classifier']['c']}, "
                    f"class_weight={meta['config']['classifier']['class_weight']}",
                ],
                [
                    "Split / inventory",
                    f"`{meta['split_version']}` / `{meta['inventory_manifest_version']}`",
                ],
                ["Preprocessing", f"`{meta['preprocessing_version']}`"],
                ["Seed", meta["seed"]],
                [
                    "Code",
                    f"`{meta['code']['commit']}`"
                    + (" (uncommitted changes)" if meta["code"]["dirty"] else ""),
                ],
                ["MLflow run", f"`{meta.get('mlflow_run_id')}`"],
            ],
        ),
        "",
        "```bash",
        "uv run wildinbox baseline train          # embed + fit",
        "uv run wildinbox evaluate                # this report + metrics.json",
        "uv run wildinbox baseline train --no-cache --models-dir /tmp/fresh",
        "uv run wildinbox evaluate --model /tmp/fresh/"
        + meta["name"]
        + " --report-dir /tmp/fresh-report --compare-to reports/baseline/metrics.json",
        "```",
        "",
        "Tolerances (absolute): "
        + ", ".join(f"{k} {v}" for k, v in meta["config"]["reproducibility"].items())
        + ".",
        "",
        "## Limitations",
        "",
        "- Scores are uncalibrated; thresholds above are reference points only.",
        "- Unseen-camera results rest on 4 development cameras; see the per-camera table.",
        "- CCT20 is ~7% empty, so empty-filtering volume here understates a real memory card.",
        "- Day/night is inferred from grayscale (infrared) frames, not from timestamps.",
        "",
    ]
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "README.md"
    path.write_text("\n".join(md))
    return path
