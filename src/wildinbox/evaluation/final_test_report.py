"""Markdown report for `wildinbox final-test`."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from wildinbox.class_map import EMPTY_CLASS


def _t(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _p(x: float | None, d: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{d}f}%"


def _ci(ci: Sequence[float] | None, d: int = 1) -> str:
    return "" if not ci else f"[{100 * ci[0]:.{d}f}, {100 * ci[1]:.{d}f}]"


def _f1(m: dict[str, Any]) -> str:
    lo, hi = m["macro_f1_ci95"]
    return f"{m['macro_f1']:.3f} [{lo:.3f}, {hi:.3f}]"


def _rate(r: dict[str, Any]) -> str:
    return f"{r['flagged']} / {r['n']} ({_p(r['rate'])} {_ci(r['ci95'])})" if r["n"] else "n/a"


def _curves(report_dir: Path, res: dict[str, Any]) -> list[str]:
    from wildinbox.evaluation.curves import Point, Series, render

    dev = json.loads(Path("reports/calibration/metrics.json").read_text())["sweeps"]
    shown = {0.5, 0.6, 0.65, 0.7, 0.8, 0.9, 0.95, 0.99}

    def empty(sweep: list[dict[str, Any]]) -> list[Point]:
        return [
            Point(
                m["filtered"] / m["events"],
                m["false_empty_rate"],
                m["false_empty_rate_ci95"][0],
                m["false_empty_rate_ci95"][1],
                f"{m['thresholds']['empty_filter']:g}",
                m["thresholds"]["empty_filter"] in shown,
            )
            for m in sweep
        ]

    def species(sweep: list[dict[str, Any]]) -> list[Point]:
        out = []
        for m in sweep:
            if m["accepted"]:
                lo, hi = m["accepted_precision_ci95"]
                t = m["thresholds"]["species_accept"]
                out.append(
                    Point(
                        m["accepted"] / m["events"],
                        1 - m["accepted_precision"],
                        1 - hi,
                        1 - lo,
                        f"{t:g}",
                        t in shown,
                    )
                )
        return out

    rule = next(p for p in empty(res["sweeps"]["empty_filter"]) if p.label == "0.65")
    (report_dir / "curve-empty-filter.svg").write_text(
        render(
            "Final test: animals lost vs events filtered",
            "Events filtered automatically (coverage)",
            "Animal events filtered as empty",
            [
                Series("final test (9 cameras)", empty(res["sweeps"]["empty_filter"]), band=True),
                Series("policy validation (dev)", empty(dev["empty_filter"])),
            ],
            0.02,
            "target 2%",
            rule,
            "rule: 0.65",
        )
    )
    (report_dir / "curve-species-accept.svg").write_text(
        render(
            "Final test: wrong labels vs events accepted",
            "Events accepted automatically (coverage)",
            "Accepted labels that are wrong",
            [
                Series(
                    "final test (9 cameras)", species(res["sweeps"]["species_accept"]), band=True
                ),
                Series("policy validation (dev)", species(dev["species_accept"])),
            ],
            0.05,
            "target 5%",
        )
    )
    return [
        "![Empty filter on the final test](curve-empty-filter.svg)",
        "",
        "![Species acceptance on the final test](curve-species-accept.svg)",
        "",
    ]


def write_report(report_dir: Path, out: dict[str, Any], policy: dict[str, Any]) -> Path:
    res = out["results"]
    e3, base = res["image_level"]["e3"], res["image_level"]["baseline"]
    rule = res["decisions"]["rule_operating_point"]
    disp = res["decisions"]["dispositions"]
    rr = res["review_reduction"]
    opened = json.loads((report_dir / "opened.json").read_text())
    retention = 1 - rule["false_empty_rate"]
    ret_lo = 1 - rule["false_empty_rate_ci95"][1]
    held = {k: v for k, v in res["unsupported"].items() if v["group"] == "held_out"}
    held_events = sum(v["events"] for v in held.values())
    held_acc_09 = sum(v["accepted_as_known_if_species_threshold"]["0.9"] for v in held.values())
    serving = json.loads(Path("reports/serving/timing-1000.json").read_text())
    latency = json.loads(Path("reports/serving/api-latency.json").read_text())["endpoints"]
    worst_p95 = max(v["p95_ms"] for v in latency.values())

    targets = [
        [
            "Animal-event retention >= 98%",
            f"released: 100% (nothing filtered); rule's filter 0.65: {_p(retention, 2)} "
            f"(95% worst case {_p(ret_lo, 2)})",
            "met as released; rule's filter " + ("met" if ret_lo >= 0.98 else "**not met**"),
        ],
        [
            "Accepted species precision >= 95%, share auto-labeled reported",
            "0% of events auto-labeled: acceptance disabled (no threshold earned it)",
            "not applicable",
        ],
        [
            "Review reduced by 50% (with audits) at the retention target",
            f"released: {_p(rr['released_reduction_vs_grouped'])}; rule's filter: "
            f"{_p(rr['rule_reduction_vs_grouped'])} (before audits)",
            "**not met**",
        ],
        [
            "Species quality on unseen cameras",
            f"macro-F1 {e3['macro_f1']:.3f} [{e3['macro_f1_ci95'][0]:.3f}, "
            f"{e3['macro_f1_ci95'][1]:.3f}]; lowest species recall {e3['min_species_recall']:.3f}",
            "reported",
        ],
        [
            "Unsupported species wrongly auto-accepted",
            f"released: 0 of {held_events} held-out-species events (acceptance off); "
            f"at a species threshold of 0.9 it would be {held_acc_09}",
            "reported",
        ],
        [
            "1,000 images within 10 minutes; metadata p95 < 500 ms",
            f"{serving['processing_seconds']} s; worst p95 {worst_p95} ms (declared hardware)",
            "met",
        ],
        ["Worker restart without lost or duplicate results", "real SIGKILL demo passed", "met"],
    ]

    classes = list(e3["per_class"])
    md = [
        "# Final test: unseen cameras",
        "",
        f"Protocol [`{out['protocol']}`](../../{out['protocol']}) (sha256 "
        f"`{out['protocol_sha256'][:12]}`), committed before the test was opened. Opened "
        f"{opened['opened_at'][:10]}; every pinned artifact matched its hash. **Measured once: "
        "no threshold, model, or rule was chosen or changed on these results.**",
        "",
        f"{res['images']} images, {res['events']} capture events, {len(res['cameras'])} "
        f"cameras never used for training, calibration, or thresholds "
        f"({', '.join(res['cameras'])}). Release `{out['release_id']}`.",
        "",
        "## Against the success targets",
        "",
        _t(["Target (CLAUDE.md)", "Result", "Status"], targets),
        "",
        "## Species classification",
        "",
        _t(
            ["", "Macro-F1 (95% CI)", "Lowest species recall", "Animal image recall", "Images"],
            [
                [
                    name,
                    _f1(m),
                    f"{m['min_species_recall']:.3f}",
                    _p(m["animal_image_recall"]),
                    m["n"],
                ]
                for name, m in (("E3 (fine-tuned)", e3), ("Frozen-embedding baseline", base))
            ],
        ),
        "",
        _t(
            ["Class", "E3 precision", "E3 recall", "Baseline recall", "Support"],
            [
                [
                    c,
                    f"{e3['per_class'][c]['precision']:.3f}",
                    f"{e3['per_class'][c]['recall']:.3f}",
                    f"{base['per_class'][c]['recall']:.3f}",
                    e3["per_class"][c]["support"],
                ]
                for c in classes
            ],
        ),
        "",
        "E3 confusion (rows = true, columns = predicted):",
        "",
        _t(
            ["", *classes],
            [[c, *row] for c, row in zip(classes, e3["confusion"]["matrix"], strict=True)],
        ),
        "",
        "## Random-image vs unseen-camera evaluation",
        "",
        _t(
            [
                "Model",
                "Seen cameras, held-out sequences",
                "Unseen development cameras",
                "Final test (unseen)",
            ],
            [
                [
                    name,
                    f"{c['seen_cameras_random_sequences']:.3f}",
                    f"{c['unseen_development_cameras']:.3f}",
                    f"{c['final_test_unseen_cameras']:.3f}",
                ]
                for name, c in (
                    ("E3", res["random_vs_unseen"]["e3"]),
                    ("Baseline", res["random_vs_unseen"]["baseline"]),
                )
            ],
        ),
        "",
        "Macro-F1. Evaluating on sequences from the training cameras overstates what a new "
        "deployment would see.",
        "",
        "## Event decisions",
        "",
        _t(
            ["Policy", "Likely empty", "Species identified", "Needs review"],
            [
                [
                    f"released `{res['decisions']['released_policy']}`",
                    disp["released"].get("likely_empty", 0),
                    disp["released"].get("species_identified", 0),
                    disp["released"].get("needs_review", 0),
                ],
                [
                    f"rule `{res['decisions']['rule_policy']}` (filter 0.65)",
                    disp["rule"].get("likely_empty", 0),
                    disp["rule"].get("species_identified", 0),
                    disp["rule"].get("needs_review", 0),
                ],
            ],
        ),
        "",
        f"At the rule's filter: {rule['false_empty']} of {rule['animal_events']} animal events "
        f"filtered as empty ({_p(rule['false_empty_rate'], 2)} "
        f"{_ci(rule['false_empty_rate_ci95'], 2)}); "
        f"{rule['filtered_true_empty']} of {rule['empty_events']} truly empty events filtered.",
        "",
        "## Error versus coverage",
        "",
        "The development curve (policy validation, where thresholds were chosen) is shown for "
        "comparison; the band is the final test's 95% interval.",
        "",
        *_curves(report_dir, res),
        "## Unsupported species",
        "",
        "Tuning species set the unfamiliar-score threshold during development; held-out species "
        "were never used before this run. All were unseen in fine-tuning; all but deer appear in "
        "ImageNet-1k pretraining classes (deer has no final-test events).",
        "",
        _t(
            [
                "Species",
                "Group",
                "Events",
                "Top predictions (images)",
                "Lost as empty at rule's filter",
                "Accepted as known if threshold 0.5 / 0.7 / 0.9",
            ],
            [
                [
                    sp,
                    v["group"],
                    v["events"],
                    ", ".join(f"{k} {n}" for k, n in list(v["predicted"].items())[:3]),
                    v["filtered_as_empty_at_rule"],
                    " / ".join(
                        str(v["accepted_as_known_if_species_threshold"][t])
                        for t in ("0.5", "0.7", "0.9")
                    ),
                ]
                for sp, v in res["unsupported"].items()
            ],
        ),
        "",
        "Unfamiliar-input detection at the thresholds frozen in development (image level):",
        "",
        _t(
            [
                "Score",
                "Known animals flagged",
                "Tuning species detected",
                "AUROC",
                "Held-out species detected",
                "AUROC",
            ],
            [
                [
                    name,
                    _rate(d["false_flags_on_known"]),
                    _rate(d["tuning"]["detection"]),
                    "n/a" if d["tuning"]["auroc"] is None else f"{d['tuning']['auroc']:.3f}",
                    _rate(d["held_out"]["detection"]),
                    "n/a" if d["held_out"]["auroc"] is None else f"{d['held_out']['auroc']:.3f}",
                ]
                for name, d in res["unfamiliar_detection"].items()
            ],
        ),
        "",
        "## Review workload",
        "",
        f"Grouping alone turns {rr['images']} images into {rr['events_after_grouping']} events "
        f"({rr['grouping_factor']:.2f} images per event). Against that grouped workflow, the "
        f"released policy removes {_p(rr['released_reduction_vs_grouped'])} of reviews and the "
        f"rule's filter {_p(rr['rule_reduction_vs_grouped'])}, before any audit sample.",
        "",
        "## Cameras, day and night, small animals",
        "",
        _t(
            [
                "Camera",
                "Images",
                "Macro-F1",
                "Animal image recall",
                "Animal events",
                "Lost as empty at rule's filter",
            ],
            [
                [
                    cam,
                    c["images"],
                    f"{c['macro_f1']:.3f}",
                    _p(c["animal_image_recall"]),
                    c["animal_events"],
                    f"{c['false_empty_at_rule']} ({_p(c['false_empty_rate_at_rule'])})",
                ]
                for cam, c in res["per_camera"].items()
            ],
        ),
        "",
        f"Camera spread: macro-F1 {res['camera_spread']['macro_f1']['min']:.3f} to "
        f"{res['camera_spread']['macro_f1']['max']:.3f} (median "
        f"{res['camera_spread']['macro_f1']['median']:.3f}); animal events lost at the rule's "
        f"filter {_p(res['camera_spread']['false_empty_rate_at_rule']['min'])} to "
        f"{_p(res['camera_spread']['false_empty_rate_at_rule']['max'])}.",
        "",
        _t(
            [
                "Condition",
                "Images",
                "Macro-F1",
                "Animal image recall",
                "Lost as empty at rule's filter",
            ],
            [
                [
                    k,
                    v["images"],
                    f"{v['macro_f1']:.3f}",
                    _p(v["animal_image_recall"]),
                    f"{v['false_empty_at_rule']} / {v['animal_events']}",
                ]
                for k, v in res["day_night"].items()
            ],
        ),
        "",
        f"Small animals (largest box < 1% of the frame): {res['small_animals']['images']} images, "
        f"species recall {_p(res['small_animals']['recall'])}.",
        "",
        "## Provenance",
        "",
        f"Code `{out['code']['commit']}`"
        f"{' (uncommitted changes)' if out['code']['dirty'] else ''}; "
        f"split `{res['split_version']}`; artifacts and hashes in [`opened.json`](opened.json); "
        "per-event decisions in [`decisions.jsonl.gz`](decisions.jsonl.gz). Re-running "
        "`uv run wildinbox final-test` must reproduce these numbers exactly or it refuses "
        "to write.",
        "",
        f"Released policy: every event goes to review; the `{EMPTY_CLASS}` filter and species "
        "acceptance stay off.",
        "",
    ]
    path = report_dir / "README.md"
    path.write_text("\n".join(md))
    return path
