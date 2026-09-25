"""Markdown report for `wildinbox calibrate`."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Rows shown in the trade-off tables; every grid point is in metrics.json.
SHOWN = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999)


def _t(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _pct(x: float | None, d: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{d}f}%"


def _ci(ci: Sequence[float] | None) -> str:
    return "n/a" if ci is None else f"[{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"


def _thr(x: float | None) -> str:
    return "disabled" if x is None else f"{x:g}"


def _shown(sweep: list[dict[str, Any]], key: str, chosen: float | None) -> list[dict[str, Any]]:
    return [m for m in sweep if m["thresholds"][key] in SHOWN or m["thresholds"][key] == chosen]


def write_report(report_dir: Path, r: dict[str, Any]) -> Path:
    cal, op = r["calibration"], r["operating_point"]
    tg = op["targets"]
    fit, choose = cal["fit_partition"], op["chosen_on"]
    m = op["metrics"]
    md = [
        f"# Calibration and operating point: `{r['model']}`",
        "",
        f"Rule: [`{r['rule']}`](../../{r['rule']}), committed before any calibrated "
        f"{choose.replace('_', '-')} result was computed. Development partitions only; "
        "**the locked final test was not opened.**",
        "",
        "## Decision",
        "",
        _t(
            ["", "Threshold", "Automation"],
            [
                [
                    "Empty filter (every usable frame P(empty) >=)",
                    _thr(op["empty_filter"]),
                    "enabled" if op["auto_filter_enabled"] else "**disabled**",
                ],
                [
                    "Species acceptance (mean P(species) >=)",
                    _thr(op["species_accept"]),
                    "enabled" if op["auto_accept_enabled"] else "**disabled**",
                ],
            ],
        ),
        "",
        f"A threshold is enabled only if the worst case of its 95% Wilson interval on "
        f"{choose.replace('_', ' ')} meets the target: false-empty rate <= "
        f"{_pct(tg['max_false_empty_rate'], 0)} of animal-containing events, accepted-label "
        f"precision >= {_pct(tg['min_accepted_precision'], 0)}. The lowest passing threshold "
        "on the grid is chosen; a disabled step sends those events to review.",
        "",
        f"At this operating point on {choose.replace('_', ' ')}:",
        "",
        _t(
            ["Measure", "Value"],
            [
                [
                    "Animal events filtered as empty",
                    f"{m['false_empty']} / {m['animal_events']} "
                    f"({_pct(m['false_empty_rate'], 2)} {_ci(m['false_empty_rate_ci95'])})",
                ],
                ["Events filtered", f"{m['filtered']} / {m['events']}"],
                [
                    "Labels accepted automatically",
                    f"{m['accepted']} / {m['events']}, precision "
                    f"{_pct(m['accepted_precision'])} {_ci(m['accepted_precision_ci95'])}",
                ],
                [
                    "Unsupported animals accepted as a known species",
                    m["unsupported_accepted_as_known"],
                ],
                ["Events left for review", f"{m['needs_review']} ({_pct(m['review_fraction'])})"],
            ],
        ),
        "",
        "## Calibration",
        "",
        f"Temperature scaling, fitted by NLL on the **{fit}** partition only "
        f"({cal['fit_images']} supported images): **T = {cal['temperature']:.3f}**. "
        + (
            "T > 1 means the raw scores were overconfident. "
            if cal["temperature"] > 1
            else "T < 1 means the raw scores were underconfident. "
        )
        + "Temperature scaling never changes the predicted class, so macro-F1 and "
        "per-class recall are unchanged. Only the supported images enter NLL and ECE; "
        "unsupported animals count as not-empty in the P(empty) table.",
        "",
        _t(
            ["Partition", "Role", "NLL raw", "NLL calibrated", "ECE raw", "ECE calibrated"],
            [
                [
                    p,
                    "fit" if p == fit else "held out",
                    f"{s['nll']['raw']:.3f}",
                    f"{s['nll']['calibrated']:.3f}",
                    f"{s['top_label_ece']['raw']:.3f}",
                    f"{s['top_label_ece']['calibrated']:.3f}",
                ]
                for p, s in cal["summary"].items()
            ],
        ),
        "",
        f"P(empty) reliability on {choose.replace('_', ' ')} (held out): how often images "
        "scored in each band are really empty. The empty filter relies on the top bands.",
        "",
        _t(
            [
                "P(empty) band",
                "Images",
                "Mean P(empty) raw",
                "Empty (raw)",
                "Images",
                "Mean P(empty) calibrated",
                "Empty (calibrated)",
            ],
            [
                [
                    f"{a['band'][0]:g}-{a['band'][1]:g}",
                    a["images"],
                    "n/a" if a["mean_p_empty"] is None else f"{a['mean_p_empty']:.3f}",
                    f"{_pct(a['observed_empty'])} {_ci(a['observed_empty_ci95'])}",
                    b["images"],
                    "n/a" if b["mean_p_empty"] is None else f"{b['mean_p_empty']:.3f}",
                    f"{_pct(b['observed_empty'])} {_ci(b['observed_empty_ci95'])}",
                ]
                for a, b in zip(
                    cal["summary"][choose]["empty_reliability"]["raw"],
                    cal["summary"][choose]["empty_reliability"]["calibrated"],
                    strict=True,
                )
            ],
        ),
        "",
        "## Review budget trade-off",
        "",
        f"Calibrated scores, {choose.replace('_', ' ')}, conservative policy. **Pass** means "
        "the threshold meets its target at the worst case of the 95% interval; operating "
        "points that fail are shown so no setting is implied to be safe when it is not.",
        "",
        "Empty filter (species acceptance disabled):",
        "",
        _t(
            [
                "P(empty) >=",
                "Filtered",
                "True empty filtered",
                "Animal events filtered (95% CI)",
                "Review",
                "Pass",
            ],
            [
                [
                    f"{e['thresholds']['empty_filter']:g}",
                    f"{e['filtered']} / {e['events']}",
                    f"{e['filtered_true_empty']} / {e['empty_events']}",
                    f"{e['false_empty']} ({_pct(e['false_empty_rate'], 2)} "
                    f"{_ci(e['false_empty_rate_ci95'])})",
                    _pct(e["review_fraction"]),
                    "**pass**" if e["passes"] else "fail",
                ]
                for e in _shown(r["sweeps"]["empty_filter"], "empty_filter", op["empty_filter"])
            ],
        ),
        "",
        f"Species acceptance (empty filter {_thr(op['empty_filter'])}):",
        "",
        _t(
            [
                "Species >=",
                "Accepted",
                "Precision (95% CI)",
                "Unsupported accepted as known",
                "Review",
                "Pass",
            ],
            [
                [
                    f"{s['thresholds']['species_accept']:g}",
                    f"{s['accepted']} / {s['events']}",
                    f"{_pct(s['accepted_precision'])} {_ci(s['accepted_precision_ci95'])}",
                    s["unsupported_accepted_as_known"],
                    _pct(s["review_fraction"]),
                    "**pass**" if s["passes"] else "fail",
                ]
                for s in _shown(
                    r["sweeps"]["species_accept"], "species_accept", op["species_accept"]
                )
            ],
        ),
        "",
        "## Same thresholds on the calibration cameras (context only)",
        "",
        _t(
            ["Measure", "Value"],
            [
                [
                    "Animal events filtered as empty",
                    f"{op['metrics_on_fit_partition']['false_empty']} / "
                    f"{op['metrics_on_fit_partition']['animal_events']}",
                ],
                [
                    "Labels accepted (precision)",
                    f"{op['metrics_on_fit_partition']['accepted']} "
                    f"({_pct(op['metrics_on_fit_partition']['accepted_precision'])})",
                ],
                ["Events left for review", _pct(op["metrics_on_fit_partition"]["review_fraction"])],
            ],
        ),
        "",
        "## Limitations",
        "",
        "- Thresholds rest on 2 policy-validation cameras (90, 125); calibration on 2 others "
        "(51, 108). New cameras may behave differently; start them conservatively and audit.",
        "- Uncalibrated sweeps on these development cameras were published in Stage 7 before "
        "this rule was written.",
        "- Choosing the lowest passing threshold on the same data is mildly optimistic; the "
        "final test measures the frozen operating point once.",
        "- The unfamiliar-input score is not part of this policy yet. When it is added, the "
        "operating point is re-chosen with this same rule.",
        "",
        "## Reproduce",
        "",
        "```bash",
        f"uv run wildinbox calibrate --rule {r['rule']}",
        "```",
        "",
        f"Code `{r['code']['commit']}`{' (uncommitted changes)' if r['code']['dirty'] else ''}; "
        f"split `{r['split_version']}`.",
        "",
    ]
    out = report_dir / "README.md"
    out.write_text("\n".join(md))
    return out
