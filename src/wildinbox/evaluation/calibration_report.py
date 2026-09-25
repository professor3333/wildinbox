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


def _rate(x: float | None, ci: Sequence[float] | None) -> str:
    return "n/a" if x is None else f"{_pct(x)} {_ci(ci)}"


def _thr(x: float | None) -> str:
    return "disabled" if x is None else f"{x:g}"


def _on(x: bool) -> str:
    return "enabled" if x else "disabled"


def _deviation(d: dict[str, Any] | None) -> list[str]:
    if not d:
        return []
    off = [
        name
        for name, flag in (
            ("automatic filtering", d["disable_auto_filter"]),
            ("automatic acceptance", d["disable_auto_accept"]),
        )
        if flag
    ]
    return [
        f"**Deviation from the rule ({d['decided']}):** {' and '.join(off)} released as "
        f"disabled. {d['reason']}",
        "",
    ]


def _policy_section(r: dict[str, Any]) -> list[str]:
    pol = r.get("policy")
    if not pol:
        return []
    unf = pol.get("unfamiliar")
    unf_text = (
        "not part of the rule"
        if not unf
        else (
            f"adopted: frames with distance > {unf['threshold']:.4f} are flagged "
            f"`possible_unknown` (version `{unf['version']}`)"
            if unf["adopted"]
            else f"evaluated, not adopted (version `{unf['version']}`); see "
            "[`reports/unfamiliar`](../unfamiliar/README.md)"
        )
    )
    return [
        "## Decision policy",
        "",
        _t(
            ["", "Value"],
            [
                [
                    "Policy",
                    f"`{pol['policy']}` ([source](../../src/wildinbox/policy/conservative.py))",
                ],
                ["Released policy version", f"`{pol['released']['policy_version']}`"],
                ["Rule's policy version (evaluated)", f"`{pol['rule']['policy_version']}`"],
                [
                    "Calibration",
                    f"temperature {pol['calibration']['temperature']:.3f}, "
                    f"version `{pol['calibration']['version']}`",
                ],
                ["Unfamiliar-input score", unf_text],
                ["Artifact", f"[`policy.json`](policy.json), version `{pol['artifact_version']}`"],
            ],
        ),
        "",
        "Every development event's saved frame predictions and both decisions are in "
        "[`decisions.jsonl.gz`](decisions.jsonl.gz); `uv run wildinbox replay` recomputes "
        "every disposition from them and the policy artifact (also run in CI). Incomplete "
        "or failed frames never allow filtering; reasons are machine-readable "
        "(`low_confidence`, `conflicting_frames`, `possible_unknown`, `processing_failure`, "
        "`species_not_validated`, `automation_disabled`).",
        "",
    ]


def _gate_section(r: dict[str, Any]) -> list[str]:
    op = r["operating_point"]
    if op.get("species_gate") is None:
        return []
    rows = []
    for sweep in r["sweeps"]["species_accept"]:
        th = sweep["thresholds"]["species_accept"]
        if th not in (0.5, 0.7, 0.9):
            continue
        for sp, v in sweep.get("accepted_by_species", {}).items():
            lo, hi = _wilson(v["correct"], v["accepted"])
            rows.append(
                [
                    f"{th:g}",
                    sp,
                    f"{v['correct']} / {v['accepted']}",
                    f"{_pct(v['correct'] / v['accepted'])} [{100 * lo:.1f}, {100 * hi:.1f}]",
                    "**pass**" if lo >= op["targets"]["min_accepted_precision"] else "fail",
                ]
            )
    enabled = op.get("accept_species")
    return [
        "## Per-species gate",
        "",
        "A species can be accepted automatically only if its own accepted events meet the "
        "precision bound at the chosen species threshold. "
        + (
            f"Enabled species: {', '.join(enabled) if enabled else 'none'}."
            if op["species_accept"] is not None
            else "No species threshold passed the overall rule, so no species is enabled."
        )
        + " Evidence per species at a few thresholds (policy validation):",
        "",
        _t(
            [
                "Species >=",
                "Predicted species",
                "Correct / accepted",
                "Precision (95% CI)",
                "Bound",
            ],
            rows,
        ),
        "",
    ]


def _wilson(k: int, n: int) -> tuple[float, float]:
    from wildinbox.evaluation.metrics import wilson

    return wilson(k, n)


def _curves_section(report_dir: Path, r: dict[str, Any]) -> list[str]:
    from wildinbox.evaluation.curves import Point, Series, render

    op, tg = r["operating_point"], r["operating_point"]["targets"]
    fit_name = r["calibration"]["fit_partition"].replace("_", " ") + " cameras"
    choose_name = op["chosen_on"].replace("_", " ") + " cameras"

    def empty_points(sweep: list[dict[str, Any]]) -> list[Point]:
        return [
            Point(
                m["filtered"] / m["events"],
                m["false_empty_rate"],
                m["false_empty_rate_ci95"][0],
                m["false_empty_rate_ci95"][1],
                label=f"{m['thresholds']['empty_filter']:g}",
                mark=m["thresholds"]["empty_filter"] in SHOWN,
            )
            for m in sweep
        ]

    def species_points(sweep: list[dict[str, Any]]) -> list[Point]:
        out = []
        for m in sweep:
            if not m["accepted"]:
                continue
            lo, hi = m["accepted_precision_ci95"]
            out.append(
                Point(
                    m["accepted"] / m["events"],
                    1 - m["accepted_precision"],
                    1 - hi,
                    1 - lo,
                    label=f"{m['thresholds']['species_accept']:g}",
                    mark=m["thresholds"]["species_accept"] in SHOWN,
                )
            )
        return out

    fit_sweeps = r.get("sweeps_fit_partition")
    if not fit_sweeps:
        return []
    chosen_e = next(
        (
            p
            for p in empty_points(r["sweeps"]["empty_filter"])
            if op["empty_filter"] is not None and float(p.label) == op["empty_filter"]
        ),
        None,
    )
    (report_dir / "curve-empty-filter.svg").write_text(
        render(
            "Empty filter: animals lost vs events filtered",
            "Events filtered automatically (coverage)",
            "Animal events filtered as empty",
            [
                Series(choose_name, empty_points(r["sweeps"]["empty_filter"]), band=True),
                Series(fit_name, empty_points(fit_sweeps["empty_filter"])),
            ],
            tg["max_false_empty_rate"],
            f"target {_pct(tg['max_false_empty_rate'], 0)}",
            chosen_e,
            f"rule: {op['empty_filter']:g}" if chosen_e else "",
        )
    )
    (report_dir / "curve-species-accept.svg").write_text(
        render(
            "Species acceptance: wrong labels vs events accepted",
            "Events accepted automatically (coverage)",
            "Accepted labels that are wrong",
            [
                Series(choose_name, species_points(r["sweeps"]["species_accept"]), band=True),
                Series(fit_name, species_points(fit_sweeps["species_accept"])),
            ],
            1 - tg["min_accepted_precision"],
            f"target {_pct(1 - tg['min_accepted_precision'], 0)}",
        )
    )
    return [
        "## Error versus coverage",
        "",
        "Each point is one threshold on the grid; moving right automates more events and "
        "shows the error it costs. The band is the 95% interval on the cameras thresholds are "
        "chosen on; the other line is the replication cameras. Tables below list the numbers.",
        "",
        "![Empty filter: animals lost vs events filtered](curve-empty-filter.svg)",
        "",
        "![Species acceptance: wrong labels vs events accepted](curve-species-accept.svg)",
        "",
    ]


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
            ["", "Rule's threshold", "Rule result", "Released"],
            [
                [
                    "Empty filter (every usable frame P(empty) >=)",
                    _thr(op["empty_filter"]),
                    _on(op["rule_auto_filter_enabled"]),
                    f"**{_on(op['auto_filter_enabled'])}**",
                ],
                [
                    "Species acceptance (mean P(species) >=)",
                    _thr(op["species_accept"]),
                    _on(op["rule_auto_accept_enabled"]),
                    f"**{_on(op['auto_accept_enabled'])}**",
                ],
            ],
        ),
        "",
        *_deviation(op.get("deviation")),
        f"A threshold is enabled only if the worst case of its 95% Wilson interval on "
        f"{choose.replace('_', ' ')} meets the target: false-empty rate <= "
        f"{_pct(tg['max_false_empty_rate'], 0)} of animal-containing events, accepted-label "
        f"precision >= {_pct(tg['min_accepted_precision'], 0)}. The lowest passing threshold "
        "on the grid is chosen; a disabled step sends those events to review.",
        "",
        f"At the rule's operating point on {choose.replace('_', ' ')}:",
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
                    f"{m['accepted']} / {m['events']}"
                    + (
                        f", precision {_pct(m['accepted_precision'])} "
                        f"{_ci(m['accepted_precision_ci95'])}"
                        if m["accepted"]
                        else ""
                    ),
                ],
                [
                    "Unsupported animals accepted as a known species",
                    m["unsupported_accepted_as_known"],
                ],
                ["Events left for review", f"{m['needs_review']} ({_pct(m['review_fraction'])})"],
            ],
        ),
        "",
        *_policy_section(r),
        *_gate_section(r),
        *_curves_section(report_dir, r),
        "## Replication on every development camera",
        "",
        "The rule's thresholds applied to each unseen development camera. The calibration "
        "cameras were used to fit the temperature, not to choose thresholds, so they are an "
        "independent check of the operating point.",
        "",
        _t(
            [
                "Camera",
                "Partition",
                "Animal events filtered as empty (95% CI)",
                "Filtered",
                "Accepted",
            ],
            [
                [
                    cam,
                    c["partition"],
                    f"{c['false_empty']} / {c['animal_events']} ({_pct(c['false_empty_rate'], 1)} "
                    f"{_ci(c['false_empty_rate_ci95'])})",
                    f"{c['filtered']} / {c['events']}",
                    c["accepted"],
                ]
                for cam, c in op["per_camera"].items()
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
                    _rate(a["observed_empty"], a["observed_empty_ci95"]),
                    b["images"],
                    "n/a" if b["mean_p_empty"] is None else f"{b['mean_p_empty']:.3f}",
                    _rate(b["observed_empty"], b["observed_empty_ci95"]),
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
                    _rate(s["accepted_precision"], s["accepted_precision_ci95"]),
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
