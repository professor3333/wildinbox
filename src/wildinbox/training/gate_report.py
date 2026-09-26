"""Markdown report for `wildinbox update gate`."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _t(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _lost(m: dict[str, Any]) -> str:
    h = m["holdout"]
    return f"{h['animal_events_suggested_empty']} / {h['animal_events']}"


def write_report(report_dir: Path, out: dict[str, Any]) -> Path:
    d, c = out["results"]["deployed"], out["results"]["candidate"]
    snap = out["snapshot"]
    checks = out["checks"]
    # candidates trained before snapshot/v2 recorded a single `reviewer`
    reviewers = snap.get("approved_reviewers") or [snap["reviewer"]]
    simulated = reviewers == ["simulated-ground-truth"]
    md = [
        f"# Update candidate: `{out['candidate']}`",
        "",
        f"Protocol [`{out['protocol']}`](../../../{out['protocol']}), committed before any "
        f"review was collected or candidate trained. Snapshot `{snap['version']}`: "
        f"{snap['images']} images from {snap['events']} reviewed events"
        + (
            ". **Reviews were simulated from the dataset's ground truth** "
            "(reviewer `simulated-ground-truth`)."
            if simulated
            else f", approved reviewers {', '.join(f'`{r}`' for r in reviewers)}."
        ),
        "",
        "## Decision",
        "",
        f"**{'Promote' if out['promote'] else 'Do not promote'}** "
        f"`{out['candidate_release']}` over `{out['deployed_release']}`.",
        "",
        _t(
            ["Check", "Result", "Requirement", "Pass"],
            [
                [
                    "Holdout event macro-F1 gain",
                    f"{checks['holdout_gain']['value']:+.3f}",
                    f">= {checks['holdout_gain']['required']:+.2f}",
                    "yes" if checks["holdout_gain"]["pass"] else "**no**",
                ],
                *[
                    [
                        f"Macro-F1 change, {k.removeprefix('no_regression_').replace('_', ' ')}",
                        f"{v['value']:+.3f}",
                        f">= {v['allowed']:+.2f}",
                        "yes" if v["pass"] else "**no**",
                    ]
                    for k, v in checks.items()
                    if k.startswith("no_regression_")
                ],
                [
                    "Holdout animal events suggested as empty",
                    f"{checks['false_empty_suggestions']['value']}",
                    f"<= {checks['false_empty_suggestions']['limit']:.1f}",
                    "yes" if checks["false_empty_suggestions"]["pass"] else "**no**",
                ],
            ],
        ),
        "",
        "## Holdout: later reviewed events on the updated cameras",
        "",
        _t(
            ["", "Deployed", "Candidate"],
            [
                [
                    "Events scored (supported label)",
                    d["holdout"]["events_scored"],
                    c["holdout"]["events_scored"],
                ],
                [
                    "Event macro-F1",
                    f"{d['holdout']['macro_f1']:.3f}",
                    f"{c['holdout']['macro_f1']:.3f}",
                ],
                *[
                    [
                        f"Event macro-F1, camera {cam}",
                        f"{v:.3f}",
                        f"{c['holdout']['per_camera'][cam]:.3f}",
                    ]
                    for cam, v in d["holdout"]["per_camera"].items()
                ],
                [
                    "Animal events suggested as empty",
                    _lost(d),
                    _lost(c),
                ],
                ["Temperature", f"{d['temperature']:.3f}", f"{c['temperature']:.3f}"],
            ],
        ),
        "",
        _t(
            ["Class", "Deployed recall", "Candidate recall", "Holdout events"],
            [
                [
                    k,
                    f"{d['holdout']['per_class'][k]['recall']:.3f}",
                    f"{c['holdout']['per_class'][k]['recall']:.3f}",
                    d["holdout"]["per_class"][k]["support"],
                ]
                for k in d["holdout"]["per_class"]
            ],
        ),
        "",
        "## Regression checks: cameras neither model trained on",
        "",
        _t(
            ["Data", "Deployed macro-F1", "Candidate macro-F1"],
            [
                [k.replace("_", " "), f"{v:.3f}", f"{c['regression_macro_f1'][k]:.3f}"]
                for k, v in d["regression_macro_f1"].items()
            ],
        ),
        "",
        "## Development data only",
        "",
        *(
            [
                "Evaluated on the snapshot holdout, the calibration cameras, and the "
                "seen-camera diagnostic; the final test was not read. The snapshot holds no "
                "protected evaluation frame (checked before scoring). This was comparison "
                f"#{out['development_data_only']['comparison_number_on_this_holdout']} on "
                "this holdout"
                + (
                    f" (budget {out['development_data_only']['comparison_budget']})."
                    if out["development_data_only"]["comparison_budget"]
                    else "; every comparison is logged in `../comparisons.jsonl`."
                ),
                "",
            ]
            if "development_data_only" in out
            else []
        ),
        "## What this shows",
        "",
        "Holdout and snapshot share cameras and backgrounds by design: the gain is what "
        "reviewing a camera buys that camera's later photos, not evidence of generalization "
        "to new cameras; the regression checks cover cameras outside the update. The "
        "candidate keeps the deployed policy settings (automation off); only its "
        "calibration was refit.",
        "",
        f"Code `{out['code']['commit']}`"
        f"{' (uncommitted changes)' if out['code']['dirty'] else ''}.",
        "",
    ]
    path = report_dir / "README.md"
    path.write_text("\n".join(md))
    return path
