"""Markdown report for `wildinbox unfamiliar`."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

NAMES = {
    "knn_cosine": "Distance to training images (kNN cosine)",
    "max_softmax": "Confidence (1 - max calibrated probability)",
}


def _t(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _r(x: dict[str, Any]) -> str:
    if not x["n"]:
        return "n/a"
    lo, hi = x["ci95"]
    return f"{x['flagged']} / {x['n']} ({100 * x['rate']:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}])"


def write_report(report_dir: Path, r: dict[str, Any]) -> Path:
    parts = r["partitions"]
    fit, check = list(parts)
    tuning = r["species"]["tuning"]
    rows = []
    for p in (fit, check):
        for name, m in parts[p]["methods"].items():
            rows.append(
                [
                    p + (" (fit)" if p == fit else " (check)"),
                    NAMES[name],
                    f"{m['threshold']:.4f}",
                    _r(m["false_flags"]),
                    _r(m["detection"]),
                    "n/a" if m["auroc"] is None else f"{m['auroc']:.3f}",
                ]
            )
    species_rows = [
        [p, NAMES[name], *(_r(m["detection_by_species"][s]) for s in tuning)]
        for p in (fit, check)
        for name, m in parts[p]["methods"].items()
    ]
    a = r["artifact"]
    md = [
        f"# Unfamiliar-input score: `{r['model']}`",
        "",
        f"Rule: [`{r['rule']}`](../../{r['rule']}), committed before any score was computed. "
        "Development partitions only; **the locked final test was not opened.**",
        "",
        "## Decision",
        "",
        f"**The distance score is {'adopted' if r['adopted'] else 'not adopted'}.** On the "
        f"{fit} cameras, at a false-flag rate of at most {100 * r['max_false_flag_rate']:.0f}% of "
        f"supported-species animal images, it flags {100 * r['detection_gain_on_fit']:+.1f} "
        f"percentage points of tuning-species images compared with ordinary confidence "
        f"(adoption needs >= {100 * r['adopt_if_detection_gain']:+.0f})."
        + (
            f" Frames whose distance exceeds {a['threshold']:.4f} mark their event "
            "`possible_unknown`, which sends it to review."
            if r["adopted"]
            else " The policy uses confidence only; the score is reported for reviewers' context."
        ),
        "",
        "## Results",
        "",
        "Known = supported-species animal images; unknown = tuning species "
        f"({', '.join(tuning)}), which the classifier was never fit on. Each threshold is fit "
        f"on {fit} and applied unchanged to {check}. AUROC = chance an unknown image scores "
        "higher than a known one.",
        "",
        _t(
            [
                "Partition",
                "Score",
                "Threshold",
                "Known flagged (false flags)",
                "Unknown flagged (detection)",
                "AUROC",
            ],
            rows,
        ),
        "",
        "Detection by tuning species:",
        "",
        _t(["Partition", "Score", *tuning], species_rows),
        "",
        "## Species and pretraining exposure",
        "",
        _t(
            ["Species", "Role", "Unseen in fine-tuning", "In ImageNet-1k pretraining classes"],
            [
                ["squirrel", "tuning", "yes", "yes (fox squirrel)"],
                ["rodent", "tuning", "yes", "related classes (hamster, marmot, beaver, porcupine)"],
                ["skunk", "held out", "yes", "yes"],
                ["bird", "held out", "yes", "yes (many bird classes)"],
                ["badger", "held out", "yes", "yes"],
                ["fox", "held out", "yes", "yes (red, kit, Arctic, grey fox)"],
                ["deer", "held out", "yes", "no (no development or final-test events)"],
            ],
        ),
        "",
        "Every unfamiliar species that can be evaluated was absent from fine-tuning but present "
        "in the pretraining class list, so these results say nothing about species the backbone "
        "has never seen. Held-out species are excluded from every number here and are "
        "evaluated once, on the final test.",
        "",
        "## Artifact",
        "",
        f"`{a['method']}`, k = {a['k']}, threshold {a['threshold']:.4f}, reference = "
        f"{a['reference']['images']} training images "
        f"(ids digest `{a['reference']['ids_digest']}`), "
        f"weights `{a['weights_digest']}`; version `{a['version']}`.",
        "",
        "## Limitations",
        "",
        f"- Tuning species on the {check} cameras are almost all squirrels; rodents appear only "
        f"on the {fit} cameras.",
        "- A 10% false-flag budget sends that share of real animals to review as possibly "
        "unknown; the operating-point rule measures the combined effect.",
        "",
        "## Reproduce",
        "",
        "```bash",
        f"uv run wildinbox unfamiliar --rule {r['rule']}",
        "```",
        "",
        f"Code `{r['code']['commit']}`{' (uncommitted changes)' if r['code']['dirty'] else ''}; "
        f"split `{r['split_version']}`.",
        "",
    ]
    out = report_dir / "README.md"
    out.write_text("\n".join(md))
    return out
