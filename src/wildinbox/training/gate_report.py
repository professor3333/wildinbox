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


def _provenance_claim(prov: dict[str, Any] | None) -> str:
    """The protocol's timing, claimed only as far as the evidence shows."""
    if prov is None:
        return "; when it was committed relative to the reviews and training was not recorded"
    reviews = {
        True: "committed before the snapshot's first review",
        False: "committed after the snapshot's reviews were made",
        None: "not shown to predate the reviews",
    }[prov["committed_before_reviews"]]
    training = {
        True: "committed before the candidate was trained",
        False: "not the version the candidate was trained with",
        None: "not shown to predate training",
    }[prov["committed_before_training"]]
    return f": {reviews}; {training} ([evidence](#protocol-provenance))"


def _yes_no(v: bool | None) -> str:
    return {True: "yes", False: "**no**", None: "not established"}[v]


def _provenance_section(prov: dict[str, Any]) -> list[str]:
    first, code = prov["first_committed"], prov["training_code"]
    if code["commit"] is None:
        trained = "not recorded"
    else:
        trained = f"`{code['commit'][:7]}`" + (
            ", with uncommitted changes" if code["dirty"] else ", clean tree"
        )
    return [
        "## Protocol provenance",
        "",
        _t(
            ["Evidence", "Value"],
            [
                ["Protocol content", f"sha256 `{prov['protocol_sha256'][:12]}`"],
                [
                    "First committed with this content",
                    f"`{first['commit'][:7]}` at {first['committed_at']}"
                    if first
                    else "not found in git history",
                ],
                [
                    "Earliest review in the snapshot",
                    prov["earliest_review_at"] or "not recorded in the snapshot",
                ],
                ["Candidate trained at", trained],
                ["Committed before the reviews", _yes_no(prov["committed_before_reviews"])],
                ["Committed before training", _yes_no(prov["committed_before_training"])],
            ],
        ),
        "",
        "Read from git history, the snapshot's review times, and the candidate's "
        "training record; nothing is claimed that they do not show. This is about when "
        "the rules were fixed, not a check of the metrics above.",
        "",
    ]


def _pass(check: dict[str, Any]) -> str:
    if check["pass"] is None:
        if "min_support" in check and check["support"] < check["min_support"]:
            return f"insufficient evidence ({check['support']} < {check['min_support']})"
        return "inconclusive"
    return "yes" if check["pass"] else "**no**"


def _decision(out: dict[str, Any]) -> list[str]:
    """Recorded gates before the promotion policy had only promote / do not promote."""
    head = f"`{out['candidate_release']}` over `{out['deployed_release']}`."
    decision = out.get("decision")
    if decision is None:
        return [f"**{'Promote' if out['promote'] else 'Do not promote'}** {head}"]
    line = {
        "promote": f"**Promote** {head}",
        "reject": f"**Reject** {head} A check failed on enough evidence.",
        "inconclusive": f"**Inconclusive: do not promote** {head} Nothing failed, but "
        "some check had too little evidence to judge. Gate again on more reviewed "
        "events or a fresh holdout.",
    }[decision]
    if out.get("promotion_policy") == "legacy":
        line += (
            "\n\nGated under the **legacy policy** (aggregate checks only: no per-species "
            "limits, no minimum evidence), to reproduce a recorded cycle. A decision "
            "under this policy is not sufficient for operational release."
        )
    return [line]


def _species_rows(checks: dict[str, Any]) -> list[list[Any]]:
    return [
        [
            f"Recall change, {k.removeprefix('species_recall_')} "
            f"({v['deployed']:.3f} -> {v['candidate']:.3f}, {v['support']} events)",
            f"{v['value']:+.3f}",
            f">= {v['allowed']:+.2f}, on >= {v['min_support']} events",
            _pass(v),
        ]
        for k, v in checks.items()
        if k.startswith("species_recall_")
    ]


def _requirement(v: dict[str, Any], rule: str) -> str:
    return rule + (f", on >= {v['min_support']} events" if "min_support" in v else "")


def write_report(report_dir: Path, out: dict[str, Any]) -> Path:
    d, c = out["results"]["deployed"], out["results"]["candidate"]
    snap = out["snapshot"]
    checks = out["checks"]
    prov = out.get("protocol_provenance")
    # candidates trained before snapshot/v2 recorded a single `reviewer`
    reviewers = snap.get("approved_reviewers") or [snap["reviewer"]]
    simulated = reviewers == ["simulated-ground-truth"]
    md = [
        f"# Update candidate: `{out['candidate']}`",
        "",
        f"Protocol [`{out['protocol']}`](../../../{out['protocol']}) "
        f"(sha256 `{out['protocol_sha256'][:12]}`){_provenance_claim(prov)}. "
        f"Snapshot `{snap['version']}`: "
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
        *_decision(out),
        "",
        _t(
            ["Check", "Result", "Requirement", "Pass"],
            [
                [
                    "Holdout event macro-F1 gain",
                    f"{checks['holdout_gain']['value']:+.3f}",
                    _requirement(
                        checks["holdout_gain"], f">= {checks['holdout_gain']['required']:+.2f}"
                    ),
                    _pass(checks["holdout_gain"]),
                ],
                *[
                    [
                        f"Macro-F1 change, {k.removeprefix('no_regression_').replace('_', ' ')}",
                        f"{v['value']:+.3f}",
                        f">= {v['allowed']:+.2f}",
                        _pass(v),
                    ]
                    for k, v in checks.items()
                    if k.startswith("no_regression_")
                ],
                [
                    "Holdout animal events suggested as empty",
                    f"{checks['false_empty_suggestions']['value']}",
                    _requirement(
                        checks["false_empty_suggestions"],
                        f"<= {checks['false_empty_suggestions']['limit']:.1f}",
                    ),
                    _pass(checks["false_empty_suggestions"]),
                ],
                *_species_rows(checks),
                *(
                    [
                        [
                            "Comparisons on this holdout",
                            checks["comparison_budget"]["value"],
                            f"<= {checks['comparison_budget']['limit']}",
                            _pass(checks["comparison_budget"]),
                        ]
                    ]
                    if "comparison_budget" in checks
                    else []
                ),
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
        *(_provenance_section(prov) if prov else []),
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
