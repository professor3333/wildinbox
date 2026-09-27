"""Checks beyond the pre-registered analysis, computed from a study export only.

The summary without named participants was declared before results were seen
(reports/study/<study>/PLAN.md); everything else here was chosen after seeing
them and is labelled post-hoc wherever it is reported.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from wildinbox.class_map import EMPTY_CLASS
from wildinbox.study.analysis import analyze
from wildinbox.study.design import CANT_TELL, CONDITIONS, OTHER, category, correct


def _model_suggestion(export: dict[str, Any], trial: dict[str, Any]) -> str | None:
    """The suggestion shown in the "suggested" condition; in "grouped", the one
    the model would have shown (the event's latest decision)."""
    if trial["displayed"] is not None:
        label: str | None = trial["displayed"]["suggested_label"]
        return label
    decisions = export["events"][trial["event_id"]]["decisions"]
    return decisions[-1]["suggested_label"] if decisions else None


def posthoc(export: dict[str, Any], protocol: dict[str, Any], without: list[str]) -> dict[str, Any]:
    plan = export["plan"]
    classes: list[str] = plan["classes"]
    truth: dict[str, str | None] = plan["truth"]
    timed = [e for s in ("A", "B") for e in plan["sets"][s]]

    def model_label(eid: str) -> str | None:
        decisions = export["events"][eid]["decisions"]
        return decisions[-1]["suggested_label"] if decisions else None

    unsupported = [e for e in timed if category(truth[e], classes) == OTHER]
    supported = [e for e in timed if truth[e] is not None and e not in unsupported]
    right = [e for e in timed if correct(model_label(e), truth[e], classes)]

    trials = [t for t in export["trials"] if t["block"] >= 1]
    missing = [t for t in trials if t["condition"] == "suggested" and t["displayed"] is None]
    if missing:
        raise ValueError(f"{len(missing)} suggested trial(s) have no displayed decision")
    by_participant: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {c: [] for c in CONDITIONS}
    )
    for t in trials:
        by_participant[t["participant"]][t["condition"]].append(t)

    def agrees(t: dict[str, Any]) -> bool:
        suggestion = _model_suggestion(export, t)
        return category(t["label"], classes) == category(suggestion, classes)

    # Accuracy and time as the pre-registered analysis computes them (same
    # exclusions), so the two reports cannot disagree; agreement over all trials.
    registered = {r["participant"]: r for r in analyze(export, protocol)["participants"]}
    participants = {}
    for code, conds in sorted(by_participant.items()):
        row: dict[str, Any] = {}
        for c, ts in conds.items():
            reg = registered.get(code, {}).get(c, {})
            row[c] = {
                "trials": len(ts),
                "matches_model": sum(agrees(t) for t in ts),
                "accuracy": reg.get("accuracy"),
                "median_seconds": reg.get("median_seconds"),
            }
        participants[code] = row

    answers: dict[str, dict[str, int]] = {}
    for c in CONDITIONS:
        ts = [t for t in trials if t["condition"] == c]
        on_unsupported = [t for t in ts if t["event_id"] in unsupported]
        on_empty = [t for t in ts if truth[t["event_id"]] == EMPTY_CLASS]
        answers[c] = {
            "unsupported_decisions": len(on_unsupported),
            "answered_other_species": sum(
                category(t["label"], classes) == OTHER for t in on_unsupported
            ),
            "answered_empty": sum(
                category(t["label"], classes) == EMPTY_CLASS for t in on_unsupported
            ),
            "answered_cant_tell": sum(
                category(t["label"], classes) == CANT_TELL for t in on_unsupported
            ),
            "empty_decisions": len(on_empty),
            "empty_recognised": sum(category(t["label"], classes) == EMPTY_CLASS for t in on_empty),
        }

    reduced = {
        **export,
        "trials": [t for t in export["trials"] if t["participant"] not in without],
        "participants": [p for p in export.get("participants", []) if p["code"] not in without],
        "ratings": [r for r in export["ratings"] if r["participant"] not in without],
    }
    return {
        "without": without,
        "summary_without": analyze(reduced, protocol)["summary"] if without else None,
        "suggestions": {
            "timed_events": len(timed),
            "correct": len(right),
            "unsupported_events": len(unsupported),
            "supported_events": len(supported),
            "correct_on_supported": len([e for e in right if e in supported]),
            "displayed_sources": dict(
                sorted(_count(t["displayed"]["source"] for t in trials if t["displayed"]).items())
            ),
        },
        "participants": participants,
        "answers": answers,
    }


def _count(items: Any) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for x in items:
        out[x] += 1
    return dict(out)


def _p(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def report(result: dict[str, Any]) -> str:
    s = result["suggestions"]
    lines = [
        "# Review study: checks beyond the pre-registered analysis",
        "",
        "Generated by `wildinbox study posthoc` from [export.json](export.json); not edited.",
        "",
    ]
    w = result["summary_without"]
    if w is not None:
        ci, aci = w["median_seconds_difference_ci95"], w["mean_accuracy_difference_ci95"]
        lines += [
            f"## Pre-declared: without {', '.join(result['without'])}",
            "",
            "| | Grouped | Suggested | Difference |",
            "|---|---|---|---|",
            f"| Median seconds per event | {w['median_seconds']['grouped']:.1f} s | "
            f"{w['median_seconds']['suggested']:.1f} s | "
            f"{w['median_seconds_difference']:+.2f} s"
            + (f" [{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else "")
            + " |",
            f"| Mean accuracy | {_p(w['mean_accuracy']['grouped'])} | "
            f"{_p(w['mean_accuracy']['suggested'])} | "
            f"{100 * w['mean_accuracy_difference']:+.1f} points"
            + (f" [{100 * aci[0]:+.1f}, {100 * aci[1]:+.1f}]" if aci else "")
            + " |",
            "",
            f"Participants: {w['participants_analysed']}; faster with suggestions: "
            f"{_p(w['share_faster_with_suggestions'])}; projected time saved: "
            f"{_p((w.get('workload') or {}).get('time_saved_share'))}.",
            "",
        ]
    sources = ", ".join(f"{k}: {v}" for k, v in s["displayed_sources"].items())
    lines += [
        "## Post-hoc: the suggestions",
        "",
        f"Right on {s['correct']} of {s['timed_events']} timed events; "
        f"{s['unsupported_events']} are unsupported species (correct answer: other species); "
        f"right on {s['correct_on_supported']} of {s['supported_events']} supported events.",
        f"Displayed decision per suggested trial ({sources}).",
        "",
        "## Post-hoc: agreement with the model",
        "",
        "Suggested block: answer matches the suggestion shown. Grouped block: answer "
        "matches the suggestion the model would have shown (not visible).",
        "",
        "| Participant | Suggested: matches suggestion | Grouped: matches model "
        "| Accuracy, grouped / suggested | Median s, grouped / suggested |",
        "|---|---|---|---|---|",
    ]
    for code, r in result["participants"].items():
        g, sg = r["grouped"], r["suggested"]
        lines.append(
            f"| {code} | {sg['matches_model']} / {sg['trials']} | "
            f"{g['matches_model']} / {g['trials']} | {_p(g['accuracy'])} / {_p(sg['accuracy'])} | "
            f"{g['median_seconds']:.1f} / {sg['median_seconds']:.1f} |"
        )
    lines += [
        "",
        "## Post-hoc: unsupported species and empty events",
        "",
        "| | Grouped | Suggested |",
        "|---|---|---|",
    ]
    a = result["answers"]
    for key, name in (
        ("unsupported_decisions", "Decisions on unsupported species"),
        ("answered_other_species", "... answered other species"),
        ("answered_empty", "... answered empty"),
        ("answered_cant_tell", "... answered can't tell"),
        ("empty_decisions", "Decisions on empty events"),
        ("empty_recognised", "... answered empty"),
    ):
        lines.append(f"| {name} | {a['grouped'][key]} | {a['suggested'][key]} |")
    return "\n".join([*lines, ""])
