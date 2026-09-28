"""Analysis of review study 2 (configs/study/review_study_2.yaml): one
reviewer, events as units, intervals by resampling events within sessions.

The verdict says "faster without unacceptable accuracy loss" only when the
time interval lies entirely below zero AND the accuracy interval's lower
bound is above -max_accuracy_drop, for a complete schedule. It always says
the result is one reviewer's.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from itertools import pairwise
from typing import Any

import numpy as np

from wildinbox.study.design import ASSISTED, category, correct

GROUPED = "grouped"


def _stratified_bootstrap(
    groups: dict[int, list[float]], stat: Any, n: int, rng: np.random.Generator
) -> np.ndarray:
    """`stat` over `n` resamples that redraw each session's events within it."""
    arrays = [np.asarray(v, dtype=float) for v in groups.values() if v]
    out = np.empty(n)
    for i in range(n):
        out[i] = stat(np.concatenate([rng.choice(a, size=len(a)) for a in arrays]))
    return out


def analyze_single(export: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    plan = export["plan"]
    classes, truth = plan["classes"], plan["truth"]
    schedule: list[str] = protocol["schedule"]["order"]
    limit = protocol["analysis"]["exclude"]["max_seconds_per_event"]
    resamples = protocol["analysis"]["bootstrap_resamples"]
    drop = protocol["analysis"]["max_accuracy_drop"]
    trials = [t for t in export["trials"] if t["block"] >= 1]
    excluded = [t for t in trials if t["seconds"] > limit]
    kept = [t for t in trials if t["seconds"] <= limit]

    seconds: dict[str, dict[int, list[float]]] = {
        GROUPED: defaultdict(list),
        ASSISTED: defaultdict(list),
    }
    right: dict[str, dict[int, list[float]]] = {
        GROUPED: defaultdict(list),
        ASSISTED: defaultdict(list),
    }
    shown_events = wrong_shown = wrong_followed = 0
    right_when_shown: list[float] = []
    right_when_not: list[float] = []
    for t in kept:
        c = t["condition"]
        seconds[c][t["block"]].append(t["seconds"])
        ok = correct(t["label"], truth.get(t["event_id"]), classes)
        if ok is not None:
            right[c][t["block"]].append(float(ok))
        if c == ASSISTED:
            sug = t.get("displayed")
            if sug:
                shown_events += 1
                if ok is not None:
                    right_when_shown.append(float(ok))
                t_true = truth.get(t["event_id"])
                if t_true is not None and category(sug["suggested_label"], classes) != category(
                    t_true, classes
                ):
                    wrong_shown += 1
                    wrong_followed += category(t["label"], classes) == category(
                        sug["suggested_label"], classes
                    )
            elif ok is not None:
                right_when_not.append(float(ok))

    def flat(d: dict[int, list[float]]) -> list[float]:
        return [x for v in d.values() for x in v]

    rng = np.random.default_rng(protocol["events"]["seed"])
    med = {c: float(np.median(flat(seconds[c]))) if flat(seconds[c]) else None for c in seconds}
    acc = {c: float(np.mean(flat(right[c]))) if flat(right[c]) else None for c in right}
    complete = sorted({t["block"] for t in trials}) == list(range(1, len(schedule) + 1))
    time_ci = acc_ci = None
    if all(flat(seconds[c]) for c in seconds) and all(flat(right[c]) for c in right):
        t_diff = _stratified_bootstrap(seconds[ASSISTED], np.median, resamples, rng) - (
            _stratified_bootstrap(seconds[GROUPED], np.median, resamples, rng)
        )
        a_diff = _stratified_bootstrap(right[ASSISTED], np.mean, resamples, rng) - (
            _stratified_bootstrap(right[GROUPED], np.mean, resamples, rng)
        )
        time_ci = [float(np.percentile(t_diff, 2.5)), float(np.percentile(t_diff, 97.5))]
        acc_ci = [float(np.percentile(a_diff, 2.5)), float(np.percentile(a_diff, 97.5))]

    success = bool(complete and time_ci and acc_ci and time_ci[1] < 0 and acc_ci[0] > -drop)
    if not complete:
        verdict = "incomplete: not every session was run; descriptive only"
    elif success:
        verdict = (
            "for this one reviewer, assisted review was faster without an accuracy loss "
            f"beyond {drop:.0%} (not evidence about other reviewers)"
        )
    else:
        verdict = "not demonstrated for this reviewer"
    return {
        "participants": sorted({t["participant"] for t in trials}),
        "sessions_run": sorted({t["block"] for t in trials}),
        "complete": complete,
        "trials_excluded_over_limit": len(excluded),
        "median_seconds": med,
        "median_seconds_difference": _diff(med),
        "median_seconds_difference_ci95": time_ci,
        "accuracy": acc,
        "accuracy_difference": _diff(acc),
        "accuracy_difference_ci95": acc_ci,
        "max_accuracy_drop": drop,
        "success": success,
        "verdict": verdict,
        "assisted": {
            "events_with_suggestion_shown": shown_events,
            "accuracy_when_shown": _mean(right_when_shown),
            "accuracy_when_not_shown": _mean(right_when_not),
            "wrong_suggestions_shown": wrong_shown,
            "wrong_suggestions_followed": wrong_followed,
        },
        "session_gaps_under_minimum": _short_gaps(
            trials, protocol["schedule"]["min_minutes_between_sessions"]
        ),
    }


def _diff(by_condition: dict[str, float | None]) -> float | None:
    a, g = by_condition[ASSISTED], by_condition[GROUPED]
    return None if a is None or g is None else a - g


def _mean(xs: list[float]) -> float | None:
    return float(np.mean(xs)) if xs else None


def _short_gaps(trials: list[dict[str, Any]], minutes: float) -> list[dict[str, Any]]:
    """Consecutive sessions closer than the protocol's minimum break (deviations)."""
    span: dict[int, tuple[datetime, datetime]] = {}
    for t in trials:
        a, b = datetime.fromisoformat(t["shown_at"]), datetime.fromisoformat(t["decided_at"])
        lo, hi = span.get(t["block"], (a, b))
        span[t["block"]] = (min(lo, a), max(hi, b))
    out = []
    blocks = sorted(span)
    for prev, nxt in pairwise(blocks):
        gap = (span[nxt][0] - span[prev][1]).total_seconds() / 60
        if gap < minutes:
            out.append({"after_session": prev, "minutes": round(gap, 1)})
    return out
