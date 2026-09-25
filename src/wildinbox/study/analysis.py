"""The pre-registered analysis of the timed review study."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

import numpy as np

from wildinbox.study.design import CONDITIONS, correct


def _bootstrap(values: list[float], stat: Any, n: int, seed: int) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    draws = [stat(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def analyze(export: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    plan = export["plan"]
    classes: list[str] = plan["classes"]
    truth: dict[str, str | None] = plan["truth"]
    excl = protocol["analysis"]["exclude"]
    resamples = protocol["analysis"]["bootstrap_resamples"]
    seed = protocol["design"]["seed"]
    per_set = protocol["design"]["events_per_set"]

    by_participant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in export["trials"]:
        if t["block"] >= 1:  # practice is not analysed
            by_participant[t["participant"]].append(t)
    ratings: dict[str, dict[str, int]] = defaultdict(dict)
    for r in export["ratings"]:
        ratings[r["participant"]][r["condition"]] = r["difficulty"]

    rows, exclusions = [], []
    for code, trials in sorted(by_participant.items()):
        done = {c: [t for t in trials if t["condition"] == c] for c in CONDITIONS}
        if any(len(done[c]) < per_set for c in CONDITIONS):
            exclusions.append({"participant": code, "reason": "did not finish both blocks"})
            continue
        slow = [t for t in trials if t["seconds"] > excl["max_seconds_per_event"]]
        if len(slow) / len(trials) > excl["max_excluded_share"]:
            exclusions.append(
                {
                    "participant": code,
                    "reason": f"{len(slow)} of {len(trials)} trials over the time limit",
                }
            )
            continue
        row: dict[str, Any] = {"participant": code, "trials_over_limit": len(slow)}
        for c in CONDITIONS:
            kept = [t for t in done[c] if t["seconds"] <= excl["max_seconds_per_event"]]
            scored = [
                x
                for t in kept
                if (x := correct(t["label"], truth.get(t["event_id"]), classes)) is not None
            ]
            row[c] = {
                "median_seconds": statistics.median(t["seconds"] for t in kept),
                "accuracy": sum(scored) / len(scored) if scored else None,
                "mean_interactions": statistics.mean(t["interactions"] for t in kept),
                "difficulty": ratings[code].get(c),
            }
        row["seconds_difference"] = (
            row["suggested"]["median_seconds"] - row["grouped"]["median_seconds"]
        )
        acc_s, acc_g = row["suggested"]["accuracy"], row["grouped"]["accuracy"]
        row["accuracy_difference"] = (
            acc_s - acc_g if acc_s is not None and acc_g is not None else None
        )
        rows.append(row)

    diffs = [r["seconds_difference"] for r in rows]
    acc_diffs = [r["accuracy_difference"] for r in rows if r["accuracy_difference"] is not None]
    n = len(rows)
    enough = n >= protocol["design"]["min_participants"]
    time_ci = _bootstrap(diffs, np.median, resamples, seed)
    acc_ci = _bootstrap(acc_diffs, np.mean, resamples, seed + 1)
    summary: dict[str, Any] = {
        "participants_analysed": n,
        "participants_excluded": exclusions,
        "enough_participants": enough,
        "median_seconds": {
            c: statistics.median(r[c]["median_seconds"] for r in rows) if rows else None
            for c in CONDITIONS
        },
        "median_seconds_difference": statistics.median(diffs) if diffs else None,
        "median_seconds_difference_ci95": time_ci,
        "share_faster_with_suggestions": sum(d < 0 for d in diffs) / n if n else None,
        "mean_accuracy": {
            c: statistics.mean(a for r in rows if (a := r[c]["accuracy"]) is not None)
            if rows
            else None
            for c in CONDITIONS
        },
        "mean_accuracy_difference": statistics.mean(acc_diffs) if acc_diffs else None,
        "mean_accuracy_difference_ci95": acc_ci,
    }
    if not enough:
        need = protocol["design"]["min_participants"]
        verdict = f"descriptive only: {n} participant(s), the protocol needs {need}"
    elif time_ci is None:
        verdict = "not enough data for an interval"
    else:
        faster = time_ci[1] < 0
        acc_ok = (summary["mean_accuracy_difference"] or 0) >= -protocol["analysis"][
            "max_accuracy_drop"
        ]
        if faster and acc_ok:
            verdict = "suggestions made review faster without an accuracy drop beyond the limit"
        elif faster:
            verdict = "suggestions made review faster, but accuracy dropped beyond the limit"
        else:
            verdict = (
                "no reliable speed gain from suggestions "
                "(the interval includes zero or favours grouped review)"
            )
    summary["verdict"] = verdict
    return {"summary": summary, "participants": rows}
