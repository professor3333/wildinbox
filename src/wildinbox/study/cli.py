"""`wildinbox study plan` and `wildinbox study analyze` (organizer commands)."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import yaml

from wildinbox.api_client import api_client, request
from wildinbox.class_map import EMPTY_CLASS
from wildinbox.datasets.events import Role
from wildinbox.study.analysis import analyze
from wildinbox.study.design import build_sets

PROTOCOL = Path("configs/study/review_study.yaml")


def event_truth(row: dict[str, str]) -> str | None:
    """Ground truth per event, with the update-cycle label rules."""
    role, label = row["event_role"], row.get("event_label") or row.get("image_label") or None
    if role == Role.EMPTY:
        return EMPTY_CLASS
    if role in ("supported_species", "unsupported_animal"):
        return label
    return None


def create_plan(
    api_url: str,
    name: str,
    batch_ids: list[str],
    truth_csv: Path,
    client: httpx.Client | None = None,
) -> str:
    protocol = yaml.safe_load(PROTOCOL.read_text())
    d = protocol["design"]
    by_file = {r["filename"]: r for r in csv.DictReader(truth_csv.open())}
    api = client or api_client(api_url)
    truth: dict[str, str | None] = {}
    for batch in batch_ids:
        offset: int | None = 0
        while offset is not None:
            page = request(
                api,
                "GET",
                "/events",
                params={"batch_id": batch, "limit": 500, "offset": offset},
            ).json()
            for e in page["events"]:
                detail = request(api, "GET", f"/events/{e['id']}").json()
                first = detail["images"][0]["filename"]
                truth[e["id"]] = event_truth(by_file[first])
            offset = page["next_offset"]
    sets = build_sets(truth, d["events_per_set"], d["practice_events_per_condition"], d["seed"])
    used = {e for v in sets.values() for e in v}
    res = request(
        api,
        "POST",
        "/study/plans",
        json={
            "name": name,
            "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
            "sets": sets,
            "truth": {e: truth[e] for e in used},
        },
    )
    plan_id: str = res.json()["id"]
    return plan_id


def write_analysis(
    api_url: str, plan_id: str, out: Path, client: httpx.Client | None = None
) -> dict[str, Any]:
    protocol = yaml.safe_load(PROTOCOL.read_text())
    api = client or api_client(api_url)
    export = request(api, "GET", f"/study/plans/{plan_id}/export").json()
    if export["plan"]["protocol_sha256"] != hashlib.sha256(PROTOCOL.read_bytes()).hexdigest():
        raise RuntimeError("the protocol file changed after this study plan was created")
    result = analyze(export, protocol)
    out.mkdir(parents=True, exist_ok=True)
    (out / "export.json").write_text(json.dumps(export, indent=2) + "\n")
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "README.md").write_text(report(export, result))
    return result


def _s(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1f} s"


def _p(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def report(export: dict[str, Any], result: dict[str, Any]) -> str:
    s = result["summary"]
    ci = s["median_seconds_difference_ci95"]
    aci = s["mean_accuracy_difference_ci95"]
    lines = [
        f"# Review study: {export['plan']['name']}",
        "",
        "Protocol [`configs/study/review_study.yaml`](../../../configs/study/review_study.yaml), "
        "committed before any participant took part. Grouped review with and without model "
        "suggestions; grouping is the same in both, so it is not credited to the model.",
        "",
        f"**Result: {s['verdict']}.**",
        "",
        "| | Grouped | Suggested | Difference (suggested - grouped) |",
        "|---|---|---|---|",
        f"| Median seconds per event | {_s(s['median_seconds']['grouped'])} | "
        f"{_s(s['median_seconds']['suggested'])} | {_s(s['median_seconds_difference'])}"
        + (f" [{ci[0]:+.1f}, {ci[1]:+.1f}]" if ci else "")
        + " |",
        f"| Mean accuracy | {_p(s['mean_accuracy']['grouped'])} | "
        f"{_p(s['mean_accuracy']['suggested'])} | "
        + (
            f"{s['mean_accuracy_difference']:+.1%}"
            if s["mean_accuracy_difference"] is not None
            else "n/a"
        )
        + (f" [{aci[0]:+.1%}, {aci[1]:+.1%}]" if aci else "")
        + " |",
        "",
        f"Participants analysed: {s['participants_analysed']}; faster with suggestions: "
        f"{_p(s['share_faster_with_suggestions'])}.",
        "",
    ]
    if s["participants_excluded"]:
        lines += (
            ["Excluded:", ""]
            + [f"- {e['participant']}: {e['reason']}" for e in s["participants_excluded"]]
            + [""]
        )
    lines += [
        "| Participant | Grouped s | Suggested s | Grouped accuracy | Suggested accuracy "
        "| Difficulty (g / s) |",
        "|---|---|---|---|---|---|",
    ]
    for r in result["participants"]:
        g, sg = r["grouped"], r["suggested"]
        lines.append(
            f"| {r['participant']} | {_s(g['median_seconds'])} | {_s(sg['median_seconds'])} | "
            f"{_p(g['accuracy'])} | {_p(sg['accuracy'])} | "
            f"{g['difficulty'] or '-'} / {sg['difficulty'] or '-'} |"
        )
    w = s.get("workload")
    if w:
        lines += [
            "",
            f"**Projected reviewer time per {w['events']:,} events, audits included** "
            f"(active release: {w['needs_review_share']:.0%} of events need review, "
            f"{w['automatic_share']:.0%} handled automatically, audit rate {w['audit_rate']:.0%}):",
            "",
            "| Workflow | Minutes | Of which audits |",
            "|---|---|---|",
            f"| Grouped, no suggestions | {w['grouped_minutes']:.0f} | - |",
            f"| WildInbox | {w['system_minutes']:.0f} | {w['audit_minutes']:.0f} |",
            "",
            f"Time saved: {_p(w['time_saved_share'])}.",
        ]
    lines += ["", "Intervals: 95% bootstrap over participants.", ""]
    return "\n".join(lines)
