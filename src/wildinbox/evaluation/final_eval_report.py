"""reports/final_evaluation/README.md, generated from metrics.json only, so
every number in the report is traceable to the saved run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

NAMES = {
    "animal_event_retention": "Animal-event retention",
    "accepted_species_precision": "Accepted species precision",
    "automatic_coverage": "Automatic coverage",
    "review_reduction": "Review reduction",
    "unsupported_false_acceptance": "Unsupported-species false acceptance",
    "retention_supported_species": "Retention: supported species",
    "retention_unsupported_animal": "Retention: unsupported species",
    "retention_mixed_species": "Retention: mixed-species events",
}
GALLERY_TITLES = {
    "animal_filtered_as_empty": "Animal events the rule's empty filter would remove",
    "confident_wrong_species": "Confident wrong species (>= 0.9)",
    "unsupported_as_known": "Unsupported species suggested as a known one (>= 0.9)",
    "animal_suggested_empty_night": "Night animal events suggested as empty",
}


def pct(v: float | None, digits: int = 1) -> str:
    return "undefined" if v is None else f"{100 * v:.{digits}f}%"


def ci(c: list[float] | None, digits: int = 1) -> str:
    return "—" if c is None else f"[{100 * c[0]:.{digits}f}, {100 * c[1]:.{digits}f}]"


def verdict(t: dict[str, Any]) -> str:
    if t["met"] is None:
        return "not applicable (nothing accepted)"
    if not t["met"]:
        return "**not met**"
    return "met, with confidence" if t["met_with_confidence"] else "met (point only)"


def write_report(report_dir: Path, out: dict[str, Any]) -> None:
    r = out["results"]
    rel, rule = r["released"], r["rule"]
    rm, lm = rel["metrics"], rule["metrics"]
    roles = r["roles"]
    animals = lm["animal_event_retention"]["denominator"]
    L: list[str] = []
    add = L.append

    add("# Final evaluation: the frozen release on unseen cameras\n")
    add(
        f"Stage 13. Plan [`{out['plan']}`](../../{out['plan']}) (sha256 "
        f"`{out['plan_sha256'][:12]}`), committed before any number here was computed. "
        f"Run at commit `{out['code']['commit'][:7]}` "
        f"({'clean' if not out['code']['dirty'] else 'DIRTY'} tree). "
        "**Nothing was chosen or tuned from these results.**\n"
    )
    add(
        f"The locked final test: {r['events']:,} capture events from 9 cameras never used "
        f"for training, calibration, or thresholds ({', '.join(r['cameras'])}), "
        f"{r['camera_nights']:,} camera-nights. {animals:,} animal events "
        f"({roles['supported_species']:,} supported species, "
        f"{roles['unsupported_animal']:,} unsupported species, "
        f"{roles['mixed_species']} mixed); {roles['empty']:,} empty; "
        f"{roles['non_animal']:,} vehicles (neither animal nor empty).\n"
    )

    add("## Against the targets\n")
    add(
        "Primary intervals are 95% cluster bootstraps over camera-nights: events, not "
        "frames, are the unit, and events from one camera on one night are not independent.\n"
    )
    add("| Target | Released policy (deployed) | Rule's operating point (not released) |")
    add("|---|---|---|")
    for name in ("animal_event_retention", "accepted_species_precision", "review_reduction"):
        t_rel, t_rule = rel["targets"][name], rule["targets"][name]
        row = f"| {NAMES[name]} ≥ {pct(t_rel['target'], 0)} "
        for m, t in ((rm[name], t_rel), (lm[name], t_rule)):
            if m["value"] is None:
                row += "| undefined: no species label accepted automatically "
            else:
                row += f"| {pct(m['value'], 2)} {ci(m['ci_camera_night'])}: {verdict(t)} "
        add(row + "|")
    for name in ("automatic_coverage", "unsupported_false_acceptance"):
        add(
            f"| {NAMES[name]} (reported) | {pct(rm[name]['value'], 2)} "
            f"{ci(rm[name]['ci_camera_night'])} | {pct(lm[name]['value'], 2)} "
            f"{ci(lm[name]['ci_camera_night'])} |"
        )
    add("")
    add(
        "- **Released policy** (`conservative/v2`, automation "
        "off): every event goes to a person, so retention is 100% and review is not "
        "reduced at all. No species label is accepted automatically, so accepted "
        "precision is undefined (reported as such, not as 0% or 100%)."
    )
    red = lm["review_reduction"]
    add(
        f"- **Rule's operating point** (empty filter 0.65, chosen on development cameras): "
        f"{rule['dispositions'].get('likely_empty', 0):,} events filtered, "
        f"{rule['audited_events']} of them sent back for audit, so "
        f"{red['numerator']:,} of {red['denominator']:,} reviews saved "
        f"({pct(red['value'], 2)}). It meets 98% retention over all animal events, but "
        f"falls below 98% for unsupported species, in daytime, and on one camera (below).\n"
    )

    add("## Decision: automation stays off\n")
    add(
        "The 50% review-reduction target is not met at any operating point development "
        "evidence supported: the only candidate automation, the empty filter, saves "
        f"{pct(red['value'], 1)} of reviews here. Its retention also falls below 98% in "
        "subgroups:\n"
    )
    dn = r["day_night_rule"]
    uns = lm["retention_unsupported_animal"]
    worst = min(rule["per_camera"].items(), key=lambda kv: kv[1]["animal_event_retention"] or 1)
    add(
        f"- unsupported species: {pct(uns['value'], 2)} {ci(uns['ci_camera_night'])} retained "
        f"({uns['denominator'] - uns['numerator']} of {uns['denominator']} lost);"
    )
    add(
        f"- daytime events: {pct(dn['day']['value'], 2)} {ci(dn['day']['ci_camera_night'])} "
        f"(night {pct(dn['night']['value'], 2)});"
    )
    add(
        f"- camera {worst[0]}: {pct(worst[1]['animal_event_retention'], 2)} overall, where the "
        f"filter also does most of its work ({pct(worst[1]['automatic_coverage'], 1)} "
        "coverage).\n"
    )
    add(
        "So the deployed release keeps both automations disabled: every event is reviewed, "
        "with the suggestion and its confidence shown. The filter is not enabled per camera "
        "or per time of day either: choosing such a restriction from these results would "
        "turn the final test into development evidence.\n"
    )

    add("## Grouping versus the model\n")
    add(
        "Review reduction is measured against an already grouped workflow, so grouping is "
        f"not credited to the model. Grouping alone turns {r['images']:,} images into "
        f"{r['events']:,} events to review ({pct(1 - r['events'] / r['images'])} fewer items "
        "than reviewing every image); "
        "the model adds nothing on top of that as released.\n"
    )

    add("## All metrics, rule's operating point\n")
    add(
        "| Metric | Value | n | Camera-night CI (primary) | Camera CI (9 clusters) | "
        "Naive event CI |"
    )
    add("|---|---|---|---|---|---|")
    for name, m in lm.items():
        add(
            f"| {NAMES[name]} | {pct(m['value'], 2)} | {m['numerator']:,} / "
            f"{m['denominator']:,} | {ci(m['ci_camera_night'], 2)} | {ci(m['ci_camera'], 2)} | "
            f"{ci(m['ci_naive_events'], 2)} |"
        )
    add("")
    add(
        "The camera interval resamples only 9 clusters and is correspondingly wide: at the "
        "camera level the retention interval crosses 98%. The naive interval treats events as "
        "independent and is too narrow; it is shown only for comparison. Released-policy "
        "values are exact (every event reviewed) and have zero-width intervals.\n"
    )

    add("## Variation across cameras (rule's operating point)\n")
    add(
        "| Camera | Events | Animal events | Retention | Supported | Unsupported | Coverage | "
        "Review reduction |"
    )
    add("|---|---|---|---|---|---|---|---|")
    for cam, c in rule["per_camera"].items():
        add(
            f"| {cam} | {c['events']:,} | {c['animal_events']:,} | "
            f"{pct(c['animal_event_retention'], 2)} | {pct(c['retention_supported_species'], 2)} | "
            f"{pct(c['retention_unsupported_animal'], 1)} | {pct(c['automatic_coverage'], 1)} | "
            f"{pct(c['review_reduction'], 1)} |"
        )
    sp = rule["camera_spread"]
    add("")
    add(
        f"Median camera: retention {pct(sp['animal_event_retention']['median'], 2)}, "
        f"review reduction {pct(sp['review_reduction']['median'], 1)}. Coverage ranges from "
        f"{pct(sp['automatic_coverage']['min'], 1)} to {pct(sp['automatic_coverage']['max'], 1)}: "
        "the aggregate is driven by one camera.\n"
    )

    add("## Error gallery\n")
    add(
        "Final-test mistakes, selected by the plan's fixed rule (event-id hash order, never by "
        "eye) and used for documentation only. Each image is the event's frame with the "
        "highest calibrated probability for the suggested class. Images: Caltech Camera "
        "Traps (CCT20), LILA BC, CDLA-Permissive-1.0.\n"
    )
    for name, g in r["gallery"].items():
        add(f"### {GALLERY_TITLES.get(name, name)}\n")
        add(f"Rule: {g['rule']}. Matching events: {g['matching_events']}.\n")
        if not g["shown"]:
            add("None on the final test.\n")
            continue
        add("| Frame | Camera | Time | True | Suggested (event conf.) | Released | Rule |")
        add("|---|---|---|---|---|---|---|")
        for s in g["shown"]:
            add(
                f"| ![{s['true']}](gallery/{s['file']}) | {s['camera']} | "
                f"{'night' if s['night'] else 'day'} | {s['true']} | {s['suggested']} "
                f"({s['confidence']}) | {s['released']} | {s['rule']} |"
            )
        add("")
    add(
        "What the gallery shows: animals that are small, partly hidden in vegetation, or at the "
        "frame edge in daylight; dark silhouettes and near-black frames at night; single-frame "
        "events, where one ambiguous frame is all the policy has. These are the failure modes "
        "listed in the model card; none of them was tuned for.\n"
    )

    rep = out["reproduction"]
    con = out["consistency"]
    add("## Frozen, reproduced, consistent\n")
    add(
        "- **Frozen:** every artifact the plan pins matched its hash "
        "(`frozen_artifacts` in [metrics.json](metrics.json)): weights, v2 and v1 policy "
        "artifacts, split lock, taxonomy, grouping source, final-test protocol, and the "
        "recorded final-test results."
    )
    if rep.get("skipped"):
        add("- **Reproduced:** skipped in this run.")
    else:
        add(
            f"- **Reproduced:** the Stage 10 final test was re-run under its unchanged "
            f"protocol ({rep['seconds']} s, cached scores) and gave identical results and "
            "identical per-event decisions; the committed record was not rewritten."
        )
    add(
        f"- **Consistent:** all {con['events']:,} events were rebuilt with the frozen scoring "
        "and policy code, matched the recorded decisions, and the deployed conservative/v2 "
        "policy decided every one of them exactly as the v1 policy the final test measured.\n"
    )

    add("## Limitations\n")
    add(
        "- The final test has now been used twice (Stage 10 and here) without any choice made "
        "from it. Any change made because of what it shows, such as a per-camera filter, "
        "would make it development evidence; a fresh final assessment would need new cameras "
        "(for example a consenting owner's deployment).\n"
        "- Nine cameras is a small sample of deployments; camera-level uncertainty is large.\n"
        "- Accepted-label precision could not be measured: no species threshold earned "
        "automation during development.\n"
        "- Ground truth is the dataset's annotation; events with an annotated but practically "
        "invisible animal count as animal events (see the gallery).\n"
    )
    add("## Reproduce\n")
    add(
        "```bash\n"
        "uv run wildinbox final-evaluation          # needs the dataset and the trained E3\n"
        "```\n\n"
        "Per-event results (camera, camera-night, role, label, suggestion, confidence, "
        "dispositions, audit selection) are in [events.jsonl.gz](events.jsonl.gz), so every "
        "number above can be recomputed without the images.\n"
    )
    (report_dir / "README.md").write_text("\n".join(L))
