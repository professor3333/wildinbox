"""The Monitoring page: operational health and model behavior, kept apart.

Operational health says whether processing works. Model behavior says what the
model decided and whether that changed; only audits and reviews say whether it
was right.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from wildinbox.ui.client import ApiClient, ApiError

LEVEL = {"critical": "🔴 Critical", "warning": "🟠 Warning", "info": "🔵 Signal"}


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def _ci(ci: list[float] | None) -> str:
    return f"{ci[0]:.0%}-{ci[1]:.0%}" if ci else "n/a"


def _num(x: float | None, fmt: str = "{:.3f}") -> str:
    return "n/a" if x is None else fmt.format(x)


def _alerts(alerts: list[dict[str, Any]], empty: str) -> None:
    if not alerts:
        st.success(empty)
    for a in alerts:
        parts = [a["area"]]
        if a.get("camera"):
            parts.append(f"camera {a['camera']}")
        text = (
            f"**{LEVEL[a['level']]}** ({' · '.join(parts)}): {a['message']} "
            f"({a['value']} vs limit {a['limit']})"
        )
        {"critical": st.error, "warning": st.warning}.get(a["level"], st.info)(text)


def operational(m: dict[str, Any]) -> None:
    ops = m["operations"]
    _alerts([a for a in m["alerts"] if a["area"] == "operations"], "No operational alerts.")
    st.caption(f"Processing health over the last {ops['window_days']} days.")
    c = st.columns(4)
    c[0].metric(
        "Jobs running / waiting", f"{ops['jobs'].get('running', 0)} / {ops['queued_waiting']}"
    )
    c[1].metric("Oldest waiting job", f"{ops['oldest_queued_seconds']:.0f} s")
    c[2].metric("Stale leases", ops["stale_leases"])
    c[3].metric("Failed jobs", len(ops["failed_jobs_in_window"]))
    c = st.columns(4)
    c[0].metric("Images scored (24 h)", ops["images_scored_last_24h"])
    c[1].metric("Frames failed", f"{ops['frames_failed']} ({_pct(ops['processing_error_rate'])})")
    c[2].metric("Unusable files", f"{ops['files_unreadable']} ({_pct(ops['unreadable_rate'])})")
    c[3].metric("Retrying jobs", ops["retrying"])

    lat = ops.get("job_latency")
    if lat:
        st.markdown("**Processing latency** (jobs finished in the window)")
        st.dataframe(
            [
                {"stage": name, "p50 (s)": lat[key]["p50"], "p95 (s)": lat[key]["p95"]}
                for name, key in (
                    ("waiting in queue", "queue_wait_seconds"),
                    ("processing", "run_seconds"),
                    ("upload to done", "upload_to_done_seconds"),
                )
            ],
            hide_index=True,
        )

    work = ops.get("workers")
    if work:
        st.markdown(f"**Workers** (restarts and deaths over the last {work['window_hours']} h)")
        c = st.columns(3)
        c[0].metric("Live workers", len(work["live"]))
        c[1].metric("Starts", work["starts_in_window"])
        c[2].metric("Died without shutting down", len(work["died_in_window"]))
        rows = [
            {
                "worker": w["id"],
                "state": state,
                "memory (MB)": w["rss_mb"] or w["peak_rss_mb"],
                "peak (MB)": w["peak_rss_mb"],
                "last seen": w["last_seen_at"][:19],
            }
            for state, group in (("live", work["live"]), ("died", work["died_in_window"]))
            for w in group
        ]
        if rows:
            st.dataframe(rows, hide_index=True)

    api = ops.get("api")
    if api:
        st.markdown("**API** (since the API process started)")
        c = st.columns(3)
        c[0].metric("Responses", api["responses"])
        c[1].metric(
            "Server errors (5xx)", f"{api['server_errors']} ({_pct(api['server_error_rate'])})"
        )
        slow = api["slowest_metadata_route"]
        c[2].metric(
            "Slowest metadata route, p95",
            f"{slow['p95_ms']:.0f} ms" if slow else "n/a",
            help=slow["route"] if slow else "Needs enough requests per route.",
        )
        with st.expander("Latency and errors by route"):
            st.dataframe([{"route": k, **v} for k, v in api["routes"].items()], hide_index=True)

    store = ops.get("storage")
    if store:
        st.markdown("**Storage**")
        st.metric(
            "Original photos stored", f"{store['original_gb']:.2f} GB ({store['images']} files)"
        )
        if store["daily"]:
            st.bar_chart(
                {d["day"]: d["bytes"] / 1e6 for d in store["daily"]},
                x_label="day",
                y_label="MB uploaded",
            )

    if ops.get("batches"):
        st.markdown("**Recent batches: processing cost**")
        st.dataframe(ops["batches"], hide_index=True)
    elif ops["cost_by_release"]:
        st.dataframe(
            [{"release": k, **v} for k, v in ops["cost_by_release"].items()], hide_index=True
        )


def behavior(m: dict[str, Any]) -> None:
    _alerts([a for a in m["alerts"] if a["area"] != "operations"], "No model-behavior alerts.")
    st.caption(
        "What the model decided, and whether that changed. A change is a warning sign, "
        "not a measure of accuracy: only audits and reviews measure mistakes."
    )
    beh = m.get("behavior")
    if beh:
        g = beh["global"]
        c = st.columns(4)
        c[0].metric("Filtered as empty", _pct(g["filtered_share"]))
        c[1].metric("Labeled automatically", _pct(g["auto_labeled_share"]))
        c[2].metric("Sent to review", _pct(g["needs_review_share"]))
        c[3].metric("Mean confidence", _num(g["mean_confidence"], "{:.2f}"))

        st.markdown("**Per camera, and each camera's latest batch against its earlier batches**")
        few = "too few events"
        rows = []
        for cam, v in beh["cameras"].items():
            lb = v["latest_batch"] or {}
            cmp = lb.get("comparison")
            top = ", ".join(f"{k} {n}" for k, n in list(v["labels"].items())[:3])
            rows.append(
                {
                    "camera": cam,
                    "events": v["events"],
                    "batches": v["batches"],
                    "filtered": _pct(v["filtered_share"]),
                    "auto-labeled": _pct(v["auto_labeled_share"]),
                    "to review": _pct(v["needs_review_share"]),
                    "top suggestions": top,
                    "latest batch events": (lb.get("latest") or {}).get("events"),
                    "label shift (PSI)": few if not cmp else f"{cmp['label_psi']:.3f}",
                    "confidence shift (PSI)": few if not cmp else f"{cmp['confidence_psi']:.3f}",
                    "mean confidence change": few
                    if not cmp
                    else _num(cmp["mean_confidence_change"], "{:+.2f}"),
                    "to-review change": few
                    if not cmp
                    else _num(cmp["needs_review_share_change"], "{:+.0%}"),
                    "sharpness change": few if not cmp else _num(cmp["blur_change"], "{:+.0%}"),
                }
            )
        st.dataframe(rows, hide_index=True)

        per = beh["periods"]
        st.markdown(
            f"**Time periods: last {per['period_days']} days against "
            f"the {per['period_days']} before**"
        )
        cmp = per["comparison"]
        if cmp is None:
            st.caption(
                f"Too few events to compare ({per['recent_events']} recent, "
                f"{per['earlier_events']} earlier)."
            )
        else:
            st.caption(
                "A global shift can come only from a different mix of cameras. "
                "'Camera mix' is the part explained by which cameras sent photos; "
                "'within cameras' is what changed on the cameras themselves."
            )
            st.dataframe(
                [
                    {
                        "distribution": name,
                        "total shift (PSI)": f"{cmp[key]['total']:.3f}",
                        "from camera mix": f"{cmp[key]['mix']:.3f}",
                        "within cameras": f"{cmp[key]['within']:.3f}",
                    }
                    for name, key in (
                        ("suggested labels", "label_psi"),
                        ("confidence", "confidence_psi"),
                    )
                ],
                hide_index=True,
            )
            if cmp["new_cameras"]:
                st.caption("New cameras this period: " + ", ".join(cmp["new_cameras"]))

    st.markdown("**Over capture time, per camera (no labels needed)**")
    rows = []
    few = "too few events"
    for cam, s in m["signals"].items():
        cmp = s["comparison"] or {}
        rows.append(
            {
                "camera": cam,
                "events": s["events"],
                "needs review": _pct(s["needs_review_share"]),
                "night frames": _pct(s["night_share"]),
                "label shift (PSI)": few if not cmp else f"{cmp['label_psi']:.3f}",
                "confidence shift (PSI)": few if not cmp else f"{cmp['confidence_psi']:.3f}",
                "release changed": few if not cmp else ("yes" if cmp["release_changed"] else "no"),
            }
        )
    st.dataframe(rows, hide_index=True)

    aud = m.get("audits")
    if aud:
        st.markdown("**Audited mistakes of automation**")
        st.caption(
            "Only the random audit sample counts, so these rates are not biased toward "
            "uncertain events. A camera with no audit labels has unknown observed quality."
        )
        st.dataframe(
            [
                {
                    "camera": cam,
                    "automatic events": v["automatic_events"],
                    "audit samples": v["audit_sampled"],
                    "audited": v["audit_reviewed"],
                    "false empties": f"{v['false_empty']} / {v['filtered_audited']}",
                    "false-empty 95% interval": _ci(v["false_empty_ci95"]),
                    "wrong species": f"{v['species_errors']} / {v['auto_labeled_audited']}",
                    "wrong-species 95% interval": _ci(v["species_error_ci95"]),
                    "observed quality": v["observed_quality"],
                }
                for cam, v in aud["by_camera"].items()
            ],
            hide_index=True,
        )

    st.markdown("**Accuracy measured from reviews**")
    st.caption(
        "How often reviewers corrected the suggestion, only for events someone reviewed "
        "(mostly uncertain ones); unreviewed events have unknown accuracy. "
        "Rates need at least 30 judged events."
    )
    for title, key in (("By camera", "by_camera"), ("By release", "by_release")):
        st.markdown(f"*{title}*")
        st.dataframe(
            [
                {
                    title.split()[-1]: k,
                    "events": v["events"],
                    "reviewed": f"{v['reviewed']} ({_pct(v['review_coverage'])})",
                    "correction rate": _pct(v["correction_rate"]),
                    "95% interval": _ci(v["correction_rate_ci95"]),
                    "unresolved": v["unresolved"],
                    "reviewers": ", ".join(v["reviewers"]),
                }
                for k, v in m["accuracy"][key].items()
            ],
            hide_index=True,
        )


def page(api: ApiClient) -> None:
    st.header("Monitoring")
    try:
        m = api.monitoring()
    except ApiError as e:
        st.error(e.detail)
        return
    levels = [a["level"] for a in m["alerts"]]
    st.caption(
        f"{levels.count('critical')} critical, {levels.count('warning')} warning, "
        f"{levels.count('info')} signal(s). Rules: configs/monitoring/monitoring.yaml "
        "(explained in docs/monitoring.md)."
    )
    ops_tab, model_tab = st.tabs(["Operational health", "Model behavior"])
    with ops_tab:
        operational(m)
    with model_tab:
        behavior(m)
