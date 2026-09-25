"""Review views: the queue, filtered events, the audit queue, the timeline,
batch progress, and the getting-started guide. Each event card shows its frames,
a badge that keeps suggested, automatic, and confirmed labels visually
distinct, the review controls, and an expander with every frame, per-frame
predictions, the decision's provenance, and the full review history."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

import streamlit as st

from wildinbox.ui.client import ApiClient, ApiError
from wildinbox.ui.logic import REASON_TEXT, current_label, label_choices, review_for

PAGE_SIZE = 8
OTHER = "other species…"
CANT_TELL = "can't tell"


def when(ts: str | None) -> str:
    if not ts:
        return "time unknown"
    return datetime.fromisoformat(ts).strftime("%a %d %b %Y, %H:%M:%S")


def badge(event: dict[str, Any]) -> str:
    """One line that says whose label it is: a person's, automation's, or a suggestion."""
    label, source = current_label(event)
    d = event.get("decision") or {}
    conf = d.get("confidence")
    pct = f" ({conf:.0%})" if conf is not None else ""
    review = event.get("latest_review") or {}
    if source == "reviewed":
        return f":green-background[✅ **{label}**] confirmed by {review.get('reviewer')}"
    if source == "unresolved":
        return f":orange-background[❔ unresolved] by {review.get('reviewer')}"
    if source == "automatic":
        audit = " · :violet-background[🔎 audit sample]" if d.get("audit_selected") else ""
        return f":blue-background[⚙️ {label}{pct}] automatic{audit}"
    return f":gray-background[🤖 suggested: {label or 'none'}{pct}] not yet reviewed"


def review_controls(
    api: ApiClient, event: dict[str, Any], reviewer: str, classes: list[str]
) -> None:
    eid = event["id"]
    suggested = (event.get("decision") or {}).get("suggested_label")
    choices = ["choose a label", *label_choices(classes), OTHER, CANT_TELL]
    c1, c2, c3, c4 = st.columns([1.2, 1.6, 1.4, 0.8])
    disabled = not reviewer
    accept = c1.button(
        f"Accept: {suggested}" if suggested else "No suggestion",
        key=f"accept-{eid}",
        disabled=disabled or not suggested,
        type="primary",
    )
    picked = c2.selectbox("Label", choices, key=f"pick-{eid}", label_visibility="collapsed")
    other = c3.text_input(
        "Species name",
        key=f"other-{eid}",
        placeholder="species name",
        label_visibility="collapsed",
        disabled=picked != OTHER,
    )
    save = c4.button("Save", key=f"save-{eid}", disabled=disabled or picked == choices[0])
    chosen: str | None
    if accept:
        chosen = suggested
    elif save:
        chosen = (
            None if picked == CANT_TELL else (other.strip().lower() if picked == OTHER else picked)
        )
        if picked == OTHER and not chosen:
            st.warning('Type the species name, or choose "can\'t tell".')
            return
    else:
        return
    outcome, label = review_for(suggested, chosen)
    try:
        api.review(eid, reviewer, outcome, label)
    except ApiError as e:
        st.error(f"Review not saved: {e.detail}")
        return
    st.toast(f"Saved: {label or 'unresolved'} ({outcome})")
    st.rerun()


def event_detail(api: ApiClient, thumbnail: Any, event_id: str) -> None:
    """Every frame with its own prediction, the decision's provenance, and history."""
    detail = api.event(event_id)
    images = detail.get("images") or []
    cols = st.columns(4)
    for n, img in enumerate(images):
        col = cols[n % 4]
        data = thumbnail(api, img["id"], 480)
        if data:
            col.image(data, use_container_width=True)
        p = img.get("prediction")
        if p:
            probs = p.get("calibrated_probabilities") or p["class_probabilities"]
            best = max(probs, key=probs.__getitem__)
            col.caption(f"{img['filename']} · {best} {probs[best]:.0%}")
        else:
            col.caption(f"{img['filename']} · {img.get('frame_status', 'no prediction')}")
    d = detail.get("decision") or {}
    reasons = "; ".join(REASON_TEXT.get(r, r) for r in d.get("reasons", [])) or "none"
    st.markdown(
        f"**Decision:** {d.get('disposition')} · suggestion **{d.get('suggested_label')}** · "
        f"reasons: {reasons}  \n"
        f"Release `{d.get('model_release_id')}` · policy `{d.get('policy_version')}`"
        + (f"  \nAudit: {d['audit_rule']}" if d.get("audit_rule") else "")
    )
    history = detail.get("reviews") or []
    if history:
        st.markdown("**Review history** (newest last; earlier reviews are kept)")
        st.dataframe(
            [
                {
                    "when": r["created_at"][:19].replace("T", " "),
                    "reviewer": r["reviewer"],
                    "outcome": r["outcome"],
                    "label": r["confirmed_label"] or "unresolved",
                    "model suggested": r["suggested_label"],
                    "note": r["note"] or "",
                }
                for r in history
            ],
            hide_index=True,
        )
    else:
        st.caption("No reviews yet.")


def event_card(
    api: ApiClient, thumbnail: Any, event: dict[str, Any], reviewer: str, classes: list[str]
) -> None:
    decision = event.get("decision") or {}
    with st.container(border=True):
        left, right = st.columns([2, 3])
        left.markdown(
            f"**{event.get('camera_id') or 'unknown camera'}** · {when(event.get('start_at'))}  \n"
            f"{len(event['image_ids'])} frame(s)"
        )
        right.markdown(badge(event))
        reasons = [REASON_TEXT.get(r, r) for r in decision.get("reasons", [])]
        if reasons and decision.get("disposition") == "needs_review":
            right.caption("Needs review: " + "; ".join(reasons))
        ids = event["image_ids"][:6]
        cols = st.columns(6)  # fixed grid: a one-frame event is not drawn huge
        for col, image_id in zip(cols, ids, strict=False):
            data = thumbnail(api, image_id, 320)
            if data:
                col.image(data, use_container_width=True)
            else:
                col.caption("image unavailable")
        if len(event["image_ids"]) > len(ids):
            st.caption(f"+{len(event['image_ids']) - len(ids)} more frames")
        review_controls(api, event, reviewer, classes)
        with st.expander("All frames, predictions, and review history"):
            if st.toggle("Show details", key=f"detail-{event['id']}"):
                event_detail(api, thumbnail, event["id"])


def _paged(
    api: ApiClient,
    thumbnail: Any,
    reviewer: str,
    classes: list[str],
    key: str,
    empty_message: str,
    **params: Any,
) -> None:
    offset = st.session_state.get(f"offset-{key}", 0)
    try:
        page = api.events(limit=PAGE_SIZE, offset=offset, **params)
    except ApiError as e:
        st.error(e.detail)
        return
    total = page["total"]
    if not total:
        st.success(empty_message)
        return
    st.caption(f"{total} capture event(s). Capture events are not individual animals.")
    for event in page["events"]:
        event_card(api, thumbnail, event, reviewer, classes)
    prev, info, nxt = st.columns([1, 2, 1])
    if prev.button("Previous", disabled=offset == 0, key=f"prev-{key}"):
        st.session_state[f"offset-{key}"] = max(0, offset - PAGE_SIZE)
        st.rerun()
    info.caption(f"{offset + 1}-{min(offset + PAGE_SIZE, total)} of {total}")
    if nxt.button("Next", disabled=page["next_offset"] is None, key=f"next-{key}"):
        st.session_state[f"offset-{key}"] = page["next_offset"]
        st.rerun()


def review_page(
    api: ApiClient, thumbnail: Any, batch_id: str | None, reviewer: str, classes: list[str]
) -> None:
    st.header("Review queue")
    status = st.radio("Show", ["To review", "Reviewed", "All"], horizontal=True, key="status")
    if not reviewer:
        st.info("Enter your name in the sidebar to record reviews.")
    params: dict[str, Any] = {"batch_id": batch_id}
    if status == "To review":
        params |= {"disposition": "needs_review", "reviewed": False}
    elif status == "Reviewed":
        params |= {"reviewed": True}
    _paged(
        api,
        thumbnail,
        reviewer,
        classes,
        f"queue-{status}",
        "Nothing here. Every event in this view has been reviewed.",
        **params,
    )


def filtered_page(
    api: ApiClient, thumbnail: Any, batch_id: str | None, reviewer: str, classes: list[str]
) -> None:
    st.header("Automatically filtered")
    st.caption(
        "Events automation marked as likely empty. They are out of the review queue but "
        "never deleted: open any of them and, if an animal is there, label it to recover it."
    )
    release = api.version().get("active_release") or {}
    _paged(
        api,
        thumbnail,
        reviewer,
        classes,
        "filtered",
        "No event has been filtered automatically. The active release "
        f"(`{release.get('id')}`) filters only if its evaluation supported a threshold; "
        "this one sends every event to review.",
        batch_id=batch_id,
        disposition="likely_empty",
    )


def audit_page(
    api: ApiClient, thumbnail: Any, batch_id: str | None, reviewer: str, classes: list[str]
) -> None:
    st.header("Audit queue")
    st.caption(
        "A random share of events that automation filtered or labeled, sent to a person "
        "anyway so confident mistakes are measured. Each event shows the rule that chose it."
    )
    _paged(
        api,
        thumbnail,
        reviewer,
        classes,
        "audit",
        "No audit samples waiting. Audits come only from events automation handled; "
        "with automation off, nothing is sampled.",
        batch_id=batch_id,
        audit=True,
        reviewed=False,
    )


def timeline_page(api: ApiClient, thumbnail: Any, batch_id: str | None) -> None:
    st.header("Timeline")
    try:
        events = api.events(batch_id=batch_id, limit=500)["events"]
    except ApiError as err:
        st.error(err.detail)
        return
    if not events:
        st.info("No events yet. Upload a memory card on the Upload page.")
        return
    cameras = sorted({ev.get("camera_id") or "unknown camera" for ev in events})
    chosen = st.multiselect("Cameras", cameras, default=cameras, key="timeline-cameras")
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ev in events:
        if (ev.get("camera_id") or "unknown camera") in chosen:
            day = ev["start_at"][:10] if ev.get("start_at") else "no capture time"
            by_day[day].append(ev)
    st.caption(
        f"{sum(len(v) for v in by_day.values())} events on {len(by_day)} day(s), camera time. "
        "Badges: ✅ confirmed by a person · ⚙️ automatic · 🤖 suggestion not yet reviewed · "
        "❔ unresolved."
    )
    for day in sorted(by_day, reverse=True):
        with st.expander(f"{day} · {len(by_day[day])} event(s)"):
            for e in sorted(by_day[day], key=lambda x: x.get("start_at") or ""):
                c1, c2, c3 = st.columns([1, 3, 4])
                data = thumbnail(api, e["image_ids"][0], 320) if e["image_ids"] else None
                if data:
                    c1.image(data, use_container_width=True)
                c2.markdown(
                    f"{(e.get('start_at') or '')[11:19] or 'time unknown'} · "
                    f"{e.get('camera_id') or 'unknown camera'} · {len(e['image_ids'])} frame(s)"
                )
                c3.markdown(badge(e))


def batches_page(api: ApiClient) -> None:
    st.header("Batches")
    try:
        batches = api.batches()
    except ApiError as e:
        st.error(e.detail)
        return
    if not batches:
        st.info("No batches yet. Upload a memory card on the Upload page.")
        return
    if st.button("Refresh", key="batches-refresh"):
        st.rerun()
    for b in batches:
        s = api.batch(b["id"])
        p = s["progress"]
        with st.container(border=True):
            st.markdown(
                f"**{s['created_at'][:16].replace('T', ' ')}** · {s['counts']['images']} files · "
                f"{s['counts']['events']} events · status **{s['status']}**"
            )
            done = p["images_scored"] / max(p["images_to_score"], 1)
            st.progress(
                min(done, 1.0), text=f"{p['images_scored']} of {p['images_to_score']} scored"
            )
            if s["failures"]:
                with st.expander(f"{len(s['failures'])} file(s) could not be used"):
                    st.dataframe(
                        [
                            {"file": f["filename"], "stage": f["stage"], "error": f["error"]}
                            for f in s["failures"]
                        ],
                        hide_index=True,
                    )


def getting_started_page() -> None:
    st.header("Getting started")
    st.markdown(
        """
WildInbox turns a memory card of trail-camera photos into **capture events**
(photos from one camera trigger) and suggests what is in each.

1. **Upload** your photos (and a camera name) on the *Upload* page, and watch
   them being processed. *Batches* shows progress for every upload and any
   files that could not be used.
2. **Review** on the *Review queue* page. Enter your name in the sidebar first.
   For each event: **Accept** the suggestion, pick another species, choose
   **empty** if there is no animal, type a species the model does not know under
   **other species…**, or choose **can't tell**. Open *All frames, predictions,
   and review history* to see every frame and what the model thought of each.
3. **Browse** with the *Timeline* (events by day) and *Last night's visitors*.
4. **Check automation**: *Automatically filtered* lists events the model set
   aside as empty (label any of them to recover it), and the *Audit queue*
   holds a random sample of automatic decisions to check.
5. **Export** your observations as a CSV on the *Export* page.

Labels are shown differently depending on whose they are:
✅ confirmed by a person · ⚙️ decided automatically · 🤖 a suggestion nobody
has reviewed yet · ❔ unresolved. The model's original suggestion is never
overwritten; every review is added to the event's history.
"""
    )
