"""WildInbox review interface (Streamlit). Run with `wildinbox ui`.

Talks only to the HTTP API (`WILDINBOX_API_URL`). Pages: the review queue,
last night's visitors, upload, and export.
"""

from __future__ import annotations

import os
import time
from datetime import date, datetime
from typing import Any

import streamlit as st

from wildinbox.ui.client import ApiClient, ApiError
from wildinbox.ui.logic import (
    REASON_TEXT,
    current_label,
    is_visitor,
    label_choices,
    last_night,
    night_window,
    representative_frame,
    review_for,
)

PAGE_SIZE = 8
OTHER = "other species…"
CANT_TELL = "can't tell"


def client() -> ApiClient:
    if "client" not in st.session_state:
        st.session_state["client"] = ApiClient(
            os.environ.get("WILDINBOX_API_URL", "http://localhost:8000")
        )
    c: ApiClient = st.session_state["client"]
    return c


@st.cache_data(show_spinner=False, max_entries=2000)
def thumbnail(_api: ApiClient, image_id: str, size: int) -> bytes | None:
    return _api.thumbnail(image_id, size)


def when(ts: str | None) -> str:
    if not ts:
        return "time unknown"
    return datetime.fromisoformat(ts).strftime("%a %d %b %Y, %H:%M:%S")


# ------------------------------------------------------------------ sidebar


def sidebar(api: ApiClient) -> tuple[str, str | None, str, list[str]]:
    st.sidebar.title("WildInbox")
    st.sidebar.caption("Find the wildlife. Skip the empty frames.")
    page = st.sidebar.radio(
        "Page", ["Review queue", "Last night's visitors", "Upload", "Export"], key="page"
    )
    reviewer = st.sidebar.text_input("Your name (recorded with reviews)", key="reviewer")
    try:
        version = api.version()
        batches = api.batches()
    except (ApiError, OSError) as e:
        st.error(f"The API is not reachable: {e}")
        st.stop()
    release = version.get("active_release") or {}
    if release.get("is_test"):
        st.sidebar.warning(
            release.get("notice") or "TEST predictor: suggestions are not model output."
        )
    st.sidebar.caption(f"Release `{release.get('id')}`  \nPolicy `{release.get('policy_version')}`")
    options = {"All batches": None} | {
        f"{b['created_at'][:16].replace('T', ' ')} · {b['images']} files · {b['events']} events": b[
            "id"
        ]
        for b in batches
    }
    label = st.sidebar.selectbox("Batch", list(options), key="batch")
    return page, options[label], reviewer.strip(), list(release.get("class_names") or [])


# ------------------------------------------------------------------ review


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


def event_card(api: ApiClient, event: dict[str, Any], reviewer: str, classes: list[str]) -> None:
    decision = event.get("decision") or {}
    with st.container(border=True):
        left, right = st.columns([2, 3])
        left.markdown(
            f"**{event.get('camera_id') or 'unknown camera'}** · {when(event.get('start_at'))}  \n"
            f"{len(event['image_ids'])} frame(s)"
        )
        conf = decision.get("confidence")
        suggestion = decision.get("suggested_label") or "no suggestion"
        right.markdown(
            f"Suggested: **{suggestion}**" + (f" ({conf:.0%})" if conf is not None else "")
        )
        reasons = [REASON_TEXT.get(r, r) for r in decision.get("reasons", [])]
        if reasons:
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
        review = event.get("latest_review")
        if review:
            label, _ = current_label(event)
            st.caption(
                f"Reviewed by {review['reviewer']}: {label or 'unresolved'} ({review['outcome']})"
            )
        review_controls(api, event, reviewer, classes)


def review_page(api: ApiClient, batch_id: str | None, reviewer: str, classes: list[str]) -> None:
    st.header("Review queue")
    status = st.radio("Show", ["To review", "Reviewed", "All"], horizontal=True, key="status")
    reviewed = {"To review": False, "Reviewed": True, "All": None}[status]
    offset = st.session_state.get("offset", 0)
    try:
        page = api.events(batch_id=batch_id, reviewed=reviewed, limit=PAGE_SIZE, offset=offset)
    except ApiError as e:
        st.error(e.detail)
        return
    total = page["total"]
    if not reviewer:
        st.info("Enter your name in the sidebar to record reviews.")
    if not total:
        st.success("Nothing here. Every event in this view has been reviewed.")
        return
    st.caption(f"{total} capture event(s). Capture events are not individual animals.")
    for event in page["events"]:
        event_card(api, event, reviewer, classes)
    prev, info, nxt = st.columns([1, 2, 1])
    if prev.button("Previous", disabled=offset == 0, key="prev"):
        st.session_state["offset"] = max(0, offset - PAGE_SIZE)
        st.rerun()
    info.caption(f"{offset + 1}-{min(offset + PAGE_SIZE, total)} of {total}")
    if nxt.button("Next", disabled=page["next_offset"] is None, key="next"):
        st.session_state["offset"] = page["next_offset"]
        st.rerun()


# ------------------------------------------------------------------ visitors


def visitors_page(api: ApiClient, batch_id: str | None) -> None:
    st.header("Last night's visitors")
    try:
        total = api.events(batch_id=batch_id, limit=1)["total"]
        recent = api.events(batch_id=batch_id, limit=500, offset=max(0, total - 500))["events"]
    except ApiError as e:
        st.error(e.detail)
        return
    starts = [datetime.fromisoformat(e["start_at"]) for e in recent if e.get("start_at")]
    day = st.date_input("Night starting on", value=last_night(starts, date.today()), key="night")
    start, end = night_window(day)
    events = api.events(
        batch_id=batch_id, start_after=start.isoformat(), start_before=end.isoformat(), limit=500
    )["events"]
    visitors = [e for e in events if is_visitor(e)]
    undated = (
        api.events(batch_id=batch_id, limit=1)["total"]
        - api.events(batch_id=batch_id, start_after="0001-01-01T00:00:00", limit=1)["total"]
    )
    if undated:
        st.caption(f"{undated} event(s) have no capture time and cannot be placed on a night.")
    reviewed = sum(1 for e in visitors if e.get("latest_review"))
    st.caption(
        f"{start:%a %d %b %Y %H:%M} to {end:%a %d %b %H:%M}: {len(visitors)} animal event(s) of "
        f"{len(events)}, {reviewed} reviewed. Unreviewed labels are suggestions. Capture events "
        "are not individual animals or counts."
    )
    if not visitors:
        st.info("No animal events in this window.")
        return
    cols = st.columns(3)
    for n, event in enumerate(visitors):
        detail = api.event(event["id"])
        frame = representative_frame(detail)
        label, source = current_label(event)
        with cols[n % 3].container(border=True):
            data = thumbnail(api, frame["id"], 480) if frame else None
            if data:
                st.image(data, use_container_width=True)
            mark = {"reviewed": "reviewed", "automatic": "automatic", "unresolved": "unresolved"}
            st.markdown(f"**{label or 'unknown animal'}** · {mark.get(source, 'suggested')}")
            st.caption(
                f"{event.get('camera_id') or 'unknown camera'} · {when(event.get('start_at'))}"
            )


# ------------------------------------------------------------------ upload and export


def upload_page(api: ApiClient) -> None:
    st.header("Upload a memory card")
    with st.form("upload"):
        files = st.file_uploader(
            "Photos (JPEG or PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True
        )
        camera = st.text_input("Camera name (optional)")
        submitted = st.form_submit_button("Upload and process", type="primary")
    if submitted and files:
        try:
            batch = api.upload([(f.name, f.getvalue()) for f in files], camera.strip() or None)
        except ApiError as e:
            st.error(f"Upload rejected: {e.detail}")
            return
        st.session_state["uploaded"] = batch["id"]
    batch_id = st.session_state.get("uploaded")
    if not batch_id:
        return
    bar = st.progress(0.0, text="Queued")
    status = st.empty()
    for _ in range(3600):
        s = api.batch(batch_id)
        p = s["progress"]
        done = p["images_scored"] / max(p["images_to_score"], 1)
        bar.progress(min(done, 1.0), text=f"{p['images_scored']} of {p['images_to_score']} scored")
        job = s["job"]
        status.caption(
            f"Batch {batch_id}: {s['status']} (job {job['status']}, attempt {job['attempts']})"
        )
        if p["finished"]:
            break
        time.sleep(1)
    s = api.batch(batch_id)
    st.success(f"{s['counts']['events']} capture events ready for review.")
    if s["failures"]:
        st.warning(f"{len(s['failures'])} file(s) could not be used:")
        st.dataframe(
            [
                {"file": f["filename"], "stage": f["stage"], "error": f["error"]}
                for f in s["failures"]
            ],
            hide_index=True,
        )


def export_page(api: ApiClient, batch_id: str | None) -> None:
    st.header("Export observations")
    if not batch_id:
        st.info("Choose a batch in the sidebar.")
        return
    st.caption(
        "One row per capture event: the reviewed label where there is one, otherwise pending "
        "review, with the model release, calibration, and policy versions behind it."
    )
    if st.button("Prepare CSV", key="prepare"):
        st.session_state["export"] = (batch_id, api.export(batch_id, "csv"))
    ready = st.session_state.get("export")
    if ready and ready[0] == batch_id:
        st.download_button(
            "Download CSV",
            ready[1],
            file_name=f"wildinbox-{batch_id}-observations.csv",
            mime="text/csv",
        )


def main() -> None:
    st.set_page_config(page_title="WildInbox", layout="wide")
    api = client()
    page, batch_id, reviewer, classes = sidebar(api)
    if st.session_state.get("_last_view") != (page, batch_id):
        st.session_state["offset"] = 0
        st.session_state["_last_view"] = (page, batch_id)
    if page == "Review queue":
        review_page(api, batch_id, reviewer, classes)
    elif page == "Last night's visitors":
        visitors_page(api, batch_id)
    elif page == "Upload":
        upload_page(api)
    else:
        export_page(api, batch_id)


main()
