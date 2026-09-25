"""WildInbox review interface (Streamlit). Run with `wildinbox ui`.

Talks only to the HTTP API (`WILDINBOX_API_URL`). Review views live in
`wildinbox.ui.review`; this module holds navigation, visitors, upload,
export, monitoring, and the study page.
"""

from __future__ import annotations

import os
import time
from datetime import date, datetime
from typing import Any

import streamlit as st

from wildinbox.ui import monitoring, review
from wildinbox.ui.client import ApiClient, ApiError
from wildinbox.ui.logic import (
    current_label,
    is_visitor,
    last_night,
    night_window,
    representative_frame,
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


def batch_label(b: dict[str, Any]) -> str:
    return f"{b['created_at'][:16].replace('T', ' ')} · {b['images']} files · {b['events']} events"


def _focus_batch(api: ApiClient, batch_id: str) -> None:
    """Select a batch everywhere and open its review queue (runs before the rerun)."""
    for b in api.batches():
        if b["id"] == batch_id:
            st.session_state["batch"] = batch_label(b)
            st.session_state["page"] = "Review queue"
            return


# ------------------------------------------------------------------ sidebar


def sidebar(api: ApiClient) -> tuple[str, str | None, str, list[str]]:
    st.sidebar.title("WildInbox")
    st.sidebar.caption("Find the wildlife. Skip the empty frames.")
    page = st.sidebar.radio(
        "Page",
        [
            "Getting started",
            "Upload",
            "Batches",
            "Review queue",
            "Timeline",
            "Last night's visitors",
            "Automatically filtered",
            "Audit queue",
            "Export",
            "Monitoring",
            "Study",
        ],
        index=3,
        key="page",
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
    options = {"All batches": None} | {batch_label(b): b["id"] for b in batches}
    label = st.sidebar.selectbox("Batch", list(options), key="batch")
    return page, options[label], reviewer.strip(), list(release.get("class_names") or [])


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
    st.button(
        "Review this batch",
        type="primary",
        key="review-uploaded",
        on_click=_focus_batch,
        args=(api, batch_id),
    )
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
        for key in [k for k in st.session_state if str(k).startswith("offset")]:
            del st.session_state[key]  # a new view or batch starts on its first page
        st.session_state["_last_view"] = (page, batch_id)
    if page == "Getting started":
        review.getting_started_page()
    elif page == "Review queue":
        review.review_page(api, thumbnail, batch_id, reviewer, classes)
    elif page == "Timeline":
        review.timeline_page(api, thumbnail, batch_id)
    elif page == "Batches":
        review.batches_page(api)
    elif page == "Automatically filtered":
        review.filtered_page(api, thumbnail, batch_id, reviewer, classes)
    elif page == "Audit queue":
        review.audit_page(api, thumbnail, batch_id, reviewer, classes)
    elif page == "Last night's visitors":
        visitors_page(api, batch_id)
    elif page == "Upload":
        upload_page(api)
    elif page == "Monitoring":
        monitoring.page(api)
    elif page == "Study":
        from wildinbox.ui.study_page import study_page

        study_page(api, thumbnail)
    else:
        export_page(api, batch_id)


main()
