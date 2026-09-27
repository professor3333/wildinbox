"""WildInbox review interface (Streamlit). Run with `wildinbox ui`.

Talks only to the HTTP API (`WILDINBOX_API_URL`). Review views live in
`wildinbox.ui.review`; this module holds navigation, visitors, upload,
export, monitoring, and the study page.
"""

from __future__ import annotations

import os
import time
from datetime import date, datetime

import streamlit as st

from wildinbox.datasets.grouping import DEFAULT_GAP_SECONDS, MAX_GAP_SECONDS
from wildinbox.ui import monitoring, review
from wildinbox.ui.client import ApiClient, ApiError
from wildinbox.ui.logic import (
    batch_label,
    card_metadata,
    current_label,
    last_night,
    night_window,
    representative_frame,
)

PAGE_SIZE = 8
VISITOR_PAGE = 12
OTHER = "other species…"
CANT_TELL = "can't tell"


def client() -> ApiClient | None:
    """The API client for this browser session, or None until the person signs in
    (when the API requires tokens)."""
    if "client" in st.session_state:
        c: ApiClient = st.session_state["client"]
        return c
    url = os.environ.get("WILDINBOX_API_URL", "http://localhost:8000")
    try:
        ApiClient(url).whoami()
        st.session_state["client"] = ApiClient(url)
        return client()
    except ApiError as e:
        if e.status != 401:
            raise
    st.title("WildInbox")
    with st.form("sign-in"):
        token = st.text_input("Access token", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")
    if submitted and token.strip():
        api = ApiClient(url, token=token.strip())
        try:
            who = api.whoami()
        except ApiError:
            st.error("That token was not accepted.")
            return None
        st.session_state["client"] = api
        st.session_state["principal"] = who.get("principal")
        st.rerun()
    st.caption("Ask the person who runs this deployment for an access token.")
    return None


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
    principal = st.session_state.get("principal")
    if principal:  # token access: reviews are recorded under the signed-in identity
        st.sidebar.markdown(f"Signed in as **{principal}** (recorded with reviews)")
        reviewer = principal
    else:
        reviewer = st.sidebar.text_input("Your name (recorded with reviews)", key="reviewer")
    try:
        version = api.version()
        listing = api.batch_page()
    except (ApiError, OSError) as e:
        st.error(f"The API is not reachable: {e}")
        st.stop()
    release = version.get("active_release") or {}
    if release.get("is_test"):
        st.sidebar.warning(
            release.get("notice") or "TEST predictor: suggestions are not model output."
        )
    st.sidebar.caption(f"Release `{release.get('id')}`  \nPolicy `{release.get('policy_version')}`")
    # Options are batch ids; labels are display only, so two batches that look
    # alike stay separate and a selection survives its counts changing.
    labels = {b["id"]: batch_label(b) for b in listing["batches"]}
    selected = st.session_state.get("batch", review.ALL_BATCHES)
    if selected != review.ALL_BATCHES and selected not in labels:
        try:  # an older batch, chosen on the Batches page
            labels[selected] = batch_label(api.batch(selected))
        except ApiError:
            st.session_state["batch"] = review.ALL_BATCHES
    batch_id = st.sidebar.selectbox(
        "Batch",
        [review.ALL_BATCHES, *labels],
        key="batch",
        format_func=lambda i: "All batches" if i == review.ALL_BATCHES else labels[i],
    )
    if listing["total"] > len(listing["batches"]):
        st.sidebar.caption(
            f"Showing the {len(listing['batches'])} newest of {listing['total']} batches; "
            "open older ones from the Batches page."
        )
    return (
        page,
        None if batch_id == review.ALL_BATCHES else batch_id,
        reviewer.strip(),
        list(release.get("class_names") or []),
    )


# ------------------------------------------------------------------ visitors


def visitors_page(api: ApiClient, batch_id: str | None) -> None:
    st.header("Last night's visitors")
    # The default night comes from the newest events with a capture time;
    # undated events sort last and would otherwise crowd them out.
    dated = {"batch_id": batch_id, "start_after": "0001-01-01T00:00:00"}
    try:
        total = api.events(batch_id=batch_id, limit=1)["total"]
        n_dated = api.events(**dated, limit=1)["total"]
        recent = api.events(**dated, limit=500, offset=max(0, n_dated - 500))["events"]
    except ApiError as e:
        st.error(e.detail)
        return
    starts = [datetime.fromisoformat(e["start_at"]) for e in recent if e.get("start_at")]
    day = st.date_input("Night starting on", value=last_night(starts, date.today()), key="night")
    start, end = night_window(day)
    window = {
        "batch_id": batch_id,
        "start_after": start.isoformat(),
        "start_before": end.isoformat(),
    }
    # Every count is the API's total over the whole night, not over the page shown.
    key = f"visitors-{day}"
    offset = st.session_state.get(f"offset-{key}", 0)
    in_window = api.events(**window, limit=1)["total"]
    page = api.events(**window, animal=True, limit=VISITOR_PAGE, offset=offset)
    animals = page["total"]
    reviewed = api.events(**window, animal=True, reviewed=True, limit=1)["total"]
    undated = total - n_dated
    if undated:
        st.caption(f"{undated} event(s) have no capture time and cannot be placed on a night.")
    st.caption(
        f"{start:%a %d %b %Y %H:%M} to {end:%a %d %b %H:%M}: {animals} animal event(s) of "
        f"{in_window}, {reviewed} reviewed. Unreviewed labels are suggestions. Capture events "
        "are not individual animals or counts."
    )
    if not animals:
        st.info("No animal events in this window.")
        return
    cols = st.columns(3)
    for n, event in enumerate(page["events"]):
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
    review.pager(key, offset, VISITOR_PAGE, animals, page["next_offset"])


# ------------------------------------------------------------------ upload and export


def upload_page(api: ApiClient) -> None:
    st.header("Upload a memory card")
    with st.form("upload"):
        files = st.file_uploader(
            "Photos (JPEG or PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True
        )
        camera = st.text_input("Camera name (optional)")
        gap = st.number_input(
            "Group photos taken within this many seconds of each other",
            min_value=0.0,
            max_value=MAX_GAP_SECONDS,
            value=DEFAULT_GAP_SECONDS,
            step=1.0,
            help=(
                "Photos from one camera stay in one capture event while each is at most this "
                "long after the previous one. Sequence ids in the metadata file take precedence. "
                "Fixed for the batch once uploaded."
            ),
        )
        meta_file = st.file_uploader(
            "Card metadata (optional JSON: capture times, sequence ids, cameras per file)",
            type=["json"],
            help="Without it, photos are grouped by camera and EXIF capture time.",
        )
        submitted = st.form_submit_button("Upload and process", type="primary")
    if submitted and files:
        try:
            metadata = card_metadata(camera, meta_file.getvalue() if meta_file else None, gap)
        except ValueError as e:
            st.error(str(e))
            return
        try:
            batch = api.upload([(f.name, f.getvalue()) for f in files], metadata=metadata)
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
    grouping = s.get("grouping") or {}
    st.success(
        f"{s['counts']['events']} capture events ready for review "
        f"(grouped within {grouping.get('gap_seconds', DEFAULT_GAP_SECONDS):g} s)."
    )
    st.button(
        "Review this batch",
        type="primary",
        key="review-uploaded",
        on_click=review.focus_batch,
        args=(batch_id,),
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
    if api is None:
        return
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
