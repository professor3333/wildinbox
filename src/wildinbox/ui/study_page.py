"""The participant's view of the timed review study (configs/study/review_study.yaml).

One event at a time. In the "grouped" condition the reviewer sees only the
frames; in "suggested" also the model's suggestion, its confidence, the review
reasons, and a one-click Accept. Seconds run from the moment the event's frames
are on screen to the moment its label is saved. Choices go to the study log,
never to production reviews.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st
import yaml

from wildinbox.ui.client import ApiClient, ApiError
from wildinbox.ui.logic import REASON_TEXT, label_choices

PROTOCOL = Path("configs/study/review_study.yaml")
CANT_TELL = "can't tell"


def _state() -> dict[str, Any]:
    if "study" not in st.session_state:
        st.session_state["study"] = {"stage": "join"}
    s: dict[str, Any] = st.session_state["study"]
    return s


def _steps(a: dict[str, Any]) -> list[dict[str, Any]]:
    """Practice events, then each block's events, in order."""
    steps = [
        {"block": 0, "condition": p["condition"], "event_id": e}
        for p in a["practice"]
        for e in p["events"]
    ]
    for b in a["blocks"]:
        steps += [
            {"block": b["block"], "condition": b["condition"], "event_id": e} for e in b["events"]
        ]
    return steps


def study_page(api: ApiClient, thumbnail: Any) -> None:
    st.header("Review study")
    s = _state()
    if s["stage"] == "join":
        plan_id = st.text_input("Study code (from the organizer)", key="study-plan")
        code = st.text_input("Your participant code (no names)", key="study-code")
        if st.button("Join", type="primary", disabled=not (plan_id and code), key="study-join"):
            try:
                s["assignment"] = api.study_join(plan_id.strip(), code.strip())
            except ApiError as e:
                st.error(e.detail)
                return
            s.update({"plan": plan_id.strip(), "code": code.strip(), "stage": "consent", "i": 0})
            st.rerun()
        return

    if s["stage"] == "consent":
        protocol = yaml.safe_load(PROTOCOL.read_text())
        st.info(protocol["consent"])
        st.markdown(
            "You will label each capture event: pick the animal, **empty** if there is none, "
            "**other species** for an animal not in the list, or **can't tell**. Work at your "
            "normal pace; accuracy matters as much as speed. First a few practice events."
        )
        if st.button("I agree, start practice", type="primary", key="study-consent"):
            s["stage"] = "trial"
            st.rerun()
        return

    steps = _steps(s["assignment"])
    if s["stage"] == "rating":
        block = s["rating_block"]
        st.subheader(f"Block {block} done")
        rating = st.radio(
            "How hard was that block?",
            [1, 2, 3, 4, 5],
            format_func=lambda v: {1: "1 very easy", 3: "3", 5: "5 very hard"}.get(v, str(v)),
            horizontal=True,
            index=None,
            key=f"study-rating-{block}",
        )
        if st.button(
            "Continue", type="primary", disabled=rating is None, key=f"study-rated-{block}"
        ):
            api.study_rating(
                s["plan"],
                {
                    "participant": s["code"],
                    "block": block,
                    "condition": s["rating_condition"],
                    "difficulty": rating,
                },
            )
            s["stage"] = "trial" if s["i"] < len(steps) else "done"
            st.rerun()
        return

    if s["stage"] == "done" or s["i"] >= len(steps):
        st.success("Thank you, that's everything. You can close this page.")
        return

    step = steps[s["i"]]
    if step["block"] == 0:
        st.caption("Practice (not timed for the study)")
    else:
        before = [x for x in steps[: s["i"]] if x["block"] == step["block"]]
        total = sum(1 for x in steps if x["block"] == step["block"])
        st.caption(f"Block {step['block']} of 2 · event {len(before) + 1} of {total}")
    _trial(api, thumbnail, s, step, steps)


def _decide(
    api: ApiClient,
    s: dict[str, Any],
    step: dict[str, Any],
    steps: list[dict[str, Any]],
    label: str | None,
) -> None:
    now = time.time()
    key = f"{step['event_id']}"
    shown = s.setdefault("shown", {}).get(key, now)
    api.study_trial(
        s["plan"],
        {
            "participant": s["code"],
            "block": step["block"],
            "condition": step["condition"],
            "event_id": step["event_id"],
            "label": label,
            "seconds": round(now - shown, 3),
            "interactions": s.setdefault("interactions", {}).get(key, 0) + 1,
            "shown_at": datetime.fromtimestamp(shown, UTC).isoformat(),
            "decided_at": datetime.fromtimestamp(now, UTC).isoformat(),
        },
    )
    s["i"] += 1
    if step["block"] >= 1 and (s["i"] >= len(steps) or steps[s["i"]]["block"] != step["block"]):
        s.update(
            {
                "stage": "rating",
                "rating_block": step["block"],
                "rating_condition": step["condition"],
            }
        )


def _bump(s: dict[str, Any], key: str) -> None:
    s.setdefault("interactions", {})[key] = s["interactions"].get(key, 0) + 1


def _trial(
    api: ApiClient,
    thumbnail: Any,
    s: dict[str, Any],
    step: dict[str, Any],
    steps: list[dict[str, Any]],
) -> None:
    eid = step["event_id"]
    event = api.event(eid)
    frames = event["image_ids"][:6]
    cols = st.columns(3)
    for n, image_id in enumerate(frames):
        data = thumbnail(api, image_id, 480)
        if data:
            cols[n % 3].image(data, use_container_width=True)
    if step["condition"] == "suggested":
        d = event.get("decision") or {}
        conf = d.get("confidence")
        reasons = "; ".join(REASON_TEXT.get(r, r) for r in d.get("reasons", []))
        st.markdown(
            f"Suggested: **{d.get('suggested_label') or 'none'}**"
            + (f" ({conf:.0%})" if conf is not None else "")
        )
        if reasons:
            st.caption(reasons)
        if d.get("suggested_label") and st.button(
            f"Accept: {d['suggested_label']}", type="primary", key=f"study-accept-{eid}"
        ):
            _decide(api, s, step, steps, d["suggested_label"])
            st.rerun()
    classes = label_choices(s["assignment"]["classes"])
    options = [*classes, "other species", CANT_TELL]
    # Two rows of five equal-width buttons, so no label is truncated.
    cells = [c for _ in range(0, len(options), 5) for c in st.columns(5)]
    for cell, label in zip(cells, classes, strict=False):
        if cell.button(label, key=f"study-{eid}-{label}", use_container_width=True):
            _decide(api, s, step, steps, label)
            st.rerun()
    other_cell, cant_cell = cells[len(classes)], cells[len(classes) + 1]
    if other_cell.button(
        "other species",
        key=f"study-{eid}-other",
        on_click=_bump,
        args=(s, eid),
        use_container_width=True,
    ):
        s.setdefault("other", {})[eid] = True
    if cant_cell.button(CANT_TELL, key=f"study-{eid}-cant", use_container_width=True):
        _decide(api, s, step, steps, None)
        st.rerun()
    if s.get("other", {}).get(eid):
        name = st.text_input("Species name", key=f"study-{eid}-name")
        if st.button("Save", key=f"study-{eid}-save", disabled=not name.strip()):
            _decide(api, s, step, steps, name.strip().lower())
            st.rerun()
    # The clock starts once this event's frames are on screen.
    s.setdefault("shown", {}).setdefault(eid, time.time())
