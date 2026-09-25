from __future__ import annotations

import re
import uuid
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import select, update

from wildinbox.api.app import create_app
from wildinbox.monitoring.metrics import (
    EventView,
    accuracy,
    alerts,
    api_health,
    audits,
    behavior,
    load_config,
    periods,
    psi,
    signals,
)
from wildinbox.monitoring.prometheus import render
from wildinbox.settings import Settings
from wildinbox.storage.db import session_factory
from wildinbox.storage.models import Image, Job, WorkerProcess
from wildinbox.workers import liveness
from wildinbox.workers.process import claim

from .test_app import NoopDispatcher, _files, _small_batch, database_url, settings  # noqa: F401
from .test_uploads import jpeg

REPO_ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(REPO_ROOT / "configs/monitoring/monitoring.yaml")


def _view(
    i: int, label: str, release: str = "r1", outcome: str | None = None, conf: float = 0.5
) -> EventView:
    return EventView(
        camera="north",
        start_at=datetime(2024, 1, 1) + timedelta(hours=i),
        release=release,
        disposition="needs_review",
        label=label,
        confidence=conf,
        reasons=["low_confidence"] if conf < 0.4 else ["automation_disabled"],
        review_outcome=outcome,
        reviewer="ranger" if outcome else None,
        night=[i % 2 == 0],
        blur=[100.0],
    )


def test_psi_is_zero_for_identical_mixes_and_grows_with_shift() -> None:
    a = Counter({"cat": 50, "dog": 50})
    assert psi(a, a) == 0
    assert psi(a, Counter({"cat": 90, "dog": 10})) > 0.25


def test_a_shifted_camera_raises_a_signal_that_names_a_release_change() -> None:
    views = [_view(i, "opossum") for i in range(150)] + [
        _view(150 + i, "cat", release="r2", conf=0.2) for i in range(100)
    ]
    sig = signals(views, CFG["drift"])["north"]
    cmp = sig["comparison"]
    assert cmp["label_psi"] > 1 and cmp["release_changed"] is True
    assert cmp["uncertainty_change"] == 1.0
    out = alerts(_ops(), {"north": sig}, accuracy(views, CFG["accuracy"]), CFG)
    messages = [a["message"] for a in out if a["area"] == "signal"]
    assert any("label mix shifted" in m and "release also changed" in m for m in messages)
    assert all(a["level"] == "info" for a in out if a["area"] == "signal")


def test_too_few_events_means_no_comparison() -> None:
    assert (
        signals([_view(i, "cat") for i in range(60)], CFG["drift"])["north"]["comparison"] is None
    )


def test_correction_rates_need_enough_reviews_and_alert_on_the_lower_bound() -> None:
    few = [_view(i, "cat", outcome="corrected") for i in range(10)]
    assert accuracy(few, CFG["accuracy"])["by_camera"]["north"]["correction_rate"] is None
    many = (
        [_view(i, "cat", outcome="corrected") for i in range(90)]
        + [_view(100 + i, "cat", outcome="confirmed") for i in range(10)]
        + [_view(200 + i, "cat") for i in range(100)]
    )
    acc = accuracy(many, CFG["accuracy"])
    north = acc["by_camera"]["north"]
    assert north["correction_rate"] == 0.9 and north["review_coverage"] == 0.5
    out = alerts(_ops(), {}, acc, CFG)
    assert [a["area"] for a in out] == ["accuracy"]


def _ops(**kw: object) -> dict[str, object]:
    base: dict[str, object] = {
        "window_days": 7,
        "jobs": {},
        "queued_waiting": 0,
        "oldest_queued_seconds": 0.0,
        "retrying": 0,
        "stale_leases": 0,
        "failed_jobs_in_window": [],
        "files_uploaded": 0,
        "files_unreadable": 0,
        "unreadable_rate": 0.0,
        "images_scored": 0,
        "frames_failed": 0,
        "processing_error_rate": 0.0,
        "images_scored_last_24h": 0,
        "cost_by_release": {},
    }
    return {**base, **kw}


def test_prometheus_output_is_valid_text_format() -> None:
    data = {
        "operations": _ops(
            jobs={"succeeded": 2}, cost_by_release={'r"1': {"seconds_per_1000_images": 80.0}}
        ),
        "signals": {"north": {"needs_review_share": 1.0}},
        "accuracy": {"by_camera": {"north": {"correction_rate": None, "review_coverage": 0.5}}},
        "alerts": [{"level": "warning"}],
    }
    text = render(data, {"GET /events": {"requests": 3, "p50_ms": 10.0, "p95_ms": 30.0}})
    sample = re.compile(r'^wildinbox_[a-z0-9_]+(\{([a-z_]+="([^"\\]|\\.)*",?)*\})? -?[0-9.e+-]+$')
    for line in text.strip().splitlines():
        assert line.startswith("# ") or sample.match(line), line
    assert 'wildinbox_seconds_per_1000_images{release="r\\"1"} 80.0' in text
    assert 'wildinbox_alerts{level="warning"} 1' in text


def test_monitoring_endpoints_and_stale_lease_alert(settings: Settings) -> None:  # noqa: F811
    with TestClient(create_app(settings)) as c:  # inline dispatch: processed on upload
        batch = c.post("/batches", files=_files(*_small_batch())).json()
        with session_factory(settings.database_url)() as s:
            qualities = [
                i.quality for i in s.scalars(select(Image)) if i.validation_status == "valid"
            ]
        assert qualities and all({"night", "blur", "brightness"} <= set(q) for q in qualities)

        m = c.get("/monitoring").json()
        ops = m["operations"]
        assert ops["images_scored"] == 3 and ops["files_unreadable"] == 4
        assert ops["jobs"] == {"succeeded": 1} and ops["cost_by_release"]
        assert ops["api_latency"]["POST /batches"]["requests"] == 1
        assert any(a["message"].startswith("many uploaded files") for a in m["alerts"])
        assert "not whether accuracy did" in m["notes"]["signals"]

        # A second batch whose worker "dies" holding the lease.
        other = c.post("/batches", files=_files(("fresh.jpg", jpeg(4242)))).json()
        job_id = uuid.UUID(other["job"]["id"]) if other["job"] else None
        assert job_id is not None
        factory = session_factory(settings.database_url)
        with factory() as s:
            s.execute(update(Job).where(Job.id == job_id).values(status="queued", attempts=0))
            s.commit()
        assert claim(factory, job_id, "doomed", settings) is not None
        with factory() as s:
            s.execute(
                update(Job)
                .where(Job.id == job_id)
                .values(heartbeat_at=datetime.now(UTC) - timedelta(hours=1))
            )
            s.commit()
        # The worker process itself had been reporting, then went silent (killed).
        liveness.beat(factory, "doomed")
        with factory() as s:
            s.execute(
                update(WorkerProcess)
                .where(WorkerProcess.id == "doomed")
                .values(last_seen_at=datetime.now(UTC) - timedelta(hours=1))
            )
            s.commit()
        m = c.get("/monitoring").json()
        assert m["operations"]["stale_leases"] == 1
        critical = {a["message"] for a in m["alerts"] if a["level"] == "critical"}
        assert any("lease" in x for x in critical)
        assert "no worker is running while jobs wait" in critical
        warning = {a["message"] for a in m["alerts"] if a["level"] == "warning"}
        assert any("died without shutting down" in x for x in warning)
        assert [w["id"] for w in m["operations"]["workers"]["died_in_window"]] == ["doomed"]
        text = c.get("/metrics").text
        assert "wildinbox_stale_leases 1" in text and 'wildinbox_alerts{level="critical"} 2' in text
        assert "wildinbox_workers_died_window 1" in text

        # A replacement worker starts: work can resume, the death stays on record.
        liveness.beat(factory, "replacement")
        m = c.get("/monitoring").json()
        critical = {a["message"] for a in m["alerts"] if a["level"] == "critical"}
        assert "no worker is running while jobs wait" not in critical
        assert [w["id"] for w in m["operations"]["workers"]["live"]] == ["replacement"]
        assert m["operations"]["workers"]["starts_in_window"] == 2
        # A clean shutdown is not a death.
        liveness.stopped(factory, "replacement")
        m = c.get("/monitoring").json()
        assert m["operations"]["workers"]["stopped_cleanly_in_window"] == 1
        assert len(m["operations"]["workers"]["died_in_window"]) == 1
        assert batch["id"]


NOW = datetime(2026, 9, 25, tzinfo=UTC)


def _ev(
    camera: str,
    batch: int,
    label: str,
    conf: float,
    days_ago: float,
    disposition: str = "needs_review",
    audit: bool = False,
    reviewed: str | None = None,
    blur: float = 100.0,
) -> EventView:
    return EventView(
        camera=camera,
        start_at=None,
        release="r1",
        disposition=disposition,
        label=label,
        confidence=conf,
        reasons=[],
        review_outcome=("confirmed" if reviewed == label else "corrected") if reviewed else None,
        reviewer="ranger" if reviewed else None,
        night=[False],
        blur=[blur],
        batch_id=f"{camera}-{batch}",
        batch_created_at=NOW - timedelta(days=days_ago),
        decided_at=NOW - timedelta(days=days_ago),
        audit_selected=audit,
        reviewed_label=reviewed,
    )


def test_a_changed_batch_moves_its_cameras_behaviour_indicators() -> None:
    usual = [_ev("north", b, "opossum", 0.8, 20 - b) for b in range(4) for _ in range(30)]
    same = [_ev("north", 9, "opossum", 0.8, 1) for _ in range(30)]
    changed = [_ev("north", 9, "empty" if i % 2 else "cat", 0.3, 1, blur=20.0) for i in range(30)]

    calm = behavior(usual + same, CFG["behavior"], NOW)
    cmp = calm["cameras"]["north"]["latest_batch"]["comparison"]
    assert cmp["label_psi"] == 0 and cmp["earlier_events"] == 120
    assert not [a for a in alerts(_ops(), {}, {"by_camera": {}}, CFG, beh=calm)]

    shifted = behavior(usual + changed, CFG["behavior"], NOW)
    cmp = shifted["cameras"]["north"]["latest_batch"]["comparison"]
    assert cmp["label_psi"] > 1 and cmp["confidence_psi"] > 1
    assert cmp["mean_confidence_change"] == -0.5 and cmp["blur_change"] == -0.8
    messages = [a["message"] for a in alerts(_ops(), {}, {"by_camera": {}}, CFG, beh=shifted)]
    assert any("confidence distribution" in m for m in messages)
    assert any("image sharpness" in m for m in messages)


def test_a_camera_mix_change_is_not_mistaken_for_a_shift_within_cameras() -> None:
    # Two stable cameras with different wildlife; the recent week is mostly "south".
    earlier = [_ev("north", 1, "opossum", 0.8, 10) for _ in range(90)] + [
        _ev("south", 1, "rabbit", 0.6, 10) for _ in range(10)
    ]
    recent = [_ev("north", 2, "opossum", 0.8, 2) for _ in range(10)] + [
        _ev("south", 2, "rabbit", 0.6, 2) for _ in range(90)
    ]
    cmp = periods(earlier + recent, CFG["behavior"], NOW)["comparison"]
    assert cmp["label_psi"]["total"] > 1
    assert cmp["label_psi"]["within"] < 0.01 and cmp["label_psi"]["mix"] > 1
    out = alerts(
        _ops(), {}, {"by_camera": {}}, CFG, beh=behavior(earlier + recent, CFG["behavior"], NOW)
    )
    assert any("only because the camera mix changed" in a["message"] for a in out)
    assert not any("within cameras (recent" in a["message"] for a in out)

    # The same mix, but north's behaviour changed: that is a shift within a camera.
    recent_changed = [_ev("north", 2, "cat", 0.8, 2) for _ in range(10)] + [
        _ev("south", 2, "rabbit", 0.6, 2) for _ in range(90)
    ]
    cmp = periods(earlier + recent_changed, CFG["behavior"], NOW)["comparison"]
    assert cmp["label_psi"]["within"] > 0.25


def test_audits_measure_false_empties_and_unknown_quality_without_labels() -> None:
    filtered = [
        _ev("north", 1, "empty", 0.95, 1, "likely_empty", audit=True, reviewed="empty")
        for _ in range(18)
    ] + [
        _ev("north", 1, "empty", 0.95, 1, "likely_empty", audit=True, reviewed="bobcat")
        for _ in range(2)
    ]
    unaudited_recovery = [
        _ev("north", 1, "empty", 0.95, 1, "likely_empty", reviewed="coyote") for _ in range(5)
    ]
    south = [_ev("south", 1, "empty", 0.95, 1, "likely_empty", audit=True) for _ in range(3)]
    aud = audits(filtered + unaudited_recovery + south, CFG["audits"])
    north = aud["by_camera"]["north"]
    # Only the random audit sample counts; the five recoveries were chosen by people.
    assert north["filtered_audited"] == 20 and north["false_empty"] == 2
    assert north["false_empty_rate"] == 0.1 and north["observed_quality"] == "measured"
    assert aud["by_camera"]["south"]["observed_quality"] == "unknown: no audit labels"
    assert aud["by_camera"]["south"]["audit_pending"] == 3
    out = alerts(_ops(), {}, {"by_camera": {}}, CFG, aud=aud)
    by_cam = {(a["camera"], a["message"]) for a in out}
    assert ("north", "an audit found an animal in an automatically filtered event") in by_cam
    assert ("south", "automation runs here with no audit labels yet: quality unknown") in by_cam


def test_api_server_errors_and_slow_metadata_routes_alert() -> None:
    routes = {
        "GET /events": {
            "requests": 40,
            "p50_ms": 80.0,
            "p95_ms": 900.0,
            "responses": 40,
            "client_errors": 0,
            "server_errors": 2,
        },
        "GET /images/{image_id}/original": {
            "requests": 40,
            "p50_ms": 800.0,
            "p95_ms": 2000.0,
            "responses": 40,
            "client_errors": 1,
            "server_errors": 0,
        },
    }
    api = api_health(routes, CFG["operations"])
    assert api["server_error_rate"] == 2 / 80
    assert api["slowest_metadata_route"] == {"route": "GET /events", "p95_ms": 900.0}
    messages = [a["message"] for a in alerts(_ops(), {}, {"by_camera": {}}, CFG, api=api)]
    assert any("server errors" in m for m in messages)
    assert any("GET /events p95" in m for m in messages)
