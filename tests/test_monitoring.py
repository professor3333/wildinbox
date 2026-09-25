from __future__ import annotations

import re
import uuid
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import select, update

from wildinbox.api.app import create_app
from wildinbox.monitoring.metrics import EventView, accuracy, alerts, load_config, psi, signals
from wildinbox.monitoring.prometheus import render
from wildinbox.settings import Settings
from wildinbox.storage.db import session_factory
from wildinbox.storage.models import Image, Job
from wildinbox.workers.process import claim

from .test_app import NoopDispatcher, _files, _small_batch, database_url, settings  # noqa: F401
from .test_uploads import jpeg

REPO_ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(REPO_ROOT / "configs/monitoring.yaml")


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
        m = c.get("/monitoring").json()
        assert m["operations"]["stale_leases"] == 1
        assert any(a["level"] == "critical" and "lease" in a["message"] for a in m["alerts"])
        text = c.get("/metrics").text
        assert "wildinbox_stale_leases 1" in text and 'wildinbox_alerts{level="critical"} 1' in text
        assert batch["id"]
