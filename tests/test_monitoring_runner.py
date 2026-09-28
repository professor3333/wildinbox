"""Monitoring summaries: shared by simultaneous callers and reused briefly."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from wildinbox.api.app import create_app
from wildinbox.monitoring import runner
from wildinbox.monitoring.runner import MonitoringRunner
from wildinbox.settings import Settings

from .test_app import database_url, settings  # noqa: F401


def test_the_api_reuses_a_summary_within_its_max_age(settings: Settings) -> None:  # noqa: F811
    cfg = settings.model_copy(update={"monitoring_max_age_seconds": 60})
    with TestClient(create_app(cfg)) as c:
        first = c.get("/monitoring").json()
        again = c.get("/monitoring").json()
        assert c.get("/metrics").status_code == 200
    assert "history" in first and "api_latency" in first["operations"]
    assert again["generated_at"] == first["generated_at"]


def test_simultaneous_callers_share_one_computation(
    settings: Settings,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[float] = []

    def slow(*args: Any) -> dict[str, Any]:
        calls.append(time.monotonic())
        time.sleep(0.3)
        return {"generated_at": str(len(calls)), "operations": {}}

    monkeypatch.setattr(runner, "compute", slow)
    r = MonitoringRunner(settings.database_url, 60, settings.monitoring_config, 0)
    results: list[dict[str, Any]] = []
    threads = [threading.Thread(target=lambda: results.append(r.summary({}))) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1 and {x["generated_at"] for x in results} == {"1"}
    assert r.summary({})["generated_at"] == "2"  # max age 0: the next call recomputes


def test_an_expired_summary_is_served_while_the_next_is_computed(
    settings: Settings,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    release = threading.Event()

    def gated(*args: Any) -> dict[str, Any]:
        calls.append(1)
        if len(calls) > 1:
            release.wait(5)
        return {"generated_at": str(len(calls)), "operations": {}}

    monkeypatch.setattr(runner, "compute", gated)
    r = MonitoringRunner(settings.database_url, 60, settings.monitoring_config, 0.05)
    assert r.summary({})["generated_at"] == "1"  # the first call waits
    time.sleep(0.1)
    started = time.monotonic()
    assert r.summary({})["generated_at"] == "1"  # expired: served, refresh starts
    assert r.summary({})["generated_at"] == "1"  # refresh running: no second one
    assert time.monotonic() - started < 0.5 and len(calls) == 2
    release.set()
    deadline = time.monotonic() + 5
    while r.summary({})["generated_at"] != "2":
        assert time.monotonic() < deadline
        time.sleep(0.01)


def test_a_failed_refresh_is_not_hidden_behind_the_old_summary(
    settings: Settings,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes: list[Any] = [{"generated_at": "1", "operations": {}}, RuntimeError("db down")]
    refreshed = threading.Event()

    def scripted(*args: Any) -> dict[str, Any]:
        outcome = outcomes.pop(0) if outcomes else RuntimeError("still down")
        refreshed.set()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(runner, "compute", scripted)
    r = MonitoringRunner(settings.database_url, 60, settings.monitoring_config, 0.05)
    r.summary({})
    time.sleep(0.1)
    refreshed.clear()
    assert r.summary({})["generated_at"] == "1"  # served while the refresh fails
    assert refreshed.wait(5)
    deadline = time.monotonic() + 5
    while r.running is not None:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    with pytest.raises(RuntimeError, match="still down"):
        r.summary({})
