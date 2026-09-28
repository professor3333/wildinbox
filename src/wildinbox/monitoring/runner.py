"""How often the monitoring summary is computed.

A summary takes seconds with a large history. Callers that arrive while one is
being computed share it, and a finished summary is served for `max_age`
seconds, so the dashboard and a Prometheus scraper do not each start their own
and add to the load reviewers' requests compete with. After that, callers still
get the last summary at once while one background thread computes the next;
only the first call waits (and every call when `max_age` is 0). A failed
background refresh drops the last summary, so the next caller waits and sees
the error instead of numbers that silently stop changing. (Computing it in a
separate process was measured and did not help: reports/history/README.md.)
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def compute(
    database_url: str, lease_seconds: int, config_path: Path, api: dict[str, Any]
) -> dict[str, Any]:
    """One summary, from its own database session."""
    from wildinbox.monitoring.metrics import load_config, summary
    from wildinbox.storage.db import session_factory

    with session_factory(database_url)() as s:
        return summary(s, lease_seconds, load_config(config_path), api=api)


class MonitoringRunner:
    def __init__(
        self, database_url: str, lease_seconds: int, config_path: Path, max_age: float
    ) -> None:
        self.args = (database_url, lease_seconds, config_path)
        self.max_age = max_age
        self.lock = threading.Lock()
        self.running: Future[dict[str, Any]] | None = None
        self.latest: tuple[float, dict[str, Any]] | None = None

    def summary(self, api: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            latest, running = self.latest, self.running
            if latest and self.max_age > 0:
                if running is None and time.monotonic() - latest[0] >= self.max_age:
                    self.running = Future()
                    threading.Thread(
                        target=self._run, args=(self.running, api), daemon=True
                    ).start()
                return latest[1]
            owner = running is None
            if owner:  # nobody is computing one: this caller does
                running = self.running = Future()
        assert running is not None
        if owner:  # compute outside the lock; the others wait on `running`
            self._run(running, api)
        return running.result()

    def _run(self, running: Future[dict[str, Any]], api: dict[str, Any]) -> None:
        try:
            out = compute(*self.args, api)
        except Exception as e:
            log.exception("monitoring summary failed")
            running.set_exception(e)
            with self.lock:
                self.running, self.latest = None, None
        else:
            running.set_result(out)
            with self.lock:
                self.running, self.latest = None, (time.monotonic(), out)
