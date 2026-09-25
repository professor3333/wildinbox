"""Worker liveness: each worker process keeps one `worker_processes` row fresh.

Monitoring reads these rows for live workers, their memory, restarts, and
workers that died without shutting down (their row goes stale with no
`stopped_at`). Job leases stay the source of truth for recovering work; this
only makes the workers themselves visible.
"""

from __future__ import annotations

import logging
import os
import resource
import socket
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, sessionmaker

from wildinbox.storage.models import WorkerProcess

log = logging.getLogger(__name__)


def memory() -> tuple[int | None, int]:
    """(current, peak) resident memory in bytes. Current needs /proc (Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak *= 1 if sys.platform == "darwin" else 1024  # bytes on macOS, KiB on Linux
    statm = Path("/proc/self/statm")
    current = (
        int(statm.read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE") if statm.exists() else None
    )
    return current, peak


def beat(factory: sessionmaker[Session], worker: str) -> None:
    """Create or refresh this worker's row."""
    now = datetime.now(UTC)
    current, peak = memory()
    values = {"last_seen_at": now, "rss_bytes": current, "peak_rss_bytes": peak}
    with factory() as s:
        s.execute(
            insert(WorkerProcess)
            .values(
                id=worker,
                hostname=socket.gethostname(),
                pid=os.getpid(),
                started_at=now,
                **values,
            )
            .on_conflict_do_update(index_elements=["id"], set_=values)
        )
        s.commit()


def stopped(factory: sessionmaker[Session], worker: str) -> None:
    """Record a clean shutdown, so monitoring does not count it as a death."""
    with factory() as s:
        s.execute(
            update(WorkerProcess)
            .where(WorkerProcess.id == worker)
            .values(stopped_at=datetime.now(UTC))
        )
        s.commit()


def heartbeat_loop(
    factory: sessionmaker[Session], worker: str, interval: float, stop: threading.Event
) -> None:
    while not stop.is_set():
        try:
            beat(factory, worker)
        except Exception:
            log.exception("worker heartbeat failed; retrying")
        stop.wait(interval)
