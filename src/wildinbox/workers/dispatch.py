"""How a queued job reaches a worker, and how workers run.

The job row in PostgreSQL is the source of truth; the queue only carries its
id. If enqueueing fails the job stays `queued` in the database and the
recovery loop dispatches it again, so a lost queue message never loses work.
Claiming (a lease in PostgreSQL) decides who processes a job, so a duplicate
queue message is harmless.
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Protocol

from wildinbox.settings import Settings
from wildinbox.storage.db import session_factory
from wildinbox.storage.objects import ObjectStore, store_from_settings
from wildinbox.workers.process import process_batch, recover_stale, worker_id

log = logging.getLogger(__name__)
JOB_FUNCTION = "wildinbox.workers.dispatch.run_job"


class Dispatcher(Protocol):
    def enqueue(self, job_id: uuid.UUID, attempt: int = 1) -> None: ...


def rq_job_id(job_id: uuid.UUID, attempt: int) -> str:
    # One queue message per attempt: a recovered job is never shadowed by the
    # message of the attempt whose worker died.
    return f"batch-job-{job_id}-a{attempt}"


class RQDispatcher:
    def __init__(self, settings: Settings) -> None:
        from redis import Redis
        from rq import Queue

        self.settings = settings
        self.queue = Queue(settings.queue_name, connection=Redis.from_url(settings.redis_url))

    def enqueue(self, job_id: uuid.UUID, attempt: int = 1) -> None:
        rq_id = rq_job_id(job_id, attempt)
        existing = self.queue.fetch_job(rq_id)
        if existing is not None and existing.get_status(refresh=True) in (
            "queued",
            "started",
            "deferred",
            "scheduled",
        ):
            return  # already on its way
        self.queue.enqueue(
            JOB_FUNCTION,
            str(job_id),
            job_id=rq_id,
            job_timeout=self.settings.job_timeout_seconds,
            result_ttl=24 * 3600,
            failure_ttl=7 * 24 * 3600,
        )


class InlineDispatcher:
    """Runs the job immediately in this process (tests and local development)."""

    def __init__(self, settings: Settings, store: ObjectStore) -> None:
        self.settings, self.store = settings, store

    def enqueue(self, job_id: uuid.UUID, attempt: int = 1) -> None:
        process_batch(
            session_factory(self.settings.database_url), self.store, job_id, self.settings
        )


def dispatcher_from_settings(settings: Settings, store: ObjectStore) -> Dispatcher:
    if settings.dispatch == "inline":
        return InlineDispatcher(settings, store)
    return RQDispatcher(settings)


def run_job(job_id: str) -> str:
    """Entry point executed by RQ workers."""
    settings = Settings()
    return process_batch(
        session_factory(settings.database_url),
        store_from_settings(settings),
        uuid.UUID(job_id),
        settings,
        worker_id(),
    )


def _recovery_loop(settings: Settings, stop: threading.Event) -> None:
    factory = session_factory(settings.database_url)
    dispatcher = RQDispatcher(settings)
    while not stop.is_set():
        try:
            recovered = recover_stale(factory, dispatcher, settings)
            if recovered:
                log.info("dispatched %d due job(s)", len(recovered))
        except Exception:
            log.exception("recovery sweep failed; retrying")
        stop.wait(settings.recovery_interval_seconds)


def run_worker(settings: Settings) -> None:
    """A worker process: recovers stale jobs on start and periodically, and runs
    jobs without forking, so each release's model is loaded once per process."""
    from redis import Redis
    from rq import Queue, SimpleWorker

    stop = threading.Event()
    threading.Thread(target=_recovery_loop, args=(settings, stop), daemon=True).start()
    conn = Redis.from_url(settings.redis_url)
    try:
        SimpleWorker(
            [Queue(settings.queue_name, connection=conn)], connection=conn, name=worker_id()
        ).work()
    finally:
        stop.set()
