"""How a queued job reaches a worker.

The job row in PostgreSQL is the source of truth; the queue only carries its
id. If enqueueing fails the job stays `queued` in the database and can be
dispatched again, so a lost queue message never loses work.
"""

from __future__ import annotations

import logging
import uuid
from typing import Protocol

from wildinbox.settings import Settings
from wildinbox.storage.db import session_factory
from wildinbox.storage.objects import ObjectStore, store_from_settings
from wildinbox.workers.process import process_batch

log = logging.getLogger(__name__)
JOB_FUNCTION = "wildinbox.workers.dispatch.run_job"


class Dispatcher(Protocol):
    def enqueue(self, job_id: uuid.UUID) -> None: ...


class RQDispatcher:
    def __init__(self, settings: Settings) -> None:
        from redis import Redis
        from rq import Queue

        self.queue = Queue(settings.queue_name, connection=Redis.from_url(settings.redis_url))

    def enqueue(self, job_id: uuid.UUID) -> None:
        # A deterministic RQ id means re-dispatching the same job replaces, not duplicates.
        self.queue.enqueue(
            JOB_FUNCTION,
            str(job_id),
            job_id=f"batch-job-{job_id}",
            job_timeout=3600,
            failure_ttl=7 * 24 * 3600,
        )


class InlineDispatcher:
    """Runs the job immediately in this process (tests and local development)."""

    def __init__(self, settings: Settings, store: ObjectStore) -> None:
        self.settings, self.store = settings, store

    def enqueue(self, job_id: uuid.UUID) -> None:
        process_batch(session_factory(self.settings.database_url), self.store, job_id)


def dispatcher_from_settings(settings: Settings, store: ObjectStore) -> Dispatcher:
    if settings.dispatch == "inline":
        return InlineDispatcher(settings, store)
    return RQDispatcher(settings)


def run_job(job_id: str) -> None:
    """Entry point executed by RQ workers."""
    settings = Settings()
    process_batch(
        session_factory(settings.database_url), store_from_settings(settings), uuid.UUID(job_id)
    )


def run_worker(settings: Settings) -> None:
    from redis import Redis
    from rq import Queue, Worker

    conn = Redis.from_url(settings.redis_url)
    Worker([Queue(settings.queue_name, connection=conn)], connection=conn).work()
