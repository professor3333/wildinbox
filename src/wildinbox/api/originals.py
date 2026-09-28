"""Writing an upload's originals to object storage, outside the API process.

The S3 client is pure Python per request (signing, HTTP, parsing). Run on
`store_concurrency` threads inside the API process, a 1,000-file upload kept
the interpreter lock busy for its whole store phase, and every other request
waited behind those threads: a 7 ms `GET /version` took up to 874 ms and an
event page up to 1.1 s. A separate process has its own interpreter lock, so
the API keeps answering while an upload stores.

`store_processes: 0` (or a store passed to `create_app`, which a child process
cannot share) keeps the threads in the API process.
"""

from __future__ import annotations

import multiprocessing
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool

from wildinbox.settings import Settings
from wildinbox.storage.objects import ObjectStore, store_from_settings

# (key, data, content type)
Original = tuple[str, bytes, str]

# Files per task: bounds each transfer to the child, and lets simultaneous
# uploads take turns instead of queueing whole batches.
CHUNK = 50

_store: ObjectStore | None = None
_concurrency = 1


class StoreError(RuntimeError):
    """A write failed in the store process (the original error, as text: the
    client library's exceptions do not all survive the trip between processes)."""


def put_all(store: ObjectStore, originals: Sequence[Original], concurrency: int) -> None:
    """Write each original unless it is already stored (content-addressed, so a
    stored key already holds these bytes). Re-raises the first failure."""

    def put(item: Original) -> None:
        key, data, content_type = item
        if not store.exists(key):
            store.put(key, data, content_type)

    with ThreadPoolExecutor(concurrency) as pool:
        list(pool.map(put, originals))


def _init(settings: Settings) -> None:
    global _store, _concurrency
    _store = store_from_settings(settings)
    _concurrency = settings.store_concurrency


def _put_chunk(originals: list[Original]) -> int:
    assert _store is not None
    try:
        put_all(_store, originals, _concurrency)
    except Exception as e:  # re-raised as text in the API process
        raise StoreError(f"{type(e).__name__}: {e}") from None
    return len(originals)


class OriginalsWriter:
    """Writes originals on `processes` spawned processes, each with its own
    store client and `store_concurrency` threads. Created lazily; a pool whose
    process died is replaced on the next upload."""

    def __init__(self, settings: Settings, processes: int) -> None:
        self.settings, self.processes = settings, processes
        self._pool: ProcessPoolExecutor | None = None
        self._lock = threading.Lock()

    def _executor(self) -> ProcessPoolExecutor:
        with self._lock:
            if self._pool is None:
                # spawn, not fork: the API process runs threads, and forking a
                # threaded process can copy locks held by other threads.
                self._pool = ProcessPoolExecutor(
                    self.processes,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_init,
                    initargs=(self.settings,),
                )
            return self._pool

    def put(self, originals: Sequence[Original]) -> None:
        pool = self._executor()
        chunks = [list(originals[i : i + CHUNK]) for i in range(0, len(originals), CHUNK)]
        try:
            futures: list[Future[int]] = [pool.submit(_put_chunk, c) for c in chunks]
            for f in futures:
                f.result()
        except BrokenProcessPool:
            with self._lock:
                if self._pool is pool:
                    self._pool = None
            pool.shutdown(wait=False, cancel_futures=True)
            raise

    def close(self) -> None:
        with self._lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)


def writer(
    settings: Settings, store: ObjectStore, injected: bool
) -> tuple[Callable[[Sequence[Original]], None], Callable[[], None]]:
    """(put, close) for the API: a store process when configured, else threads
    in this process (always for an injected store)."""
    if injected or settings.store_processes == 0:
        return (lambda originals: put_all(store, originals, settings.store_concurrency)), (
            lambda: None
        )
    w = OriginalsWriter(settings, settings.store_processes)
    return w.put, w.close
