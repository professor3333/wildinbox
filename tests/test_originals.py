"""Originals are written from a separate process, off the API's interpreter lock."""

from __future__ import annotations

import hashlib
import os
import signal
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from wildinbox.api.app import create_app
from wildinbox.api.originals import CHUNK, OriginalsWriter, StoreError, put_all, writer
from wildinbox.settings import Settings
from wildinbox.storage.objects import LocalStore, original_key

from .test_app import NoopDispatcher, database_url, settings  # noqa: F401
from .test_uploads import jpeg


def _local(tmp_path: Path, **kw: Any) -> Settings:
    return Settings(object_store_backend="local", local_store_dir=tmp_path / "objects", **kw)


def _originals(n: int, salt: str = "") -> list[tuple[str, bytes, str]]:
    out = []
    for i in range(n):
        data = f"{salt}image-{i}".encode()
        out.append((original_key(hashlib.sha256(data).hexdigest()), data, "image/jpeg"))
    return out


def test_a_store_process_writes_every_original_in_chunks(tmp_path: Path) -> None:
    w = OriginalsWriter(_local(tmp_path, store_concurrency=4), processes=1)
    try:
        items = _originals(2 * CHUNK + 7)  # three chunks, the last partial
        w.put(items)
        w.put(items)  # already stored: content-addressed, nothing rewritten
    finally:
        w.close()
    store = LocalStore(tmp_path / "objects")
    assert all(store.get(k) == d for k, d, _ in items)


def test_a_failed_write_surfaces_in_the_api_process(tmp_path: Path) -> None:
    blocker = tmp_path / "objects"
    blocker.write_text("a file where the store's directory should be")
    w = OriginalsWriter(_local(tmp_path), processes=1)
    try:
        with pytest.raises(StoreError):
            w.put(_originals(3))
    finally:
        w.close()


def test_a_dead_store_process_is_replaced_on_the_next_upload(tmp_path: Path) -> None:
    w = OriginalsWriter(_local(tmp_path), processes=1)
    try:
        w.put(_originals(1, "first"))
        pool = w._pool
        assert pool is not None
        for pid in list(pool._processes):  # the OOM killer, say
            os.kill(pid, signal.SIGKILL)
        with pytest.raises(BrokenProcessPool):
            w.put(_originals(1, "lost"))
        w.put(_originals(1, "after"))  # a fresh process
        assert w._pool is not pool
    finally:
        w.close()
    store = LocalStore(tmp_path / "objects")
    assert all(store.exists(k) for k, _, _ in _originals(1, "after"))


def test_threads_in_process_when_configured_or_for_an_injected_store(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "objects")
    for s, injected in ((_local(tmp_path, store_processes=0), False), (_local(tmp_path), True)):
        put, close = writer(s, store, injected=injected)
        put(_originals(2, str(injected)))
        close()
        assert all(store.exists(k) for k, _, _ in _originals(2, str(injected)))
    put_all(store, [], 1)  # nothing to write


def test_an_upload_through_the_api_stores_originals_from_the_store_process(
    settings: Settings,  # noqa: F811
) -> None:
    s = settings.model_copy(update={"store_processes": 1})
    files = [(f"f{i}.jpg", jpeg(900 + i)) for i in range(3)]
    with TestClient(create_app(s, dispatcher=NoopDispatcher())) as c:
        res = c.post("/batches", files=[("files", (n, d, "image/jpeg")) for n, d in files])
        assert res.status_code == 202, res.text
        pool = c.app.state.put_originals.__self__._pool  # type: ignore[attr-defined]
        assert pool is not None  # the write went through the store process
    store = LocalStore(s.local_store_dir)
    for _, d in files:
        assert store.get(original_key(hashlib.sha256(d).hexdigest())) == d
