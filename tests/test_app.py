"""End-to-end API tests against PostgreSQL.

Set WILDINBOX_TEST_DATABASE_URL (CI provides a Postgres service). The schema
is created by running the Alembic migrations, so these tests also check them.
"""

from __future__ import annotations

import io
import json
import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select, text, update

from wildinbox.api.app import create_app
from wildinbox.settings import Settings
from wildinbox.storage.db import session_factory
from wildinbox.storage.models import Batch, Decision, Event, Image, Job, Prediction
from wildinbox.storage.objects import LocalStore
from wildinbox.workers.process import process_batch

from .conftest import REPO_ROOT
from .test_uploads import jpeg

DB_URL = os.environ.get("WILDINBOX_TEST_DATABASE_URL")
TABLES = (
    "reviews, decisions, predictions, images, events, jobs, batches, "
    "release_activations, model_releases, worker_processes"
)


@pytest.fixture(scope="session")
def database_url() -> str:
    if not DB_URL:
        if os.environ.get("CI"):
            pytest.fail("WILDINBOX_TEST_DATABASE_URL must be set in CI")
        pytest.skip("set WILDINBOX_TEST_DATABASE_URL to run database tests")
    engine = create_engine(DB_URL)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public"))
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", DB_URL)
    command.upgrade(cfg, "head")
    return DB_URL


class NoopDispatcher:
    def __init__(self) -> None:
        self.enqueued: list[uuid.UUID] = []

    def enqueue(self, job_id: uuid.UUID, attempt: int = 1) -> None:
        self.enqueued.append(job_id)


class BrokenDispatcher:
    def enqueue(self, job_id: uuid.UUID, attempt: int = 1) -> None:
        raise ConnectionError("redis is down")


@pytest.fixture
def settings(database_url: str, tmp_path: Path) -> Settings:
    with create_engine(database_url).begin() as conn:
        conn.execute(text(f"TRUNCATE {TABLES} CASCADE"))
    return Settings(
        database_url=database_url,
        object_store_backend="local",
        local_store_dir=tmp_path / "objects",
        dispatch="inline",
        config_path=REPO_ROOT / "configs/example.yaml",
        max_files_per_batch=10,
        max_file_bytes=100_000,
        max_batch_bytes=500_000,
        retry_backoff_seconds=0,
        auth="disabled",
        # A fresh summary per request, so tests see each change;
        # tests/test_monitoring_runner.py covers the reuse.
        monitoring_max_age_seconds=0,
        # Originals written by threads in this process: spawning a store process
        # per test app doubles the suite's time; tests/test_originals.py covers it.
        store_processes=0,
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as c:
        yield c


def _files(*items: tuple[str, bytes]) -> list[tuple[str, tuple[str, bytes, str]]]:
    return [("files", (name, data, "application/octet-stream")) for name, data in items]


def _count(settings: Settings, model: Any) -> int:
    with session_factory(settings.database_url)() as s:
        return int(s.scalar(select(func.count()).select_from(model)) or 0)


SMALL_BATCH = (
    ("night-1.jpg", jpeg(1, exif_time="2024:05:01 21:03:00")),
    ("night-2.jpg", jpeg(2, exif_time="2024:05:01 21:03:01")),
    ("day.png", None),  # filled below
    ("notes.gif", b"GIF89a" + b"\x00" * 20),
    ("empty.jpg", b""),
    ("huge.jpg", b"\xff\xd8\xff" + b"\x00" * 200_000),
    ("broken.jpg", b"\xff\xd8\xff\xe0" + b"corrupt" * 20),
)


def _small_batch() -> list[tuple[str, bytes]]:
    import io

    from PIL import Image as PILImage

    buf = io.BytesIO()
    PILImage.new("RGB", (40, 30), (120, 80, 40)).save(buf, "PNG")
    return [(n, d if d is not None else buf.getvalue()) for n, d in SMALL_BATCH]


def test_small_batch_travels_from_upload_to_results(client: TestClient) -> None:
    items = _small_batch()
    res = client.post(
        "/batches",
        files=_files(*items),
        data={"metadata": json.dumps({"camera_id": "north-trail"})},
    )
    assert res.status_code == 202, res.text
    batch = res.json()
    assert batch["release"]["is_test"] is True and "TEST PREDICTOR" in batch["release"]["notice"]

    summary = client.get(f"/batches/{batch['id']}").json()
    assert summary["status"] == "completed_with_errors"
    assert summary["job"]["status"] == "succeeded" and summary["job"]["attempts"] == 1
    assert summary["counts"] == {
        "images": 7,
        "pending": 0,
        "valid": 3,
        "invalid": 4,
        "duplicate": 0,
        "events": 2,
        "processing_failed": 0,
    }
    assert summary["progress"]["images_scored"] == 3 and summary["progress"]["finished"]
    assert {f["filename"] for f in summary["failures"]} == {
        "notes.gif",
        "empty.jpg",
        "huge.jpg",
        "broken.jpg",
    }

    images = {
        i["filename"]: i for i in client.get(f"/batches/{batch['id']}/images").json()["images"]
    }
    errors = {
        n: i["validation_error"] for n, i in images.items() if i["validation_status"] == "invalid"
    }
    assert errors["notes.gif"].startswith("unsupported_type") and "GIF" in errors["notes.gif"]
    assert errors["empty.jpg"].startswith("empty_file")
    assert errors["huge.jpg"].startswith("file_too_large")
    assert errors["broken.jpg"].startswith("unreadable image")
    assert images["night-1.jpg"]["captured_at"] == "2024-05-01T21:03:00"  # from EXIF

    events = client.get("/events", params={"batch_id": batch["id"]}).json()["events"]
    assert len(events) == 2
    night = next(e for e in events if len(e["image_ids"]) == 2)
    assert {images["night-1.jpg"]["id"], images["night-2.jpg"]["id"]} == set(night["image_ids"])
    for e in events:
        assert e["decision"]["disposition"] == "needs_review"
        assert e["decision"]["reasons"] == ["automation_disabled"]
        assert e["release"]["is_test"] is True
        assert e["camera_id"] == "north-trail"

    detail = client.get(f"/events/{night['id']}").json()
    for img in detail["images"]:
        probs = img["prediction"]["class_probabilities"]
        assert abs(sum(probs.values()) - 1) < 1e-6
        assert img["prediction"]["model_release_id"] == "test-predictor-v0"

    original = client.get(images["night-1.jpg"]["original_url"])
    assert original.status_code == 200 and original.content == items[0][1]
    assert original.headers["content-type"] == "image/jpeg"


def test_resubmitting_the_same_request_creates_no_duplicates(
    client: TestClient, settings: Settings
) -> None:
    files = _files(("a.jpg", jpeg(10)), ("b.jpg", jpeg(11)))
    first = client.post("/batches", files=files)
    again = client.post("/batches", files=files)
    assert first.status_code == 202 and again.status_code == 200
    assert again.json()["id"] == first.json()["id"] and again.json()["duplicate_request"] is True
    assert _count(settings, Batch) == 1 and _count(settings, Job) == 1

    keyed = client.post(
        "/batches",
        files=_files(("c.jpg", jpeg(12))),
        headers={"Idempotency-Key": "card-2024-05-01"},
    )
    keyed_again = client.post(
        "/batches",
        files=_files(("c.jpg", jpeg(12))),
        headers={"Idempotency-Key": "card-2024-05-01"},
    )
    assert keyed_again.json()["id"] == keyed.json()["id"]
    conflict = client.post(
        "/batches",
        files=_files(("d.jpg", jpeg(13))),
        headers={"Idempotency-Key": "card-2024-05-01"},
    )
    assert conflict.status_code == 409 and conflict.json()["error"] == "idempotency_conflict"
    assert _count(settings, Job) == 2


def test_simultaneous_identical_uploads_create_one_batch(
    client: TestClient, settings: Settings
) -> None:
    """Both requests pass the duplicate check before either inserts; the loser
    hits the unique request key and answers with the winner's batch."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    barrier = threading.Barrier(2, timeout=30)
    put = client.app.state.put_originals  # type: ignore[attr-defined]

    def put_after_both_checked(originals: Any) -> None:
        barrier.wait()
        put(originals)

    client.app.state.put_originals = put_after_both_checked  # type: ignore[attr-defined]
    files = _files(("a.jpg", jpeg(20)), ("b.jpg", jpeg(21)), ("a-again.jpg", jpeg(20)))
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: client.post("/batches", files=files), range(2)))
    assert sorted(r.status_code for r in results) == [200, 202]
    assert len({r.json()["id"] for r in results}) == 1
    assert _count(settings, Batch) == 1 and _count(settings, Job) == 1
    assert _count(settings, Image) == 3  # the loser's rows were rolled back


def test_rerunning_a_job_does_not_duplicate_results(client: TestClient, settings: Settings) -> None:
    batch = client.post(
        "/batches", files=_files(("a.jpg", jpeg(20)), ("b.jpg", jpeg(21)), ("c.jpg", jpeg(22)))
    ).json()
    before = [_count(settings, m) for m in (Event, Prediction, Decision)]
    factory = session_factory(settings.database_url)
    with factory() as s:
        s.execute(update(Job).values(status="queued"))
        s.commit()
    process_batch(factory, LocalStore(settings.local_store_dir), uuid.UUID(batch["job"]["id"]))
    assert [_count(settings, m) for m in (Event, Prediction, Decision)] == before
    job = client.get(f"/jobs/{batch['job']['id']}").json()
    assert job["status"] == "succeeded" and job["attempts"] == 2


def test_batch_limits_reject_whole_request(client: TestClient, settings: Settings) -> None:
    too_many = client.post("/batches", files=_files(*[(f"{i}.jpg", jpeg(i)) for i in range(11)]))
    assert too_many.status_code == 413 and too_many.json()["error"] == "too_many_files"
    too_big = client.post(
        "/batches", files=_files(*[(f"{i}.jpg", b"\xff\xd8\xff" + bytes(99_000)) for i in range(6)])
    )
    assert too_big.status_code == 413 and too_big.json()["error"] == "batch_too_large"
    assert client.post("/batches", data={"x": "1"}).json()["error"] == "no_files"
    bad_meta = client.post(
        "/batches", files=_files(("a.jpg", jpeg())), data={"metadata": '{"camera": 1}'}
    )
    assert bad_meta.status_code == 422 and bad_meta.json()["error"] == "invalid_metadata"
    assert _count(settings, Batch) == 0


def _multipart(size: int, boundary: str = "b0undary") -> tuple[bytes, bytes, bytes]:
    head = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="files"; '
        'filename="big.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'
    ).encode()
    return head, b"\xff\xd8\xff" + bytes(size - 3), f"\r\n--{boundary}--\r\n".encode()


def test_streamed_batch_counts_actual_file_sizes(settings: Settings) -> None:
    # Without Content-Length the header check is skipped; the batch limit must
    # still use each file's real size, not the bytes read up to the file limit.
    small = settings.model_copy(update={"max_file_bytes": 100, "max_batch_bytes": 500})
    head, body, tail = _multipart(5_000)
    with TestClient(create_app(small)) as c:
        r = c.post(
            "/batches",
            content=iter([head, body, tail]),  # chunked: no Content-Length header
            headers={"content-type": "multipart/form-data; boundary=b0undary"},
        )
    assert r.status_code == 413 and r.json()["error"] == "batch_too_large"
    assert _count(small, Batch) == 0


def test_streamed_batch_is_rejected_while_it_arrives(settings: Settings) -> None:
    import anyio

    from wildinbox.api.app import MULTIPART_OVERHEAD

    budget = settings.max_batch_bytes + MULTIPART_OVERHEAD
    chunk = 64 * 1024
    head, _, _ = _multipart(3)
    pulled = 0
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        nonlocal pulled
        pulled += 1
        body = head if pulled == 1 else b"\x00" * chunk
        return {"type": "http.request", "body": body, "more_body": pulled < 1_000}  # ~64 MB

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/batches",
        "raw_path": b"/batches",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"multipart/form-data; boundary=b0undary")],
        "client": ("test", 1),
        "server": ("test", 80),
    }
    app = create_app(settings)
    with TestClient(app):  # runs the lifespan
        anyio.run(app, scope, receive, send)
    assert sent[0]["status"] == 413
    assert json.loads(sent[1]["body"])["error"] == "batch_too_large"
    assert pulled <= budget // chunk + 2  # stopped reading once over budget
    assert _count(settings, Batch) == 0


def test_duplicate_images_are_recorded_not_reprocessed(client: TestClient) -> None:
    a = jpeg(30)
    first = client.post("/batches", files=_files(("a.jpg", a))).json()
    second = client.post(
        "/batches", files=_files(("copy-of-a.jpg", a), ("a-again.jpg", a), ("new.jpg", jpeg(31)))
    ).json()
    assert second["counts"]["duplicate"] == 2 and second["counts"]["valid"] == 1
    first_image = client.get(f"/batches/{first['id']}/images").json()["images"][0]
    dupes = [
        i
        for i in client.get(f"/batches/{second['id']}/images").json()["images"]
        if i["validation_status"] == "duplicate"
    ]
    assert {d["duplicate_of"] for d in dupes} == {first_image["id"]}


def test_undispatched_job_stays_queued_and_failures_retry_then_fail(settings: Settings) -> None:
    with TestClient(create_app(settings, dispatcher=BrokenDispatcher())) as c:
        res = c.post("/batches", files=_files(("a.jpg", jpeg(40))))
        assert res.status_code == 202
        assert res.json()["status"] == "queued" and res.json()["job"]["status"] == "queued"

    store = LocalStore(settings.local_store_dir)
    with session_factory(settings.database_url)() as s:
        img = s.scalars(select(Image)).one()
        store._path(img.storage_key or "").unlink()  # the original vanished
    job_id = uuid.UUID(res.json()["job"]["id"])
    factory = session_factory(settings.database_url)
    # Bounded retries: each attempt fails and is recorded; the last is terminal.
    assert process_batch(factory, store, job_id, settings) == "queued"
    assert process_batch(factory, store, job_id, settings) == "queued"
    assert process_batch(factory, store, job_id, settings) == "failed"
    assert process_batch(factory, store, job_id, settings) == "skipped"
    with TestClient(create_app(settings, dispatcher=NoopDispatcher())) as c:
        summary = c.get(f"/batches/{res.json()['id']}").json()
    assert summary["status"] == "failed" and summary["job"]["status"] == "failed"
    assert summary["job"]["attempts"] == 3
    assert "originals/" in summary["job"]["error"]


def test_reviews_form_a_chain(client: TestClient) -> None:
    batch = client.post("/batches", files=_files(("a.jpg", jpeg(50)))).json()
    event = client.get("/events", params={"batch_id": batch["id"]}).json()["events"][0]
    suggested = event["decision"]["suggested_label"]
    other = "raccoon" if suggested != "raccoon" else "coyote"

    first = client.post(
        f"/events/{event['id']}/reviews",
        json={"reviewer": "vol-1", "outcome": "corrected", "confirmed_label": other},
    )
    assert first.status_code == 201 and first.json()["previous_review_id"] is None
    assert first.json()["suggested_label"] == suggested  # original suggestion preserved
    second = client.post(
        f"/events/{event['id']}/reviews", json={"reviewer": "vol-2", "outcome": "unresolved"}
    )
    assert second.json()["previous_review_id"] == first.json()["id"]
    bad = client.post(
        f"/events/{event['id']}/reviews",
        json={"reviewer": "vol-3", "outcome": "confirmed", "confirmed_label": other},
    )
    assert bad.status_code == 422 and bad.json()["error"] == "invalid_review"
    history = client.get(f"/events/{event['id']}").json()["reviews"]
    assert [r["outcome"] for r in history] == ["corrected", "unresolved"]


@pytest.mark.parametrize("prior_reviews", [0, 1])
def test_concurrent_reviews_never_fork_the_history(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, prior_reviews: int
) -> None:
    """Two reviews that both read the same history (none yet, or one) before
    either commits: one is recorded, the other gets 409, and the event keeps a
    single chain with one first review."""
    import threading

    from wildinbox.api import app as app_module

    app = create_app(settings)
    with TestClient(app) as c:
        batch = c.post("/batches", files=_files(("a.jpg", jpeg(51)))).json()
        event = c.get("/events", params={"batch_id": batch["id"]}).json()["events"][0]
        url = f"/events/{event['id']}/reviews"
        for _ in range(prior_reviews):
            assert (
                c.post(url, json={"reviewer": "vol-0", "outcome": "unresolved"}).status_code == 201
            )

    both_read = threading.Barrier(2, timeout=10)
    real_contract = app_module.ReviewContract

    def contract_after_both_read(**kw: Any) -> Any:
        both_read.wait()  # each request has read the chain's tail; neither has committed
        return real_contract(**kw)

    monkeypatch.setattr(app_module, "ReviewContract", contract_after_both_read)
    results: dict[str, Any] = {}

    def submit(who: str) -> None:
        with TestClient(app) as c:
            results[who] = c.post(url, json={"reviewer": who, "outcome": "unresolved"})

    threads = [threading.Thread(target=submit, args=(w,)) for w in ("vol-a", "vol-b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    codes = sorted(r.status_code for r in results.values())
    assert codes == [201, 409]
    (lost,) = [r for r in results.values() if r.status_code == 409]
    assert lost.json()["error"] == "review_conflict"

    monkeypatch.setattr(app_module, "ReviewContract", real_contract)
    with TestClient(app) as c:
        history = c.get(f"/events/{event['id']}").json()["reviews"]
    assert len(history) == prior_reviews + 1
    assert sum(r["previous_review_id"] is None for r in history) == 1
    by_previous = {r["previous_review_id"] for r in history}
    assert len(by_previous) == len(history)  # a single chain: nothing superseded twice


def test_every_consumer_reads_the_chains_last_review_not_the_newest_timestamp(
    settings: Settings,
) -> None:
    """Request A starts its transaction, B records a review and commits, then A
    reads B's review and supersedes it. A's `created_at` (its transaction's
    start) is earlier than B's, but A is the chain's last review, and the API,
    export, monitoring, and snapshot input must all say so."""
    import threading

    from sqlalchemy import event as orm_event
    from sqlalchemy.orm import ORMExecuteState
    from sqlalchemy.orm import Session as OrmSession

    from wildinbox.monitoring.metrics import event_views
    from wildinbox.storage.models import Review

    app = create_app(settings)
    with TestClient(app) as c:
        batch = c.post("/batches", files=_files(("a.jpg", jpeg(54)))).json()
        event = c.get("/events", params={"batch_id": batch["id"]}).json()["events"][0]
    suggested = event["decision"]["suggested_label"]
    b_label, a_label = [x for x in ("empty", "cat", "coyote") if x != suggested][:2]
    url = f"/events/{event['id']}/reviews"

    armed, a_paused, b_committed = threading.Event(), threading.Event(), threading.Event()

    def pause_first_chain_read(state: ORMExecuteState) -> None:
        # A's transaction is open (it has read the event) but its chain is not.
        loads_reviews = any(m.class_ is Review for m in state.all_mappers)
        if armed.is_set() and state.is_relationship_load and loads_reviews:
            armed.clear()
            a_paused.set()
            assert b_committed.wait(10)

    orm_event.listen(OrmSession, "do_orm_execute", pause_first_chain_read)
    results: dict[str, Any] = {}

    def review(who: str, label: str) -> None:
        with TestClient(app) as c:
            results[who] = c.post(
                url, json={"reviewer": who, "outcome": "corrected", "confirmed_label": label}
            )

    try:
        armed.set()
        a = threading.Thread(target=review, args=("A", a_label))
        a.start()
        assert a_paused.wait(10)
        review("B", b_label)
        b_committed.set()
        a.join()
    finally:
        orm_event.remove(OrmSession, "do_orm_execute", pause_first_chain_read)

    assert results["B"].status_code == 201 and results["A"].status_code == 201
    first, last = results["B"].json(), results["A"].json()
    assert last["previous_review_id"] == first["id"]
    assert last["created_at"] < first["created_at"]  # the timestamps disagree with the chain

    with TestClient(app) as c:
        listed = c.get("/events", params={"batch_id": batch["id"]}).json()["events"][0]
        detail = c.get(f"/events/{event['id']}").json()
        exported = c.get(f"/batches/{batch['id']}/export", params={"format": "json"}).json()
    assert listed["latest_review"]["id"] == last["id"]
    assert listed["latest_review"]["confirmed_label"] == a_label
    assert [r["id"] for r in detail["reviews"]] == [first["id"], last["id"]]
    (row,) = exported["observations"]
    assert (row["observation"], row["review_id"]) == (a_label, last["id"])
    with session_factory(settings.database_url)() as s:
        (view,) = event_views(s)
    assert view.reviewed_label == a_label and view.reviewer == "A"


def test_a_broken_review_chain_is_an_error_not_a_guess() -> None:
    from wildinbox.storage.models import Review, review_chain

    first = Review(id=uuid.uuid4(), previous_review_id=None)
    stray = Review(id=uuid.uuid4(), previous_review_id=uuid.uuid4())
    assert review_chain([]) == []
    with pytest.raises(ValueError, match="1 of 2 reviews"):
        review_chain([first, stray])


def test_the_database_allows_one_first_review_per_event(settings: Settings) -> None:
    from sqlalchemy.exc import IntegrityError

    from wildinbox.storage.models import Review

    with TestClient(create_app(settings)) as c:
        batch = c.post("/batches", files=_files(("a.jpg", jpeg(52)))).json()
        event = c.get("/events", params={"batch_id": batch["id"]}).json()["events"][0]
    factory = session_factory(settings.database_url)
    for reviewer in ("first", "second"):
        with factory() as s:
            s.add(Review(event_id=uuid.UUID(event["id"]), reviewer=reviewer, outcome="unresolved"))
            if reviewer == "first":
                s.commit()
            else:
                with pytest.raises(IntegrityError, match="uq_reviews_one_first_review_per_event"):
                    s.commit()


def test_the_migration_refuses_histories_that_are_already_forked(settings: Settings) -> None:
    with TestClient(create_app(settings)) as c:
        batch = c.post("/batches", files=_files(("a.jpg", jpeg(53)))).json()
        event_id = c.get("/events", params={"batch_id": batch["id"]}).json()["events"][0]["id"]
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", settings.database_url)
    engine = create_engine(settings.database_url)
    try:
        command.downgrade(cfg, "4803c5c84b95")
        with engine.begin() as conn:
            for who in ("a", "b"):  # a fork made before the index existed
                conn.execute(
                    text(
                        "INSERT INTO reviews (id, event_id, reviewer, outcome, created_at) "
                        "VALUES (:id, :e, :who, 'unresolved', now())"
                    ),
                    {"id": uuid.uuid4(), "e": event_id, "who": who},
                )
        with pytest.raises(RuntimeError, match="more than one first review"):
            command.upgrade(cfg, "head")
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM reviews WHERE reviewer = 'b'"))
    finally:
        command.upgrade(cfg, "head")


def test_status_page_labels_test_output_and_escapes_input(client: TestClient) -> None:
    batch = client.post(
        "/batches", files=_files(("<script>x</script>.gif", b"GIF89a"), ("a.jpg", jpeg(60)))
    ).json()
    page = client.get(f"/batches/{batch['id']}/view").text
    assert "TEST PREDICTOR" in page and "Suggested label (TEST)" in page
    assert "<script>x</script>" not in page and "&lt;script&gt;" in page
    assert "unsupported_type" in page
    assert "Upload a batch" in client.get("/").text
    assert client.get(f"/batches/{uuid.uuid4()}").status_code == 404


def test_batch_list_time_window_and_thumbnails(client: TestClient, settings: Settings) -> None:
    res = client.post(
        "/batches",
        files=_files(*_small_batch()),
        data={"metadata": json.dumps({"camera_id": "north-trail"})},
    )
    batch = res.json()
    listed = client.get("/batches").json()["batches"]
    assert listed[0]["id"] == batch["id"] and listed[0]["events"] == 2

    night = client.get(
        "/events",
        params={"start_after": "2024-05-01T21:00:00", "start_before": "2024-05-01T22:00:00"},
    ).json()
    assert night["total"] == 1 and len(night["events"][0]["image_ids"]) == 2
    before = client.get("/events", params={"start_before": "2024-05-01T00:00:00"}).json()
    assert before["total"] == 0

    image_id = night["events"][0]["image_ids"][0]
    thumb = client.get(f"/images/{image_id}/thumbnail", params={"size": 64})
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/jpeg"
    from PIL import Image as PILImage

    img = PILImage.open(io.BytesIO(thumb.content))
    assert max(img.size) <= 64
    stored = list((settings.local_store_dir / "thumbnails").rglob("*.jpg"))
    assert len(stored) == 1  # generated once, kept in object storage
    assert client.get(f"/images/{image_id}/thumbnail", params={"size": 64}).content == thumb.content
    assert client.get(f"/images/{uuid.uuid4()}/thumbnail").status_code == 404


def test_thumbnail_shrinks_and_applies_exif_orientation() -> None:
    from PIL import Image as PILImage

    from wildinbox.api.app import _thumbnail

    buf = io.BytesIO()
    exif = PILImage.Exif()
    exif[0x0112] = 6  # displayed rotated 90 degrees
    PILImage.new("RGB", (400, 300)).save(buf, "JPEG", exif=exif)
    out = PILImage.open(io.BytesIO(_thumbnail(buf.getvalue(), 100)))
    assert out.size == (75, 100)  # portrait after rotation, long side 100


def test_batch_history_pages_reach_every_batch(client: TestClient) -> None:
    made = [
        client.post("/batches", files=_files((f"{n}.jpg", jpeg(70 + n)))).json() for n in range(5)
    ]
    first = client.get("/batches", params={"limit": 2}).json()
    assert first["total"] == 5 and first["next_offset"] == 2
    assert first["batches"][0]["id"] == made[-1]["id"]  # newest first
    seen, offset = [], 0
    while offset is not None:
        page = client.get("/batches", params={"limit": 2, "offset": offset}).json()
        seen += [b["id"] for b in page["batches"]]
        offset = page["next_offset"]
    assert seen == [b["id"] for b in reversed(made)]  # the oldest is reachable
    assert all(b["images"] == 1 and b["events"] == 1 for b in first["batches"])


def test_camera_catalog_covers_every_event(client: TestClient) -> None:
    meta = {"files": {"a.jpg": {"camera_id": "north"}, "b.jpg": {"camera_id": "south"}}}
    batch = client.post(
        "/batches",
        files=_files(("a.jpg", jpeg(80)), ("b.jpg", jpeg(81)), ("c.jpg", jpeg(82))),
        data={"metadata": json.dumps(meta)},
    ).json()
    client.post(
        "/batches", files=_files(("d.jpg", jpeg(83))), data={"metadata": '{"camera_id": "west"}'}
    )
    listed = client.get("/cameras", params={"batch_id": batch["id"]}).json()["cameras"]
    assert listed == [
        {"camera_id": "north", "events": 1},
        {"camera_id": "south", "events": 1},
        {"camera_id": None, "events": 1},
    ]
    everywhere = client.get("/cameras").json()["cameras"]
    assert [c["camera_id"] for c in everywhere] == ["north", "south", "west", None]
    assert client.get("/batches").json()["batches"][1]["cameras"] == ["north", "south"]


def test_animal_filter_matches_the_interfaces_visitor_rule(
    client: TestClient, settings: Settings
) -> None:
    """`animal` is evaluated in SQL over the whole result, so a night with many
    empty events cannot hide an animal. It must agree with `is_visitor`."""
    from datetime import timedelta

    from wildinbox.storage.models import Decision as DecisionRow
    from wildinbox.ui.logic import is_visitor

    # suggestion, reviews (outcome, label) in order, a newer decision's suggestion
    cases: dict[str, tuple[str | None, list[tuple[str, str | None]], str | None]] = {
        "suggested-animal": ("raccoon", [], None),
        "suggested-empty": ("empty", [], None),
        "no-suggestion": (None, [], None),
        "corrected-to-animal": ("empty", [("corrected", "raccoon")], None),
        "corrected-to-empty": ("raccoon", [("corrected", "empty")], None),
        "unresolved": ("empty", [("unresolved", None)], None),
        "superseded": ("raccoon", [("confirmed", "raccoon"), ("corrected", "empty")], None),
        "newer-decision": ("raccoon", [], "empty"),
    }
    names = list(cases)
    meta = {"files": {f"{n}.jpg": {"sequence_id": n} for n in names}}
    batch = client.post(
        "/batches",
        files=_files(*[(f"{n}.jpg", jpeg(90 + i)) for i, n in enumerate(names)]),
        data={"metadata": json.dumps(meta)},
    ).json()
    events = client.get("/events", params={"batch_id": batch["id"]}).json()["events"]
    images = {
        i["id"]: i["filename"]
        for i in client.get(f"/batches/{batch['id']}/images").json()["images"]
    }
    by_case = {images[e["image_ids"][0]][:-4]: e["id"] for e in events}
    with session_factory(settings.database_url)() as s:
        for name, (suggested, _, newer) in cases.items():
            d = s.scalar(
                select(DecisionRow).where(DecisionRow.event_id == uuid.UUID(by_case[name]))
            )
            assert d is not None
            d.suggested_label = suggested
            if newer:
                s.add(
                    DecisionRow(
                        event_id=d.event_id,
                        model_release_id=d.model_release_id,
                        policy_version="later",
                        disposition="needs_review",
                        suggested_label=newer,
                        reasons=[],
                        created_at=d.created_at + timedelta(hours=1),
                    )
                )
        s.commit()
    for name, (_, reviews, _) in cases.items():
        for outcome, label in reviews:
            res = client.post(
                f"/events/{by_case[name]}/reviews",
                json={"reviewer": "ann", "outcome": outcome, "confirmed_label": label},
            )
            assert res.status_code == 201, res.text

    rows = client.get("/events", params={"batch_id": batch["id"]}).json()["events"]
    expected = {e["id"] for e in rows if is_visitor(e)}
    animal = client.get("/events", params={"batch_id": batch["id"], "animal": True}).json()
    other = client.get("/events", params={"batch_id": batch["id"], "animal": False}).json()
    assert {e["id"] for e in animal["events"]} == expected
    assert {e["id"] for e in other["events"]} == {e["id"] for e in rows} - expected
    assert {n for n, eid in by_case.items() if eid in expected} == {
        "suggested-animal",
        "corrected-to-animal",
        "unresolved",
    }


def test_event_lists_run_a_bounded_number_of_queries(settings: Settings) -> None:
    """Listing events must not add queries per event: a page of 100 costs about
    what a page of 10 does, on every endpoint that lists events."""
    from sqlalchemy import event as sa_event
    from sqlalchemy.engine import Engine

    cfg = settings.model_copy(update={"max_files_per_batch": 100, "max_batch_bytes": 2_000_000})
    names = [f"{i}.jpg" for i in range(100)]
    meta = {
        "camera_id": "north",
        "files": {n: {"sequence_id": n, "captured_at": "2024-05-01T21:00:00"} for n in names},
    }
    queries: list[str] = []

    def count(*args: Any) -> None:
        if args[2].lstrip().upper().startswith("SELECT"):
            queries.append(args[2])

    def selects(c: TestClient, path: str, **params: Any) -> int:
        queries.clear()
        assert c.get(path, params=params).status_code == 200
        return len(queries)

    with TestClient(create_app(cfg)) as c:
        batch = c.post(
            "/batches",
            files=_files(*[(n, jpeg(5000 + i)) for i, n in enumerate(names)]),
            data={"metadata": json.dumps(meta)},
        ).json()["id"]
        for eid in [e["id"] for e in c.get("/events", params={"limit": 3}).json()["events"]]:
            c.post(f"/events/{eid}/reviews", json={"reviewer": "ann", "outcome": "unresolved"})
        sa_event.listen(Engine, "before_cursor_execute", count)
        try:
            ten = selects(c, "/events", batch_id=batch, limit=10)
            hundred = selects(c, "/events", batch_id=batch, limit=100)
            animals = selects(c, "/events", batch_id=batch, limit=100, animal=True)
            view = selects(c, f"/batches/{batch}/view")
            export = selects(c, f"/batches/{batch}/export")
        finally:
            sa_event.remove(Engine, "before_cursor_execute", count)
    counts = {"10": ten, "100": hundred, "animal": animals, "view": view, "export": export}
    assert hundred == ten and animals <= ten, counts
    assert max(view, export) <= 15, counts


def test_monitoring_queries_do_not_grow_with_batches(client: TestClient) -> None:
    from sqlalchemy import event as sa_event
    from sqlalchemy.engine import Engine

    queries: list[str] = []

    def count(*args: Any) -> None:
        if args[2].lstrip().upper().startswith("SELECT"):
            queries.append(args[2])

    def selects() -> int:
        queries.clear()
        sa_event.listen(Engine, "before_cursor_execute", count)
        try:
            assert client.get("/monitoring").status_code == 200
        finally:
            sa_event.remove(Engine, "before_cursor_execute", count)
        return len(queries)

    for n in range(2):
        client.post("/batches", files=_files((f"{n}.jpg", jpeg(6000 + n))))
    two = selects()
    for n in range(2, 6):
        client.post("/batches", files=_files((f"{n}.jpg", jpeg(6000 + n))))
    six = selects()
    assert six == two, (two, six)
    assert client.get("/monitoring").json()["operations"]["batches"][0]["images"] == 1


@pytest.mark.parametrize("gap", [-1, 3601, "NaN", "soon"])
def test_grouping_interval_is_validated(client: TestClient, gap: Any) -> None:
    raw = '{"gap_seconds": ' + (gap if gap == "NaN" else json.dumps(gap)) + "}"
    res = client.post("/batches", files=_files(("a.jpg", jpeg(90))), data={"metadata": raw})
    assert res.status_code == 422 and res.json()["error"] == "invalid_metadata", res.text


def test_grouping_interval_joins_the_fingerprint_only_when_given(client: TestClient) -> None:
    import hashlib

    from wildinbox.api.uploads import BatchMetadata, UploadedFile, request_fingerprint

    files = [UploadedFile(0, "a.jpg", 3, None, sha256="ab")]
    legacy = hashlib.sha256(
        json.dumps(
            {
                "files": [["a.jpg", "ab", 3, None]],
                "metadata": {"camera_id": "trail", "files": {}},
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    assert request_fingerprint(files, BatchMetadata(camera_id="trail")) == legacy
    assert request_fingerprint(files, BatchMetadata(camera_id="trail", gap_seconds=5)) != legacy

    def post(meta: dict[str, Any]) -> str:
        res = client.post(
            "/batches", files=_files(("a.jpg", jpeg(91))), data={"metadata": json.dumps(meta)}
        )
        assert res.status_code in (200, 202), res.text
        return str(res.json()["id"])

    assert post({"gap_seconds": 0}) == post({"gap_seconds": 0})  # a resubmission
    assert post({"gap_seconds": 0}) != post({"gap_seconds": 30})
    assert client.get("/batches", params={"limit": 5}).json()["total"] == 2


def test_migration_records_the_grouping_of_earlier_batches(settings: Settings) -> None:
    with TestClient(create_app(settings)) as c:
        old = c.post("/batches", files=_files(("a.jpg", jpeg(92)))).json()["id"]
        chosen = c.post(
            "/batches", files=_files(("b.jpg", jpeg(93))), data={"metadata": '{"gap_seconds": 9}'}
        ).json()["id"]
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", settings.database_url)
    engine = create_engine(settings.database_url)

    def grouping(batch_id: str) -> Any:
        with engine.begin() as conn:
            return conn.execute(
                text("SELECT manifest -> 'grouping' FROM batches WHERE id = :id"), {"id": batch_id}
            ).scalar()

    try:
        with engine.begin() as conn:  # a batch from before the interval was recorded
            conn.execute(
                text("UPDATE batches SET manifest = manifest - 'grouping' WHERE id = :id"),
                {"id": old},
            )
        command.downgrade(cfg, "5d8e2b1f9a47")
        assert grouping(old) is None and grouping(chosen)["gap_seconds"] == 9
        command.upgrade(cfg, "head")
        assert grouping(old) == {
            "gap_seconds": 5.0,
            "gap_source": "legacy_default",
            "time_gap_rule": "time_gap/v1(gap_s=5)",
        }
        assert grouping(chosen)["gap_seconds"] == 9  # untouched
        command.downgrade(cfg, "5d8e2b1f9a47")
        assert grouping(old) is None and grouping(chosen)["gap_seconds"] == 9
    finally:
        command.upgrade(cfg, "head")


def _orm_event_views(s: Any) -> list[Any]:
    """The earlier ORM implementation of `event_views`, kept as the reference."""
    from sqlalchemy.orm import selectinload

    from wildinbox.monitoring.metrics import EventView
    from wildinbox.storage.models import Batch as BatchRow
    from wildinbox.storage.models import Event as EventRow

    out = []
    for e, created in s.execute(
        select(EventRow, BatchRow.created_at)
        .join(BatchRow)
        .options(
            selectinload(EventRow.decisions),
            selectinload(EventRow.reviews),
            selectinload(EventRow.images),
        )
    ).all():
        d = max(e.decisions, key=lambda d: d.created_at, default=None)
        r = e.current_review
        q = [i.quality for i in e.images if i.quality and "night" in i.quality]
        out.append(
            EventView(
                camera=e.camera_id or "unknown",
                start_at=e.start_at,
                batch_id=str(e.batch_id),
                batch_created_at=created,
                decided_at=d.created_at if d else None,
                audit_selected=bool(d.audit_selected) if d else False,
                reviewed_label=r.confirmed_label if r else None,
                release=d.model_release_id if d else None,
                disposition=d.disposition if d else None,
                label=d.suggested_label if d else None,
                confidence=d.confidence if d else None,
                reasons=list(d.reasons) if d else [],
                review_outcome=r.outcome if r else None,
                reviewer=r.reviewer if r else None,
                night=[bool(x["night"]) for x in q],
                blur=[float(x["blur"]) for x in q],
            )
        )
    return out


def test_monitoring_event_views_match_the_orm_reading(
    client: TestClient, settings: Settings
) -> None:
    from datetime import timedelta

    from wildinbox.monitoring.metrics import event_views
    from wildinbox.storage.models import Decision as DecisionRow
    from wildinbox.storage.models import Image as ImageRow

    meta = {
        "camera_id": "north",
        "files": {
            "a.jpg": {"sequence_id": "s1"},
            "b.jpg": {"sequence_id": "s1"},
            "c.jpg": {"sequence_id": "s2", "camera_id": None},
        },
    }
    batch = client.post(
        "/batches",
        files=_files(("a.jpg", jpeg(95)), ("b.jpg", jpeg(96)), ("c.jpg", jpeg(97))),
        data={"metadata": json.dumps(meta)},
    ).json()
    events = client.get("/events", params={"batch_id": batch["id"]}).json()["events"]
    first = events[0]["id"]
    for outcome, label in (("corrected", "raccoon"), ("unresolved", None)):
        client.post(
            f"/events/{first}/reviews",
            json={"reviewer": "ann", "outcome": outcome, "confirmed_label": label},
        )
    with session_factory(settings.database_url)() as s:
        d = s.scalar(select(DecisionRow).where(DecisionRow.event_id == uuid.UUID(first)))
        assert d is not None
        s.add(  # a newer decision under another policy wins
            DecisionRow(
                event_id=d.event_id,
                model_release_id=d.model_release_id,
                policy_version="later",
                disposition="species_identified",
                suggested_label="coyote",
                confidence=0.9,
                reasons=[],
                audit_selected=True,
                created_at=d.created_at + timedelta(minutes=1),
            )
        )
        img = s.scalar(select(ImageRow).where(ImageRow.original_filename == "b.jpg"))
        assert img is not None
        img.quality = None  # a frame without quality is left out of night and blur
        s.commit()
        new, reference = event_views(s), _orm_event_views(s)
    key = lambda v: (v.batch_id, v.start_at is None, v.camera, v.label or "")  # noqa: E731
    assert sorted(new, key=key) == sorted(reference, key=key)
    changed = next(v for v in new if v.label == "coyote")
    assert changed.review_outcome == "unresolved" and changed.audit_selected
    assert len(changed.night) == 1 and len(changed.blur) == 1


def test_monitoring_analyses_only_its_history_window(
    client: TestClient, settings: Settings
) -> None:
    from datetime import timedelta

    from sqlalchemy import update as sa_update

    from wildinbox.storage.models import Event as EventRow

    for n in range(3):
        client.post("/batches", files=_files((f"{n}.jpg", jpeg(98 + n))))
    with session_factory(settings.database_url)() as s:
        old = s.scalars(select(EventRow.id).limit(2)).all()
        s.execute(
            sa_update(EventRow)
            .where(EventRow.id.in_(old))
            .values(created_at=func.now() - timedelta(days=200))
        )
        s.commit()
    history = client.get("/monitoring").json()["history"]
    assert history["days"] == 90
    assert (history["events_analysed"], history["older_events_excluded"]) == (1, 2)
