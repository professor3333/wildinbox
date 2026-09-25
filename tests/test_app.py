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
