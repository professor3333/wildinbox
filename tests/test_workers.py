"""Stage 9 acceptance: reliable asynchronous inference.

A worker killed mid-batch resumes without lost inputs, duplicate predictions,
or prematurely finalized events; a corrupt image produces an explicit error
while unrelated events continue; a job always runs the release it was pinned to.
"""

from __future__ import annotations

import hashlib
import io
import json
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image as PILImage
from sqlalchemy import func, select, update
from sqlalchemy.exc import DBAPIError

from wildinbox.api.app import create_app
from wildinbox.class_map import ClassMap
from wildinbox.config import load_config
from wildinbox.inference import serving
from wildinbox.inference.architecture import build_model
from wildinbox.inference.releases import ReleaseError, activate, build_release, register_release
from wildinbox.inference.serving import PlumbingScorer, ReleaseScorer
from wildinbox.policy.conservative import POLICY_NAME, PolicyConfig
from wildinbox.preprocessing import build_eval_transform, load_image
from wildinbox.settings import Settings
from wildinbox.storage.db import session_factory
from wildinbox.storage.models import Decision, Event, Image, Job, ModelRelease, Prediction
from wildinbox.storage.objects import LocalStore
from wildinbox.workers import process
from wildinbox.workers.process import LeaseLost, claim, process_batch, recover_stale, renew

from .conftest import REPO_ROOT
from .test_app import NoopDispatcher, _files, database_url, settings  # noqa: F401
from .test_uploads import jpeg


class Killed(BaseException):
    """Stands in for SIGKILL: nothing in the worker catches it."""


def _count(cfg: Settings, model: Any, *where: Any) -> int:
    with session_factory(cfg.database_url)() as s:
        return int(s.scalar(select(func.count()).select_from(model).where(*where)) or 0)


def _upload(cfg: Settings, n: int, extra: tuple[tuple[str, bytes], ...] = ()) -> dict[str, Any]:
    """Upload a batch without processing it; return the batch summary."""
    items = tuple(
        (f"cam-{i}.jpg", jpeg(100 + i, exif_time=f"2024:05:01 21:00:{i:02d}")) for i in range(n)
    )
    with TestClient(create_app(cfg, dispatcher=NoopDispatcher())) as c:
        res = c.post(
            "/batches",
            files=_files(*items, *extra),
            data={"metadata": json.dumps({"camera_id": "trail"})},
        )
        assert res.status_code == 202, res.text
        return res.json()


class CrashingScorer:
    """The test scorer, but the process 'dies' on chunk `die_on`."""

    def __init__(self, inner: PlumbingScorer, die_on: int | None) -> None:
        self.inner, self.die_on, self.calls = inner, die_on, 0
        self.release_id, self.class_names = inner.release_id, inner.class_names

    def score(self, images: Any, sha256s: Any) -> list[dict[str, float]]:
        self.calls += 1
        if self.die_on is not None and self.calls == self.die_on:
            raise Killed
        return self.inner.score(images, sha256s)


@pytest.fixture
def worker_settings(settings: Settings) -> Settings:  # noqa: F811
    return settings.model_copy(update={"inference_chunk": 2, "lease_seconds": 60})


def _patch_scorer(monkeypatch: pytest.MonkeyPatch, die_on: int | None) -> None:
    def fake(release: ModelRelease, store: Any, device: str = "cpu") -> CrashingScorer:
        return CrashingScorer(PlumbingScorer(release), die_on)

    monkeypatch.setattr(process, "scorer_for", fake)


def _expire_lease(cfg: Settings, job_id: uuid.UUID) -> None:
    with session_factory(cfg.database_url)() as s:
        s.execute(
            update(Job)
            .where(Job.id == job_id)
            .values(heartbeat_at=datetime.now(UTC) - timedelta(hours=1))
        )
        s.commit()


# ------------------------------------------------------------ acceptance gate


def test_killed_worker_resumes_without_loss_duplicates_or_early_events(
    worker_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = _upload(worker_settings, 7)
    job_id = uuid.UUID(batch["job"]["id"])
    factory = session_factory(worker_settings.database_url)
    store = LocalStore(worker_settings.local_store_dir)

    _patch_scorer(monkeypatch, die_on=3)  # dies on the third chunk of two
    with pytest.raises(Killed):
        process_batch(factory, store, job_id, worker_settings, "worker-a")

    # Two chunks were committed; nothing was finalized.
    assert _count(worker_settings, Prediction) == 4
    assert _count(worker_settings, Event) == 0 and _count(worker_settings, Decision) == 0
    with factory() as s:
        job = s.get(Job, job_id)
        assert job is not None and job.status == "running" and job.worker_id == "worker-a"
        assert job.batch.status == "processing"

    # While the lease is fresh, nobody else can take the job.
    _patch_scorer(monkeypatch, die_on=None)
    assert process_batch(factory, store, job_id, worker_settings, "worker-b") == "skipped"

    # The dead worker's lease expires; recovery requeues and dispatches it.
    _expire_lease(worker_settings, job_id)
    dispatcher = NoopDispatcher()
    assert recover_stale(factory, dispatcher, worker_settings) == [job_id]
    assert dispatcher.enqueued == [job_id]
    assert process_batch(factory, store, job_id, worker_settings, "worker-b") == "succeeded"

    assert _count(worker_settings, Prediction) == 7  # nothing lost
    with factory() as s:
        ids = s.scalars(select(Prediction.image_id)).all()
        assert len(ids) == len(set(ids)) == 7  # nothing duplicated
        events = s.scalars(select(Event)).all()
        # One burst: its frames were scored across chunks, before and after the crash.
        assert len(events) == 1 and len(events[0].images) == 7
        assert all(len(e.decisions) == 1 for e in events)
        job = s.get(Job, job_id)
        assert job is not None and job.status == "succeeded" and job.attempts == 2
        assert job.batch.status == "completed" and "stopped responding" not in (job.error or "")

    # Running it again changes nothing.
    assert process_batch(factory, store, job_id, worker_settings) == "skipped"
    assert _count(worker_settings, Prediction) == 7 and _count(worker_settings, Decision) == len(
        events
    )


def test_a_worker_that_lost_its_lease_cannot_write(worker_settings: Settings) -> None:
    batch = _upload(worker_settings, 2)
    job_id = uuid.UUID(batch["job"]["id"])
    factory = session_factory(worker_settings.database_url)
    token_a = claim(factory, job_id, "worker-a", worker_settings)
    assert token_a is not None
    _expire_lease(worker_settings, job_id)
    recover_stale(factory, NoopDispatcher(), worker_settings)
    token_b = claim(factory, job_id, "worker-b", worker_settings)
    assert token_b is not None and token_b != token_a
    with factory() as s, pytest.raises(LeaseLost):
        renew(s, job_id, token_a)  # the zombie's next commit is refused
    with factory() as s:
        renew(s, job_id, token_b)


def test_corrupt_image_fails_explicitly_while_other_events_complete(
    worker_settings: Settings,
) -> None:
    # Passes the upload sniff (JPEG magic) but cannot be decoded.
    broken = b"\xff\xd8\xff\xe0" + b"not really a jpeg" * 10
    batch = _upload(worker_settings, 3, extra=(("broken.jpg", broken),))
    factory = session_factory(worker_settings.database_url)
    store = LocalStore(worker_settings.local_store_dir)
    process_batch(factory, store, uuid.UUID(batch["job"]["id"]), worker_settings)
    with TestClient(create_app(worker_settings, dispatcher=NoopDispatcher())) as c:
        summary = c.get(f"/batches/{batch['id']}").json()
    assert summary["status"] == "completed_with_errors"
    (failure,) = summary["failures"]
    assert failure["filename"] == "broken.jpg" and failure["error"].startswith("unreadable image")
    # The three good frames (a second apart) still form one decided event.
    assert summary["counts"]["events"] == 1
    assert _count(worker_settings, Decision) == 1


def test_inference_failure_on_one_image_becomes_a_failed_frame(
    worker_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = _upload(worker_settings, 3)
    factory = session_factory(worker_settings.database_url)
    with factory() as s:
        bad = s.scalars(select(Image).order_by(Image.position)).first()
        assert bad is not None
        bad_sha = bad.sha256

    class PickyScorer(PlumbingScorer):
        def score(self, images: Any, sha256s: Any) -> list[dict[str, float]]:
            if bad_sha in sha256s:
                raise RuntimeError("model rejected this input")
            return super().score(images, sha256s)

    monkeypatch.setattr(process, "scorer_for", lambda r, st, d="cpu": PickyScorer(r))
    status = process_batch(
        factory,
        LocalStore(worker_settings.local_store_dir),
        uuid.UUID(batch["job"]["id"]),
        worker_settings,
    )
    assert status == "succeeded"
    with factory() as s:
        (event,) = s.scalars(select(Event)).all()
        (decision,) = event.decisions
        assert decision.disposition == "needs_review"
        assert decision.reasons[0] == "processing_failure"
        img = s.scalars(select(Image).where(Image.sha256 == bad_sha)).one()
        assert img.processing_error and "model rejected" in img.processing_error
    assert _count(worker_settings, Prediction) == 2


# ------------------------------------------------------------------ releases


def _fake_model_dir(tmp_path: Path, seed: int = 0) -> tuple[Path, Path]:
    cfg = load_config(REPO_ROOT / "configs/example.yaml")
    d = tmp_path / f"model-{seed}"
    d.mkdir(parents=True)
    torch.manual_seed(seed)
    buf = io.BytesIO()
    torch.save(build_model(len(cfg.classes), None).state_dict(), buf)
    (d / "model.pt").write_bytes(buf.getvalue())
    sha = hashlib.sha256(buf.getvalue()).hexdigest()
    (d / "meta.json").write_text(
        json.dumps(
            {
                "name": "tiny-test-model",
                "classes": cfg.classes,
                "class_map_fingerprint": ClassMap(cfg.classes).fingerprint(),
                "preprocessing": cfg.preprocessing.model_dump(mode="json"),
                "preprocessing_version": cfg.preprocessing.fingerprint(),
            }
        )
    )
    released = PolicyConfig(0.65, None, False, False, ())
    policy = {
        "policy": POLICY_NAME,
        "model": "tiny-test-model",
        "classes": cfg.classes,
        "calibration": {
            "method": "temperature",
            "temperature": 2.0,
            "weights_digest": sha[:12],
            "version": "cal1",
        },
        "unfamiliar": None,
        "released": {
            "empty_threshold": 0.65,
            "species_threshold": None,
            "auto_filter_enabled": False,
            "auto_accept_enabled": False,
            "accept_species": [],
            "policy_version": f"{POLICY_NAME}+{released.fingerprint()}",
        },
        "artifact_version": f"art{seed}",
    }
    p = d / "policy.json"
    p.write_text(json.dumps(policy))
    return d, p


@pytest.fixture
def real_release(worker_settings: Settings, tmp_path: Path) -> Iterator[ModelRelease]:
    serving._LOADED.clear()
    model_dir, policy = _fake_model_dir(tmp_path)
    store = LocalStore(worker_settings.local_store_dir)
    with TestClient(create_app(worker_settings, dispatcher=NoopDispatcher())):
        pass  # runs migrations-dependent startup (test release)
    with session_factory(worker_settings.database_url)() as s:
        release, created = register_release(s, store, model_dir, policy, "test")
        assert created
        again, created_again = register_release(s, store, model_dir, policy, "test")
        assert again.id == release.id and not created_again
        activate(s, release.id)
        s.commit()
        yield release
    serving._LOADED.clear()


def test_real_release_runs_end_to_end_with_versions_reported(
    worker_settings: Settings, real_release: ModelRelease
) -> None:
    batch = _upload(worker_settings, 3)
    assert batch["job"]["model_release_id"] == real_release.id
    status = process_batch(
        session_factory(worker_settings.database_url),
        LocalStore(worker_settings.local_store_dir),
        uuid.UUID(batch["job"]["id"]),
        worker_settings,
    )
    assert status == "succeeded"
    with TestClient(create_app(worker_settings, dispatcher=NoopDispatcher())) as c:
        version = c.get("/version").json()["active_release"]
        assert version["id"] == real_release.id
        assert version["weights_sha256"] == real_release.weights_sha256
        assert version["policy_version"].startswith("conservative/v1+")
        (event,) = c.get("/events", params={"batch_id": batch["id"]}).json()["events"]
        assert event["decision"]["policy_version"] == real_release.policy_version
        assert event["decision"]["reasons"][-1] == "automation_disabled"
        detail = c.get(f"/events/{event['id']}").json()
        for img in detail["images"]:
            assert img["frame_status"] == "completed"
            cal = img["prediction"]["calibrated_probabilities"]
            assert abs(sum(cal.values()) - 1) < 1e-6
            raw = img["prediction"]["class_probabilities"]
            # T = 2 softens: the calibrated top score is never higher than the raw one.
            assert max(cal.values()) <= max(raw.values()) + 1e-9


def test_retry_keeps_the_release_it_was_created_with(
    worker_settings: Settings, real_release: ModelRelease, tmp_path: Path
) -> None:
    batch = _upload(worker_settings, 2)
    job_id = uuid.UUID(batch["job"]["id"])
    factory = session_factory(worker_settings.database_url)
    store = LocalStore(worker_settings.local_store_dir)
    other_dir, other_policy = _fake_model_dir(tmp_path, seed=1)
    with factory() as s:
        newer, _ = register_release(s, store, other_dir, other_policy, "newer")
        activate(s, newer.id, "a new default arrives mid-job")
        s.commit()
    assert claim(factory, job_id, "worker-a", worker_settings) is not None
    _expire_lease(worker_settings, job_id)  # the worker died
    recover_stale(factory, NoopDispatcher(), worker_settings)
    assert process_batch(factory, store, job_id, worker_settings) == "succeeded"
    with factory() as s:
        releases = set(s.scalars(select(Prediction.model_release_id)))
        assert releases == {real_release.id}
        job = s.get(Job, job_id)
        assert job is not None and job.model_release_id == real_release.id


def test_releases_are_immutable(
    worker_settings: Settings, real_release: ModelRelease, tmp_path: Path
) -> None:
    factory = session_factory(worker_settings.database_url)
    with factory() as s, pytest.raises(DBAPIError, match="immutable"):
        s.execute(update(ModelRelease).where(ModelRelease.id == real_release.id).values(notes="x"))
        s.commit()
    # Same id, different content: refused.
    model_dir, policy = _fake_model_dir(tmp_path / "again", seed=0)
    raw = json.loads(policy.read_text())
    raw["released"]["empty_threshold"] = 0.5
    policy.write_text(json.dumps(raw))
    with factory() as s, pytest.raises(ReleaseError):
        register_release(s, LocalStore(worker_settings.local_store_dir), model_dir, policy, None)


def test_release_refuses_calibration_from_other_weights(tmp_path: Path) -> None:
    model_dir, policy = _fake_model_dir(tmp_path)
    raw = json.loads(policy.read_text())
    raw["calibration"]["weights_digest"] = "000000000000"
    policy.write_text(json.dumps(raw))
    with pytest.raises(ReleaseError, match="different weights"):
        build_release(model_dir, policy)


def test_serving_tensors_match_evaluation_for_rotated_photos(
    worker_settings: Settings, real_release: ModelRelease
) -> None:
    """Preprocessing consistency: the worker's input equals the evaluation input,
    including EXIF orientation (a rotated camera photo)."""
    arr = np.random.default_rng(3).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    img = PILImage.fromarray(arr)
    exif = PILImage.Exif()
    exif[0x0112] = 6  # rotate 90 degrees on display
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    data = buf.getvalue()
    scorer = ReleaseScorer(real_release, LocalStore(worker_settings.local_store_dir))
    served = scorer.tensors([data])[0]
    evaluated = build_eval_transform(scorer.preprocessing)(load_image(data))
    assert torch.equal(served, evaluated)
    assert load_image(data).size == (48, 64)  # orientation applied


# ------------------------------------------------------------------ API and export


def test_export_carries_reviews_and_provenance(
    worker_settings: Settings, real_release: ModelRelease
) -> None:
    batch = _upload(worker_settings, 2)
    process_batch(
        session_factory(worker_settings.database_url),
        LocalStore(worker_settings.local_store_dir),
        uuid.UUID(batch["job"]["id"]),
        worker_settings,
    )
    with TestClient(create_app(worker_settings, dispatcher=NoopDispatcher())) as c:
        (event,) = c.get("/events", params={"batch_id": batch["id"]}).json()["events"]
        pending = c.get(f"/batches/{batch['id']}/export", params={"format": "json"}).json()
        (row,) = pending["observations"]
        assert row["label_source"] == "pending_review" and row["observation"] is None
        label = "coyote" if event["decision"]["suggested_label"] != "coyote" else "bobcat"
        res = c.post(
            f"/events/{event['id']}/reviews",
            json={"reviewer": "ranger", "outcome": "corrected", "confirmed_label": label},
        )
        assert res.status_code == 201, res.text
        (row,) = c.get(f"/batches/{batch['id']}/export", params={"format": "json"}).json()[
            "observations"
        ]
        assert (row["observation"], row["label_source"], row["reviewer"]) == (
            label,
            "review",
            "ranger",
        )
        assert row["weights_sha256"] == real_release.weights_sha256
        assert row["policy_version"] == real_release.policy_version
        assert row["calibration_version"] == "cal1" and row["preprocessing_version"]
        csv_text = c.get(f"/batches/{batch['id']}/export").text
        header, line = csv_text.strip().splitlines()
        assert header.split(",")[0] == "event_id" and label in line
        assert c.get(f"/batches/{batch['id']}/export", params={"format": "xml"}).status_code == 400


def test_event_listing_reports_totals_and_filters(worker_settings: Settings) -> None:
    for n in (2, 3):
        batch = _upload(worker_settings, n)
        process_batch(
            session_factory(worker_settings.database_url),
            LocalStore(worker_settings.local_store_dir),
            uuid.UUID(batch["job"]["id"]),
            worker_settings,
        )
    with TestClient(create_app(worker_settings, dispatcher=NoopDispatcher())) as c:
        page = c.get("/events", params={"limit": 1}).json()
        total = page["total"]
        assert total >= 2 and page["next_offset"] == 1 and len(page["events"]) == 1
        last = c.get("/events", params={"limit": 1, "offset": total - 1}).json()
        assert last["next_offset"] is None
        reasons = c.get("/events", params={"reason": "automation_disabled"}).json()
        assert reasons["total"] == total
        assert c.get("/events", params={"reviewed": True}).json()["total"] == 0
        assert c.get("/releases").json()["active_release_id"] == "test-predictor-v0"


def test_worker_names_are_unique_per_process_start() -> None:
    """A restarted container keeps its hostname and often PID 1; RQ refuses a
    worker name a killed worker still holds, so each start adds a random part."""
    import os
    import socket

    host, pid, suffix = process.worker_id().rsplit(":", 2)
    assert (host, pid) == (socket.gethostname(), str(os.getpid()))
    assert len(suffix) == 8 and process.worker_id() == process.worker_id()
