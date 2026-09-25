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


def _fake_model_dir(
    tmp_path: Path,
    seed: int = 0,
    policy_name: str = POLICY_NAME,
    empty_threshold: float = 0.65,
    auto_filter: bool = False,
) -> tuple[Path, Path]:
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
    released = PolicyConfig(empty_threshold, None, auto_filter, False, ())
    policy = {
        "policy": policy_name,
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
            "empty_threshold": empty_threshold,
            "species_threshold": None,
            "auto_filter_enabled": auto_filter,
            "auto_accept_enabled": False,
            "accept_species": [],
            "policy_version": released.version(policy_name),
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


def test_activation_history_shows_release_and_rollback(
    worker_settings: Settings, real_release: ModelRelease, tmp_path: Path
) -> None:
    factory = session_factory(worker_settings.database_url)
    other_dir, other_policy = _fake_model_dir(tmp_path, seed=2)
    with factory() as s:
        newer, _ = register_release(
            s, LocalStore(worker_settings.local_store_dir), other_dir, other_policy, None
        )
        activate(s, newer.id, "release candidate")
        activate(s, real_release.id, "roll back")
        s.commit()
    with TestClient(create_app(worker_settings, dispatcher=NoopDispatcher())) as c:
        body = c.get("/releases").json()
        assert body["active_release_id"] == real_release.id
        history = [(a["release_id"], a["note"]) for a in body["activations"]]
        assert history[:2] == [(real_release.id, "roll back"), (newer.id, "release candidate")]
        assert c.get("/version").json()["active_release"]["id"] == real_release.id


def test_workers_decide_with_the_policy_their_release_names(
    worker_settings: Settings, tmp_path: Path
) -> None:
    from wildinbox.policy.conservative import POLICY_V2

    serving._LOADED.clear()
    model_dir, policy = _fake_model_dir(tmp_path, seed=4, policy_name=POLICY_V2)
    with TestClient(create_app(worker_settings, dispatcher=NoopDispatcher())):
        pass
    with session_factory(worker_settings.database_url)() as s:
        release, _ = register_release(
            s, LocalStore(worker_settings.local_store_dir), model_dir, policy, None
        )
        activate(s, release.id)
        s.commit()
    batch = _upload(worker_settings, 3)
    process_batch(
        session_factory(worker_settings.database_url),
        LocalStore(worker_settings.local_store_dir),
        uuid.UUID(batch["job"]["id"]),
        worker_settings,
    )
    with session_factory(worker_settings.database_url)() as s:
        (decision,) = s.scalars(select(Decision)).all()
        assert decision.policy_version.startswith("conservative/v2+")
        assert "low_confidence" not in decision.reasons or decision.suggested_label == "empty"
    serving._LOADED.clear()


def test_filtering_release_fills_the_filtered_view_and_samples_audits(
    worker_settings: Settings, tmp_path: Path
) -> None:
    """A release WITH automatic filtering (a test release; the released model
    filters nothing): filtered events stay findable, audits are sampled with a
    recorded rule, and a reviewer can recover a filtered event."""
    serving._LOADED.clear()
    settings_all = worker_settings.model_copy(update={"audit_rate": 1.0})
    model_dir, policy = _fake_model_dir(tmp_path, seed=5, empty_threshold=0.0001, auto_filter=True)
    with TestClient(create_app(settings_all, dispatcher=NoopDispatcher())):
        pass
    with session_factory(settings_all.database_url)() as s:
        release, _ = register_release(
            s, LocalStore(settings_all.local_store_dir), model_dir, policy, None
        )
        activate(s, release.id)
        s.commit()
    batch = _upload(settings_all, 3)
    process_batch(
        session_factory(settings_all.database_url),
        LocalStore(settings_all.local_store_dir),
        uuid.UUID(batch["job"]["id"]),
        settings_all,
    )
    with TestClient(create_app(settings_all, dispatcher=NoopDispatcher())) as c:
        filtered = c.get("/events", params={"disposition": "likely_empty"}).json()
        assert filtered["total"] == 1
        (event,) = filtered["events"]
        d = event["decision"]
        assert d["audit_selected"] is True
        assert d["audit_rule"] == "sha256-uniform(rate=1, seed=wildinbox-audit-v1)"
        assert c.get("/events", params={"audit": True, "reviewed": False}).json()["total"] == 1
        # Recover it: the reviewer saw an animal.
        r = c.post(
            f"/events/{event['id']}/reviews",
            json={"reviewer": "auditor", "outcome": "corrected", "confirmed_label": "bobcat"},
        )
        assert r.status_code == 201
        (row,) = c.get(f"/batches/{batch['id']}/export", params={"format": "json"}).json()[
            "observations"
        ]
        assert (row["observation"], row["label_source"], row["disposition"]) == (
            "bobcat",
            "review",
            "likely_empty",
        )
        assert row["audit_selected"] is True and row["audit_rule"].startswith("sha256-uniform")
        assert c.get("/events", params={"audit": True, "reviewed": False}).json()["total"] == 0
    serving._LOADED.clear()


# ------------------------------------------------- incomplete events (review fix)

CORRUPT = b"\xff\xd8\xff\xe0" + b"not image data" * 40  # passes upload, fails to decode


@pytest.fixture
def filtering(worker_settings: Settings, tmp_path: Path) -> Iterator[Settings]:
    """A test release that filters every event whose usable frames all look
    empty, so any gap in the incomplete-event safeguard shows as likely_empty."""
    serving._LOADED.clear()
    model_dir, policy = _fake_model_dir(tmp_path, seed=7, empty_threshold=0.0001, auto_filter=True)
    with TestClient(create_app(worker_settings, dispatcher=NoopDispatcher())):
        pass
    with session_factory(worker_settings.database_url)() as s:
        release, _ = register_release(
            s, LocalStore(worker_settings.local_store_dir), model_dir, policy, None
        )
        activate(s, release.id)
        s.commit()
    yield worker_settings
    serving._LOADED.clear()


def _process(cfg: Settings, files: list[tuple[str, bytes]], seqs: dict[str, str]) -> dict[str, Any]:
    meta = {"files": {n: {"sequence_id": q, "camera_id": "trail"} for n, q in seqs.items()}}
    with TestClient(create_app(cfg, dispatcher=NoopDispatcher())) as c:
        res = c.post("/batches", files=_files(*files), data={"metadata": json.dumps(meta)})
        assert res.status_code == 202, res.text
        batch = res.json()
        process_batch(
            session_factory(cfg.database_url),
            LocalStore(cfg.local_store_dir),
            uuid.UUID(batch["job"]["id"]),
            cfg,
        )
        events = c.get("/events", params={"batch_id": batch["id"]}).json()["events"]
        details = [c.get(f"/events/{e['id']}").json() for e in events]
        summary = c.get(f"/batches/{batch['id']}").json()
    return {"summary": summary, "events": details}


def _by_file(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {i["filename"]: e for e in events for i in e["images"]}


def test_a_frame_that_fails_to_decode_keeps_its_event_in_review(filtering: Settings) -> None:
    out = _process(
        filtering,
        [("good.jpg", jpeg(1)), ("corrupt.jpg", CORRUPT), ("control.jpg", jpeg(2))],
        {"good.jpg": "s1", "corrupt.jpg": "s1", "control.jpg": "s2"},
    )
    ev = _by_file(out["events"])
    assert ev["control.jpg"]["decision"]["disposition"] == "likely_empty"  # the release filters
    s1 = ev["good.jpg"]
    assert ev["corrupt.jpg"]["id"] == s1["id"]  # the failed frame stays a member
    statuses = {i["filename"]: i["frame_status"] for i in s1["images"]}
    assert statuses == {"good.jpg": "completed", "corrupt.jpg": "invalid"}
    assert s1["decision"]["disposition"] == "needs_review"
    assert "processing_failure" in s1["decision"]["reasons"]


def test_a_file_rejected_at_upload_keeps_its_event_in_review(filtering: Settings) -> None:
    out = _process(
        filtering,
        [("good.jpg", jpeg(3)), ("empty.jpg", b""), ("notes.txt", b"field notes")],
        {"good.jpg": "s1", "empty.jpg": "s1"},
    )
    ev = _by_file(out["events"])
    s1 = ev["good.jpg"]
    assert ev["empty.jpg"]["id"] == s1["id"]
    assert s1["decision"]["disposition"] == "needs_review"
    assert "processing_failure" in s1["decision"]["reasons"]
    # A stray non-image file with nothing to place it creates no event of its own,
    # and stays reported as a failed file.
    assert "notes.txt" not in ev
    assert "notes.txt" in {f["filename"] for f in out["summary"]["failures"]}


def test_failed_files_without_a_usable_companion_stay_batch_failures(
    filtering: Settings,
) -> None:
    out = _process(
        filtering,
        [("a.jpg", CORRUPT), ("b.jpg", CORRUPT + b"x"), ("ok.jpg", jpeg(6))],
        {"a.jpg": "s1", "b.jpg": "s1", "ok.jpg": "s2"},
    )
    # Nothing in s1 can be reviewed, so it is not an event; both files are reported.
    assert set(_by_file(out["events"])) == {"ok.jpg"}
    assert {"a.jpg", "b.jpg"} <= {f["filename"] for f in out["summary"]["failures"]}


def test_a_duplicate_frame_counts_with_its_originals_prediction(filtering: Settings) -> None:
    x = jpeg(4)
    _process(filtering, [("x.jpg", x)], {"x.jpg": "a"})
    out = _process(
        filtering, [("y.jpg", jpeg(5)), ("x-again.jpg", x)], {"y.jpg": "b", "x-again.jpg": "b"}
    )
    (event,) = out["events"]
    statuses = {i["filename"]: i["frame_status"] for i in event["images"]}
    assert statuses == {"y.jpg": "completed", "x-again.jpg": "duplicate"}
    # Both frames are evidence (the duplicate through its original's prediction),
    # so the event is complete and the release may filter it.
    assert event["decision"]["disposition"] == "likely_empty"
    assert "processing_failure" not in event["decision"]["reasons"]


class ByContentScorer:
    """Deterministic scores by file content: the listed hashes look like a
    bobcat, everything else looks empty."""

    def __init__(self, release: ModelRelease, animals: set[str]) -> None:
        self.release_id, self.class_names = release.id, list(release.class_names)
        self.animals = animals

    def score(self, images: Any, sha256s: Any) -> list[dict[str, float]]:
        out = []
        for sha in sha256s:
            top = "bobcat" if sha in self.animals else "empty"
            rest = (1 - 0.97) / (len(self.class_names) - 1)
            out.append({c: 0.97 if c == top else rest for c in self.class_names})
        return out


def test_overlapping_upload_keeps_the_earlier_animal_frame_as_evidence(
    worker_settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """External review, issue 2: upload an animal frame, then a batch with the
    same frame plus an empty frame of its sequence. The second event must not
    be decided on the empty frame alone."""
    serving._LOADED.clear()
    model_dir, policy = _fake_model_dir(
        tmp_path, seed=8, empty_threshold=0.6, auto_filter=True
    )  # calibrated 0.97 -> ~0.68
    with TestClient(create_app(worker_settings, dispatcher=NoopDispatcher())):
        pass
    with session_factory(worker_settings.database_url)() as s:
        release, _ = register_release(
            s, LocalStore(worker_settings.local_store_dir), model_dir, policy, None
        )
        activate(s, release.id)
        s.commit()
    animal, empty = jpeg(20), jpeg(21)
    animal_sha = hashlib.sha256(animal).hexdigest()
    monkeypatch.setattr(
        process, "scorer_for", lambda r, store, device="cpu": ByContentScorer(r, {animal_sha})
    )

    first = _process(worker_settings, [("bobcat.jpg", animal)], {"bobcat.jpg": "seq-1"})
    assert first["events"][0]["decision"]["disposition"] == "needs_review"  # an animal

    control = _process(worker_settings, [("bare.jpg", jpeg(22))], {"bare.jpg": "seq-2"})
    assert control["events"][0]["decision"]["disposition"] == "likely_empty"  # it filters

    out = _process(
        worker_settings,
        [("bobcat-copy.jpg", animal), ("after.jpg", empty)],
        {"bobcat-copy.jpg": "seq-1", "after.jpg": "seq-1"},
    )
    (event,) = out["events"]
    frames = {i["filename"]: i for i in event["images"]}
    assert set(frames) == {"bobcat-copy.jpg", "after.jpg"}  # membership preserved
    assert frames["bobcat-copy.jpg"]["frame_status"] == "duplicate"
    # The duplicate's evidence is its original's prediction: shown and used.
    assert frames["bobcat-copy.jpg"]["prediction"]["suggested_label"] == "bobcat"
    assert event["decision"]["disposition"] == "needs_review"
    assert event["decision"]["suggested_label"] == "bobcat"
    assert out["summary"]["counts"]["duplicate"] == 1  # stored and scored once
    assert _count(worker_settings, Prediction) == 3  # bobcat, bare, after: no rescoring
    serving._LOADED.clear()


def test_a_duplicate_scored_only_by_another_release_sends_its_event_to_review(
    worker_settings: Settings, tmp_path: Path
) -> None:
    """The original's prediction is reused only under the same release; otherwise
    the duplicate is a pending frame and the event goes to review."""
    serving._LOADED.clear()
    frame = jpeg(30)
    _process(worker_settings, [("old.jpg", frame)], {"old.jpg": "s"})  # test predictor release
    model_dir, policy = _fake_model_dir(tmp_path, seed=9, empty_threshold=0.0001, auto_filter=True)
    with session_factory(worker_settings.database_url)() as s:
        release, _ = register_release(
            s, LocalStore(worker_settings.local_store_dir), model_dir, policy, None
        )
        activate(s, release.id)
        s.commit()
    out = _process(
        worker_settings,
        [("old-again.jpg", frame), ("new.jpg", jpeg(31))],
        {"old-again.jpg": "t", "new.jpg": "t"},
    )
    (event,) = out["events"]
    assert event["decision"]["disposition"] == "needs_review"
    assert "processing_failure" in event["decision"]["reasons"]
    serving._LOADED.clear()
