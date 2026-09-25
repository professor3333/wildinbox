"""Staging hardening: token access, readiness that loads the model, structured
logs, and settings for AWS S3."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from wildinbox.api.app import create_app
from wildinbox.api.auth import new_token, principal, token_hash
from wildinbox.inference import serving
from wildinbox.logs import JsonFormatter
from wildinbox.settings import Settings
from wildinbox.storage.db import session_factory
from wildinbox.storage.models import ModelRelease
from wildinbox.storage.objects import LocalStore
from wildinbox.workers.dispatch import preload

from .test_app import NoopDispatcher, database_url, settings  # noqa: F401
from .test_uploads import jpeg
from .test_workers import real_release, worker_settings  # noqa: F401

TOKEN = new_token()


@pytest.fixture
def secured(settings: Settings) -> Iterator[TestClient]:  # noqa: F811
    s = settings.model_copy(update={"auth": "tokens", "api_tokens": {"alice": token_hash(TOKEN)}})
    with TestClient(create_app(s, dispatcher=NoopDispatcher())) as c:
        yield c


def test_principal_matches_only_a_valid_bearer_token() -> None:
    tokens = {"alice": token_hash(TOKEN), "ui": token_hash("other")}
    assert principal(f"Bearer {TOKEN}", tokens) == "alice"
    assert principal(f"bearer  {TOKEN} ", tokens) == "alice"
    assert principal("Bearer wrong", tokens) is None
    assert principal(TOKEN, tokens) is None  # no scheme
    assert principal(f"Basic {TOKEN}", tokens) is None
    assert principal(None, tokens) is None


def test_data_endpoints_require_a_token(secured: TestClient) -> None:
    for path in ("/batches", "/events", "/version", "/releases", "/metrics", "/whoami"):
        res = secured.get(path)
        assert res.status_code == 401, path
        assert res.json()["error"] == "unauthorized"
        assert res.headers["WWW-Authenticate"] == "Bearer"
    upload = secured.post("/batches", files=[("files", ("a.jpg", jpeg(1), "image/jpeg"))])
    assert upload.status_code == 401
    bad = secured.get("/events", headers={"Authorization": "Bearer nope"})
    assert bad.status_code == 401


def test_a_valid_token_gets_through_and_is_named(secured: TestClient) -> None:
    auth = {"Authorization": f"Bearer {TOKEN}", "X-Request-ID": "req-123"}
    res = secured.get("/whoami", headers=auth)
    assert res.json() == {"principal": "alice", "auth": "tokens"}
    assert res.headers["X-Request-ID"] == "req-123"
    upload = secured.post(
        "/batches", files=[("files", ("a.jpg", jpeg(1), "image/jpeg"))], headers=auth
    )
    assert upload.status_code == 202
    assert secured.get("/events", headers=auth).status_code == 200


def test_liveness_readiness_and_upload_page_stay_public(secured: TestClient) -> None:
    assert secured.get("/health").status_code == 200
    assert secured.get("/ready").status_code in (200, 503)  # never 401
    page = secured.get("/")
    assert page.status_code == 200 and 'name="token"' in page.text


def test_token_mode_without_tokens_refuses_to_start(settings: Settings) -> None:  # noqa: F811
    with pytest.raises(RuntimeError, match="WILDINBOX_API_TOKENS is empty"):
        create_app(settings.model_copy(update={"auth": "tokens", "api_tokens": {}}))


def test_ready_loads_the_expected_release(settings: Settings) -> None:  # noqa: F811
    with TestClient(create_app(settings, dispatcher=NoopDispatcher())) as c:
        body = c.get("/ready").json()
    assert body["status"] == "ready"
    assert body["checks"]["model"] == {"ok": True, "ms": body["checks"]["model"]["ms"],
                                      "detail": "test-predictor-v0"}  # fmt: skip
    assert "queue" not in body["checks"]  # inline dispatch has no queue


def test_ready_fails_when_another_release_is_active(settings: Settings) -> None:  # noqa: F811
    s = settings.model_copy(update={"expected_release": "finetune@abc"})
    with TestClient(create_app(s, dispatcher=NoopDispatcher())) as c:
        res = c.get("/ready")
    assert res.status_code == 503
    assert "expects 'finetune@abc'" in res.json()["checks"]["model"]["error"]


def test_ready_fails_when_the_model_artifact_is_corrupt(
    worker_settings: Settings,  # noqa: F811
    real_release: ModelRelease,  # noqa: F811
) -> None:
    s = worker_settings.model_copy(update={"expected_release": real_release.id})
    with TestClient(create_app(s, dispatcher=NoopDispatcher())) as c:
        assert c.get("/ready").status_code == 200  # the real weights load
        serving._LOADED.clear()
        assert real_release.weights_key is not None
        LocalStore(s.local_store_dir).put(real_release.weights_key, b"garbage", "x")
        res = c.get("/ready")
    assert res.status_code == 503
    assert "SHA-256" in res.json()["checks"]["model"]["error"]
    # A worker refuses to start on the same artifact.
    with pytest.raises(ValueError, match="SHA-256"):
        preload(s, session_factory(s.database_url), LocalStore(s.local_store_dir))


def test_worker_preload_refuses_an_unregistered_release(settings: Settings) -> None:  # noqa: F811
    s = settings.model_copy(update={"expected_release": "missing@1"})
    with TestClient(create_app(settings, dispatcher=NoopDispatcher())):
        pass  # creates the schema's test release
    with pytest.raises(RuntimeError, match="not registered"):
        preload(s, session_factory(s.database_url), LocalStore(s.local_store_dir))


def test_settings_for_aws_s3_and_token_hashes() -> None:
    s = Settings(object_store_url="", object_store_access_key="", api_tokens={"a": "AB" * 32})
    assert s.object_store_url is None and s.object_store_access_key is None
    assert s.api_tokens == {"a": "ab" * 32}
    with pytest.raises(ValidationError, match="SHA-256"):
        Settings(api_tokens={"a": "plain-token"})


def test_json_logs_carry_structured_fields() -> None:
    record = logging.LogRecord("wildinbox.x", logging.INFO, "f", 1, "job %s", ("done",), None)
    record.fields = {"job_id": "j1", "seconds": 1.5, "message": "not allowed to override"}
    out = json.loads(JsonFormatter().format(record))
    assert out["message"] == "job done" and out["level"] == "INFO"
    assert out["job_id"] == "j1" and out["seconds"] == 1.5


def test_upload_reports_its_server_phases(settings: Settings) -> None:  # noqa: F811
    with TestClient(create_app(settings, dispatcher=NoopDispatcher())) as c:
        files = [("files", (f"{i}.jpg", jpeg(i), "image/jpeg")) for i in range(3)]
        res = c.post("/batches", files=files)
    assert res.status_code == 202
    phases = dict(p.strip().split(";dur=") for p in res.headers["Server-Timing"].split(","))
    assert set(phases) == {"receive", "validate", "store", "db"}
    assert all(float(v) >= 0 for v in phases.values())
    job = res.json()["job"]
    assert job["created_at"] >= res.json()["created_at"]  # queued after originals stored


def test_token_new_prints_tokens_and_a_ready_env_line(capsys: pytest.CaptureFixture[str]) -> None:
    from wildinbox.cli import main

    assert main(["token", "new", "alice", "ui"]) == 0
    lines = capsys.readouterr().out.splitlines()
    tokens = [line.strip() for line in lines if line.strip().startswith("wi_")]
    env = lines[-1]
    assert env.startswith("WILDINBOX_API_TOKENS='") and env.endswith("'")
    hashes = json.loads(env.split("=", 1)[1].strip("'"))
    assert list(hashes) == ["alice", "ui"]
    assert [token_hash(t) for t in tokens] == list(hashes.values())
    assert Settings(api_tokens=hashes).api_tokens == hashes  # the line parses as settings
    assert main(["token", "new", "a", "a"]) == 2
