"""Readiness: can this deployment process batches with the release it should?

Liveness (`/health`) only says the API answers. Readiness also checks the
queue and object store, that the active release is the one the deployment
expects, and that its weights load and pass their SHA-256 check in this
process. A deployment whose model artifact is missing or corrupt is therefore
never reported ready.
"""

from __future__ import annotations

import time
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from wildinbox.inference.releases import active_release_id
from wildinbox.inference.serving import scorer_for
from wildinbox.settings import Settings
from wildinbox.storage.models import ModelRelease
from wildinbox.storage.objects import ObjectStore


def _run(checks: dict[str, Any], name: str, fn: Any) -> Any:
    start = time.perf_counter()
    try:
        detail = fn()
        checks[name] = {"ok": True, "ms": round(1000 * (time.perf_counter() - start), 1)}
        if detail:
            checks[name]["detail"] = detail
        return detail
    except Exception as e:
        checks[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:500]}
        return None


def check(session: Session, settings: Settings, store: ObjectStore) -> tuple[bool, dict[str, Any]]:
    checks: dict[str, Any] = {}
    _run(checks, "database", lambda: session.execute(select(1)) and None)
    if settings.dispatch == "rq":

        def _redis() -> None:
            from redis import Redis

            Redis.from_url(settings.redis_url, socket_timeout=3).ping()

        _run(checks, "queue", _redis)
    _run(checks, "object_store", store.check)

    def _release() -> str:
        active = active_release_id(session, settings)
        if settings.expected_release and active != settings.expected_release:
            raise RuntimeError(
                f"active release is {active!r}, deployment expects {settings.expected_release!r}"
            )
        release = session.get(ModelRelease, active)
        if release is None:
            raise RuntimeError(f"release {active!r} is not registered")
        scorer_for(release, store, settings.inference_device)  # loaded once, then cached
        return active

    if checks["database"]["ok"]:
        _run(checks, "model", _release)
    else:
        checks["model"] = {"ok": False, "error": "skipped: database unavailable"}
    ok = all(c["ok"] for c in checks.values())
    return ok, {
        "status": "ready" if ok else "not_ready",
        "expected_release": settings.expected_release,
        "checks": checks,
    }
