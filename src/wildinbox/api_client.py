"""HTTP client for CLI commands that work through the API (snapshots, the
review study): one way to authenticate and to check responses."""

from __future__ import annotations

import os
from typing import Any

import httpx


class APIError(RuntimeError):
    """The API refused or failed a request the command needs."""


def api_client(api_url: str) -> httpx.Client:
    """Client for `api_url`, sending `Authorization: Bearer $WILDINBOX_TOKEN` when
    set (the convention of the UI and scripts)."""
    token = os.environ.get("WILDINBOX_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return httpx.Client(base_url=api_url, timeout=120, headers=headers)


def request(api: httpx.Client, method: str, path: str, **kwargs: Any) -> httpx.Response:
    """The response, or `APIError` for any non-2xx status; bodies are only read
    from successful responses."""
    try:
        res = api.request(method, path, **kwargs)
    except httpx.HTTPError as e:
        raise APIError(f"{method} {path}: {e}") from e
    if res.status_code in (401, 403):
        sent = "a token" if "authorization" in api.headers else "no token"
        raise APIError(
            f"{method} {path}: {res.status_code} {res.reason_phrase} ({sent} sent); "
            "set WILDINBOX_TOKEN to a valid API token for this deployment"
        )
    if not res.is_success:
        raise APIError(f"{method} {path}: {res.status_code} {res.reason_phrase}: {res.text[:200]}")
    return res
