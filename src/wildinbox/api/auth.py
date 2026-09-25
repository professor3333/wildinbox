"""Bearer-token access for the single workspace.

Tokens are random strings handed to each person or service; the deployment
stores only their SHA-256 (`WILDINBOX_API_TOKENS`), so a leaked settings file
does not leak working tokens. Every endpoint except liveness, readiness, the
API schema, and the static upload page requires a valid token.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

PUBLIC_PATHS = frozenset({"/health", "/ready", "/", "/docs", "/openapi.json"})


def new_token() -> str:
    return "wi_" + secrets.token_urlsafe(32)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def principal(authorization: str | None, tokens: dict[str, str]) -> str | None:
    """The name whose token matches the `Authorization` header, or None."""
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    digest = token_hash(token.strip())
    match = None
    for name, expected in tokens.items():  # compare against every entry: constant work
        if hmac.compare_digest(digest, expected):
            match = name
    return match
