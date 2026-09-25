"""The review interface's only way to the data: the HTTP API.

Going through the API keeps every review on the same validation and audit path
as any other client.
"""

from __future__ import annotations

import json
from typing import Any

import httpx


class ApiError(RuntimeError):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status, self.detail = status, detail


class ApiClient:
    def __init__(self, base_url: str, timeout: float = 30) -> None:
        self.http = httpx.Client(base_url=base_url, timeout=timeout)

    def _json(self, res: httpx.Response) -> Any:
        if res.status_code >= 400:
            try:
                detail = res.json().get("detail", res.text)
            except ValueError:
                detail = res.text
            raise ApiError(res.status_code, str(detail))
        return res.json()

    def version(self) -> dict[str, Any]:
        out: dict[str, Any] = self._json(self.http.get("/version"))
        return out

    def study_join(self, plan_id: str, code: str) -> dict[str, Any]:
        out: dict[str, Any] = self._json(
            self.http.post(f"/study/plans/{plan_id}/participants", json={"code": code})
        )
        return out

    def study_trial(self, plan_id: str, trial: dict[str, Any]) -> None:
        self._json(self.http.post(f"/study/plans/{plan_id}/trials", json=trial))

    def study_rating(self, plan_id: str, rating: dict[str, Any]) -> None:
        self._json(self.http.post(f"/study/plans/{plan_id}/ratings", json=rating))

    def monitoring(self) -> dict[str, Any]:
        out: dict[str, Any] = self._json(self.http.get("/monitoring", timeout=120))
        return out

    def batches(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = self._json(self.http.get("/batches"))["batches"]
        return out

    def batch(self, batch_id: str) -> dict[str, Any]:
        out: dict[str, Any] = self._json(self.http.get(f"/batches/{batch_id}"))
        return out

    def events(self, **params: Any) -> dict[str, Any]:
        clean = {k: v for k, v in params.items() if v is not None}
        out: dict[str, Any] = self._json(self.http.get("/events", params=clean))
        return out

    def event(self, event_id: str) -> dict[str, Any]:
        out: dict[str, Any] = self._json(self.http.get(f"/events/{event_id}"))
        return out

    def review(
        self,
        event_id: str,
        reviewer: str,
        outcome: str,
        label: str | None,
        note: str | None = None,
    ) -> dict[str, Any]:
        body = {"reviewer": reviewer, "outcome": outcome, "confirmed_label": label, "note": note}
        out: dict[str, Any] = self._json(self.http.post(f"/events/{event_id}/reviews", json=body))
        return out

    def upload(
        self, files: list[tuple[str, bytes]], camera_id: str | None = None
    ) -> dict[str, Any]:
        data = {"metadata": json.dumps({"camera_id": camera_id})} if camera_id else None
        res = self.http.post(
            "/batches",
            files=[
                ("files", (name, content, "application/octet-stream")) for name, content in files
            ],
            data=data,
            timeout=600,
        )
        out: dict[str, Any] = self._json(res)
        return out

    def thumbnail(self, image_id: str, size: int = 320) -> bytes | None:
        res = self.http.get(f"/images/{image_id}/thumbnail", params={"size": size})
        return res.content if res.status_code == 200 else None

    def export(self, batch_id: str, fmt: str = "csv") -> bytes:
        res = self.http.get(f"/batches/{batch_id}/export", params={"format": fmt}, timeout=120)
        if res.status_code >= 400:
            raise ApiError(res.status_code, res.text)
        return res.content
