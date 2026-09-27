"""The synthetic history seeder writes data the API reads like real history."""

from __future__ import annotations

import argparse
import importlib.util

from fastapi.testclient import TestClient

from wildinbox.api.app import create_app
from wildinbox.settings import Settings

from .conftest import REPO_ROOT
from .test_app import database_url, settings  # noqa: F401

_spec = importlib.util.spec_from_file_location(
    "seed_history", REPO_ROOT / "scripts/seed_history.py"
)
assert _spec and _spec.loader
seed_history = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(seed_history)


def test_seeded_history_reads_like_real_history_and_can_be_removed(
    settings: Settings,  # noqa: F811
) -> None:
    args = argparse.Namespace(
        batches=4, images_per_batch=30, cameras=2, days=200, review_share=0.5, seed=1
    )
    with TestClient(create_app(settings)) as c:  # startup registers the test release
        counts = seed_history.seed(args, settings)
        assert counts["batches"] == 4 and counts["images"] == 120
        assert counts["predictions"] == 120 and counts["decisions"] == counts["events"]

        listing = c.get("/batches", params={"limit": 10}).json()
        assert listing["total"] == 4
        assert all(b["images"] == 30 and b["status"] == "completed" for b in listing["batches"])
        page = c.get("/events", params={"limit": 500}).json()
        assert page["total"] == counts["events"]
        reviewed = [e for e in page["events"] if e["latest_review"]]
        assert reviewed  # review chains are well formed: every row serializes
        history = c.get("/monitoring").json()["history"]
        assert history["events_analysed"] + history["older_events_excluded"] == counts["events"]
        assert 0 < history["older_events_excluded"] < counts["events"]  # history spans 200 days

        assert seed_history.remove(settings) == 4
        assert c.get("/events").json()["total"] == 0
