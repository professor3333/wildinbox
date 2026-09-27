"""Seed a deployment's database with synthetic processing history, for load
tests that need a large, old history (not for anything a person reviews).

Run inside the API container of a disposable stack, so it uses that stack's
database settings:

    docker compose -p wildinbox-history exec -T api \\
        python - --batches 250 --images-per-batch 1000 --days 365 < scripts/seed_history.py

Each synthetic batch is one camera's memory card: images grouped into capture
events of 1-5 frames, a prediction per image, a decision per event under the
active release, a succeeded job, and reviews (some superseded) on a share of
events. Uploads are spread evenly over the last `--days` days, so history
reaches back past monitoring's history window. Values are random but
deterministic (`--seed`). No image files are stored: thumbnails and originals
of synthetic images do not exist.

Every synthetic batch is marked `"synthetic_history": true` in its manifest;
`--remove` deletes them all (and, by cascade, everything under them).
"""

from __future__ import annotations

import argparse
import random
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from wildinbox.inference.releases import active_release_id
from wildinbox.settings import Settings
from wildinbox.storage.db import session_factory
from wildinbox.storage.models import ModelRelease

SPECIES_WEIGHTS = {  # roughly CCT20's mix of supported classes, empty most common
    "empty": 45,
    "opossum": 12,
    "raccoon": 10,
    "coyote": 9,
    "rabbit": 8,
    "bobcat": 6,
    "cat": 5,
    "dog": 5,
}


def _dsn(url: str) -> str:
    return url.replace("postgresql+psycopg://", "postgresql://", 1)


def _probabilities(rng: random.Random, classes: list[str], label: str) -> dict[str, float]:
    top = rng.uniform(0.35, 0.99)
    rest = [c for c in classes if c != label]
    weights = [rng.random() for _ in rest]
    total = sum(weights) or 1.0
    out = {c: round((1 - top) * w / total, 4) for c, w in zip(rest, weights, strict=True)}
    out[label] = round(top, 4)
    return out


def seed(args: argparse.Namespace, settings: Settings) -> dict[str, Any]:
    rng = random.Random(args.seed)
    now = datetime.now(UTC)
    with session_factory(settings.database_url)() as s:  # the release the API serves
        release_id = active_release_id(s, settings)
        release = s.get(ModelRelease, release_id) if release_id else None
        if release is None:
            raise SystemExit("no active release: start the API once first")
        classes, policy = list(release.class_names), release.policy_version
    with psycopg.connect(_dsn(settings.database_url)) as conn:
        labels = [c for c in SPECIES_WEIGHTS if c in classes] or list(classes)
        weights = [SPECIES_WEIGHTS.get(c, 1) for c in labels]
        cameras = [f"history-cam-{n:02d}" for n in range(args.cameras)]
        Rows = list[tuple[Any, ...]]
        batches: Rows = []
        images: Rows = []
        predictions: Rows = []
        events: Rows = []
        decisions: Rows = []
        jobs: Rows = []
        reviews: Rows = []
        step = timedelta(days=args.days) / max(args.batches, 1)
        for b in range(args.batches):
            batch_id = uuid.uuid4()
            uploaded = now - timedelta(days=args.days) + step * b
            camera = cameras[b % len(cameras)]
            batches.append(
                (
                    batch_id,
                    settings.workspace,
                    f"synthetic:{args.seed}:{b}",
                    "completed",
                    Jsonb(
                        {
                            "synthetic_history": True,
                            "seed": args.seed,
                            "files": [],
                            "grouping": {
                                "gap_seconds": 5.0,
                                "gap_source": "default",
                                "time_gap_rule": "time_gap/v1(gap_s=5)",
                            },
                        }
                    ),
                    uploaded,
                    uploaded,
                    uploaded + timedelta(minutes=2),
                )
            )
            jobs.append(
                (
                    uuid.uuid4(),
                    batch_id,
                    "process_batch",
                    "succeeded",
                    1,
                    3,
                    release_id,
                    uploaded,
                    uploaded + timedelta(seconds=1),
                    uploaded + timedelta(seconds=args.images_per_batch * 0.08),
                )
            )
            card_start = uploaded - timedelta(days=7)
            position = 0
            while position < args.images_per_batch:
                size = min(
                    rng.choice((1, 1, 2, 2, 3, 3, 3, 4, 5)), args.images_per_batch - position
                )
                event_id = uuid.uuid4()
                label = rng.choices(labels, weights)[0]
                start = (card_start + timedelta(seconds=rng.uniform(0, 7 * 86400))).replace(
                    tzinfo=None, microsecond=0
                )
                night = start.hour >= 19 or start.hour < 6
                confidences = []
                for f in range(size):
                    image_id = uuid.uuid4()
                    probs = _probabilities(rng, classes, label)
                    confidences.append(probs[label])
                    images.append(
                        (
                            image_id,
                            batch_id,
                            position,
                            f"IMG_{position:05d}.JPG",
                            rng.randint(80_000, 400_000),
                            uuid.uuid4().hex + uuid.uuid4().hex,
                            "image/jpeg",
                            camera,
                            start + timedelta(seconds=f),
                            "valid",
                            Jsonb(
                                {
                                    "night": night,
                                    "blur": round(rng.lognormvariate(4.5, 0.8), 1),
                                    "brightness": round(rng.uniform(20, 200), 1),
                                }
                            ),
                            event_id,
                            uploaded,
                        )
                    )
                    predictions.append(
                        (
                            uuid.uuid4(),
                            image_id,
                            event_id,
                            release_id,
                            Jsonb(probs),
                            Jsonb(probs),
                            label,
                            probs[label],
                            uploaded,
                        )
                    )
                    position += 1
                disposition = rng.choices(
                    ("needs_review", "likely_empty", "species_identified"), (80, 15, 5)
                )[0]
                if disposition == "likely_empty":
                    label = "empty"
                automatic = disposition != "needs_review"
                confidence = round(min(confidences), 4)
                events.append(
                    (
                        event_id,
                        batch_id,
                        camera,
                        "time_gap/v1(gap_s=5)",
                        uuid.uuid4().hex,
                        start,
                        start + timedelta(seconds=size - 1),
                        uploaded,
                    )
                )
                decisions.append(
                    (
                        uuid.uuid4(),
                        event_id,
                        release_id,
                        policy,
                        disposition,
                        label,
                        confidence,
                        Jsonb([] if automatic else ["low_confidence"]),
                        automatic and rng.random() < 0.05,
                        uploaded,
                    )
                )
                if rng.random() < args.review_share:
                    previous = None
                    for _ in range(2 if rng.random() < 0.1 else 1):
                        review_id = uuid.uuid4()
                        outcome = rng.choices(("confirmed", "corrected", "unresolved"), (6, 3, 1))[
                            0
                        ]
                        confirmed = {
                            "confirmed": label,
                            "corrected": rng.choice(labels),
                            "unresolved": None,
                        }[outcome]
                        reviews.append(
                            (
                                review_id,
                                event_id,
                                "history-reviewer",
                                outcome,
                                label,
                                confirmed,
                                previous,
                                uploaded + timedelta(days=1),
                            )
                        )
                        previous = review_id

        tables: dict[str, tuple[str, Rows]] = {
            "batches": (
                "id, workspace, request_key, status, manifest, created_at, updated_at, "
                "completed_at",
                batches,
            ),
            "jobs": (
                "id, batch_id, kind, status, attempts, max_attempts, model_release_id, "
                "created_at, started_at, finished_at",
                jobs,
            ),
            "events": (
                "id, batch_id, camera_id, grouping_rule, group_key, start_at, end_at, created_at",
                events,
            ),
            "images": (
                "id, batch_id, position, original_filename, size_bytes, sha256, content_type, "
                "camera_id, captured_at, validation_status, quality, event_id, created_at",
                images,
            ),
            "predictions": (
                "id, image_id, event_id, model_release_id, class_probabilities, "
                "calibrated_probabilities, suggested_label, confidence, created_at",
                predictions,
            ),
            "decisions": (
                "id, event_id, model_release_id, policy_version, disposition, suggested_label, "
                "confidence, reasons, audit_selected, created_at",
                decisions,
            ),
            "reviews": (
                "id, event_id, reviewer, outcome, suggested_label, confirmed_label, "
                "previous_review_id, created_at",
                reviews,
            ),
        }
        with conn.cursor() as cur:
            for table, (columns, rows) in tables.items():
                with cur.copy(f"COPY {table} ({columns}) FROM STDIN") as copy:
                    for row in rows:
                        copy.write_row(row)
        conn.execute("ANALYZE")
    return {table: len(rows) for table, (_, rows) in tables.items()}


def remove(settings: Settings) -> int:
    with psycopg.connect(_dsn(settings.database_url)) as conn:
        cur = conn.execute("DELETE FROM batches WHERE manifest ? 'synthetic_history'")
        return cur.rowcount


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--batches", type=int, default=250)
    ap.add_argument("--images-per-batch", type=int, default=1000)
    ap.add_argument("--cameras", type=int, default=40)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--review-share", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--remove", action="store_true", help="delete every synthetic batch")
    args = ap.parse_args()
    settings = Settings()
    if args.remove:
        print(f"removed {remove(settings)} synthetic batches")
        return
    started = time.monotonic()
    counts = seed(args, settings)
    print({**counts, "seconds": round(time.monotonic() - started, 1)})


if __name__ == "__main__":
    main()
