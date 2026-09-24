"""Smoke test against a running deployment: upload -> process -> results.

    uv run python scripts/smoke.py [--url http://localhost:8000]

Uses synthetic images and the TEST predictor, so it checks plumbing only.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import uuid

import httpx
import numpy as np
from PIL import Image


def jpeg(seed: int, when: str) -> bytes:
    arr = np.random.default_rng(seed).integers(0, 256, (120, 160, 3), dtype=np.uint8)
    exif = Image.Exif()
    exif.get_ifd(0x8769)[0x9003] = when
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", exif=exif)
    return buf.getvalue()


def check(cond: bool, message: str) -> None:
    print(("ok   " if cond else "FAIL ") + message)
    if not cond:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    run = uuid.uuid4().hex[:8]
    files = [
        (
            "files",
            (
                f"{run}-{i}.jpg",
                jpeg(hash(run) % 10_000 + i, f"2024:05:01 21:03:0{i}"),
                "image/jpeg",
            ),
        )
        for i in range(4)
    ]
    files.append(("files", (f"{run}-notes.gif", b"GIF89a" + bytes(16), "image/gif")))
    meta = {"metadata": f'{{"camera_id": "smoke-{run}"}}'}

    with httpx.Client(base_url=args.url, timeout=30) as c:
        check(c.get("/health").status_code == 200, "API healthy")
        res = c.post("/batches", files=files, data=meta)
        check(res.status_code == 202, f"batch accepted ({res.status_code})")
        batch = res.json()
        deadline = time.time() + args.timeout
        while batch["status"] in ("queued", "processing") and time.time() < deadline:
            time.sleep(1)
            batch = c.get(f"/batches/{batch['id']}").json()
        check(batch["status"] == "completed_with_errors", f"batch finished: {batch['status']}")
        counts = batch["counts"]
        check(counts["valid"] == 4 and counts["invalid"] == 1, f"counts {counts}")
        check(counts["events"] == 1, "4 frames 1 s apart grouped into 1 event")
        check(batch["release"]["is_test"] is True, "results labeled as TEST predictor output")

        images = c.get(f"/batches/{batch['id']}/images").json()["images"]
        gif = next(i for i in images if i["filename"].endswith(".gif"))
        check((gif["validation_error"] or "").startswith("unsupported_type"), "GIF rejected")
        first = next(i for i in images if i["validation_status"] == "valid")
        check(c.get(first["original_url"]).content == files[0][1][1], "original retrievable")

        events = c.get("/events", params={"batch_id": batch["id"]}).json()["events"]
        check(events[0]["decision"]["disposition"] == "needs_review", "event needs review")

        again = c.post("/batches", files=files, data=meta).json()
        check(again["id"] == batch["id"] and again.get("duplicate_request"), "resubmit deduped")
        check(again["job"]["attempts"] == 1, "no second job run")
    print(f"smoke test passed: {args.url}/batches/{batch['id']}/view")


if __name__ == "__main__":
    main()
