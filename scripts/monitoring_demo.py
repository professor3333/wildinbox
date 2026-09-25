"""Stage 11 acceptance demo for monitoring, against the Docker Compose deployment.

    uv run python scripts/monitoring_demo.py

Part 1, staged worker failure. Uploads a drill batch (camera "crash-drill",
so real cameras' statistics stay clean), SIGKILLs the worker mid-batch, and
polls GET /monitoring until the operational alerts fire. Then restarts the
worker, waits for the batch to finish, and records which alerts cleared.

Part 2, changed inputs. From camera 90's photos (its deployment batch is the
baseline), uploads a control batch of unchanged photos (re-encoded at JPEG
quality 95, since exact duplicates are skipped), then the SAME photos
degraded (blurred, fogged, darkened, as through a dirty or misted lens), and
records camera 90's model-behavior indicators and alerts after each.

Writes reports/monitoring/acceptance.json.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, ImageEnhance, ImageFilter


def auth_headers() -> dict[str, str]:
    """Bearer token from WILDINBOX_TOKEN, for deployments that require one."""
    token = os.environ.get("WILDINBOX_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def check(cond: bool, message: str) -> None:
    print(("ok   " if cond else "FAIL ") + message)
    if not cond:
        sys.exit(1)


def compose(*args: str) -> str:
    return subprocess.run(
        ["docker", "compose", *args], check=True, capture_output=True, text=True
    ).stdout


def post(api: httpx.Client, files: list[tuple[str, bytes]], meta: dict[str, Any], key: str) -> str:
    res = api.post(
        "/batches",
        files=[("files", (n, d, "image/jpeg")) for n, d in files],
        data={"metadata": json.dumps(meta)},
        headers={"Idempotency-Key": key},
        timeout=600,
    )
    res.raise_for_status()
    return str(res.json()["id"])


def wait(api: httpx.Client, batch_id: str, timeout: float = 1800) -> dict[str, Any]:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        s: dict[str, Any] = api.get(f"/batches/{batch_id}").json()
        if s["progress"]["finished"]:
            return s
        time.sleep(2)
    sys.exit(f"batch {batch_id} did not finish")


def alerts(m: dict[str, Any], area: str | None = None) -> list[dict[str, Any]]:
    return [a for a in m["alerts"] if area is None or a["area"] == area]


def reencode(data: bytes, tag: str) -> bytes:
    """Same picture, new bytes (JPEG quality 95 plus a comment naming the run):
    the deployment skips exact duplicates of photos it already holds."""
    buf = io.BytesIO()
    Image.open(io.BytesIO(data)).convert("RGB").save(buf, "JPEG", quality=95, comment=tag.encode())
    return buf.getvalue()


def degrade(data: bytes, tag: str) -> bytes:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = img.filter(ImageFilter.GaussianBlur(6))
    img = ImageEnhance.Contrast(img).enhance(0.45)
    img = ImageEnhance.Brightness(img).enhance(0.6)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85, comment=tag.encode())
    return buf.getvalue()


def worker_failure(api: httpx.Client, args: argparse.Namespace, run: str) -> dict[str, Any]:
    print("Part 1: staged worker failure")
    before = api.get("/monitoring").json()
    check(
        not [a for a in alerts(before, "operations") if a["level"] == "critical"],
        "no critical operational alert before the drill",
    )
    live = before["operations"]["workers"]["live"]
    deaths_before = len(before["operations"]["workers"]["died_in_window"])
    check(bool(live), f"{len(live)} live worker(s) before the drill")
    images = sorted((args.drill_dir / "images").iterdir())
    files = [(p.name, reencode(p.read_bytes(), f"drill-{run}")) for p in images]
    bid = post(api, files, {"camera_id": "crash-drill"}, f"crash-drill-{run}")
    while True:
        p = api.get(f"/batches/{bid}").json()["progress"]
        if p["images_scored"] >= 0.2 * p["images_to_score"]:
            break
        time.sleep(0.5)
    compose("kill", "-s", "SIGKILL", "worker")
    killed = time.monotonic()
    print(f"  SIGKILL worker at {p['images_scored']}/{p['images_to_score']} scored")
    wanted = {
        "stale lease": lambda a: a["level"] == "critical" and "lease" in a["message"],
        "no live worker": lambda a: a["level"] == "critical" and "no worker" in a["message"],
        # Earlier deaths stay on record for 24 h: only a NEW death counts.
        "worker died": lambda a: (
            "died without shutting down" in a["message"] and a["value"] > deaths_before
        ),
    }
    seen: dict[str, float] = {}
    fired: list[dict[str, Any]] = []
    while len(seen) < len(wanted) and time.monotonic() - killed < 400:
        m = api.get("/monitoring").json()
        for name, match in wanted.items():
            hit = [a for a in alerts(m, "operations") if match(a)]
            if hit and name not in seen:
                seen[name] = round(time.monotonic() - killed, 1)
                fired.extend(hit)
                print(
                    f"  {seen[name]:6.1f}s after the kill: {hit[0]['level']}: {hit[0]['message']}"
                )
        time.sleep(5)
    check(len(seen) == len(wanted), "every expected operational alert fired")
    metrics = api.get("/metrics").text
    critical = next(
        line
        for line in metrics.splitlines()
        if line.startswith('wildinbox_alerts{level="critical"}')
    )
    check(not critical.endswith(" 0"), f"/metrics exposes it: {critical}")

    compose("start", "worker")
    restarted = time.monotonic()
    done = wait(api, bid)
    check(done["progress"]["finished"], f"batch finished after restart: {done['status']}")
    time.sleep(3)
    after = api.get("/monitoring").json()
    still = [a for a in alerts(after, "operations") if a["level"] == "critical"]
    check(not still, "critical alerts cleared once the job was recovered")
    remembered = [a for a in alerts(after, "operations") if "died" in a["message"]]
    check(bool(remembered), "the worker death stays on record as a warning (24 h window)")
    return {
        "batch_id": bid,
        "photos": len(files),
        "killed_at_scored": p["images_scored"],
        "worker_deaths_on_record_before": deaths_before,
        "alerts_fired": fired,
        "seconds_to_alert": seen,
        "recovered_seconds_after_restart": round(time.monotonic() - restarted, 1),
        "batch_status": done["status"],
        "operational_alerts_after_recovery": alerts(after, "operations"),
        "workers_after": after["operations"]["workers"],
    }


def changed_inputs(api: httpx.Client, args: argparse.Namespace, run: str) -> dict[str, Any]:
    print("Part 2: a batch with substantially changed inputs")
    meta = json.loads((args.camera_dir / "metadata.json").read_text())["files"]
    by_seq: dict[str, list[str]] = {}
    for name, m in sorted(meta.items()):
        if m["camera_id"] == args.camera:
            by_seq.setdefault(m["sequence_id"], []).append(name)
    seqs = sorted(by_seq)
    random.Random(args.seed).shuffle(seqs)
    names = sorted(n for s in seqs[: args.sequences] for n in by_seq[s])
    files = [(n, (args.camera_dir / "images" / n).read_bytes()) for n in names]
    file_meta = {"files": {n: meta[n] for n in names}}

    def camera_view() -> tuple[dict[str, Any], list[dict[str, Any]]]:
        m = api.get("/monitoring").json()
        return m["behavior"]["cameras"][args.camera], [
            a for a in alerts(m, "behavior") if a["camera"] == args.camera
        ]

    out: dict[str, Any] = {"camera": args.camera, "photos": len(files), "sequences": args.sequences}
    for name, payload in (
        ("control", [(n, reencode(d, f"control-{run}")) for n, d in files]),
        ("degraded", [(n, degrade(d, f"degraded-{run}")) for n, d in files]),
    ):
        bid = post(api, payload, file_meta, f"monitoring-{name}-{run}")
        wait(api, bid)
        cam, fired = camera_view()
        lb = cam["latest_batch"]
        check(lb["batch_id"] == bid and lb["comparison"] is not None, f"{name} batch compared")
        c = lb["comparison"]
        print(
            f"  {name}: label PSI {c['label_psi']:.3f}, confidence PSI {c['confidence_psi']:.3f}, "
            f"mean confidence {c['mean_confidence_change']:+.3f}, "
            f"sharpness {c['blur_change']:+.0%}; {len(fired)} behavior alert(s)"
        )
        out[name] = {"batch_id": bid, "latest": lb["latest"], "comparison": c, "alerts": fired}
    check(not out["control"]["alerts"], "the control batch raised no behavior alert")
    check(bool(out["degraded"]["alerts"]), "the degraded batch changed the behavior indicators")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--drill-dir", type=Path, default=Path("data/samples/dev-1000"))
    parser.add_argument("--camera-dir", type=Path, default=Path("data/samples/deploy-cams-90-125"))
    parser.add_argument("--camera", default="cct-90")
    parser.add_argument("--sequences", type=int, default=40)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--skip-drill", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("reports/monitoring/acceptance.json"))
    args = parser.parse_args()
    api = httpx.Client(base_url=args.url, timeout=120, headers=auth_headers())
    run = time.strftime("%Y%m%dT%H%M%S")
    report: dict[str, Any] = {
        "run": run,
        "release": api.get("/version").json()["active_release"]["id"],
    }
    if not args.skip_drill:
        report["worker_failure"] = worker_failure(api, args, run)
    report["changed_inputs"] = changed_inputs(api, args, run)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
