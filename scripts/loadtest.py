"""Stage 12 load test against a deployment (staging over an SSH tunnel, or local).

    uv run python scripts/loadtest.py --label workers-1 \\
        --api http://localhost:8000 --token-env WILDINBOX_TOKEN \\
        --ssh "ssh wildinbox-staging" --stack "cd /opt/wildinbox && deploy/staging/wi" \\
        --scenarios small,thousand,invalid,concurrent,restart \\
        --out reports/staging/workers-1.json

Scenarios (each uploads files no earlier run has seen, so duplicates never
short-circuit processing; see `variant`):

- small       24 clean images
- thousand    1,000 clean images (the throughput target)
- invalid     20 clean images plus corrupt, truncated, empty, non-image,
              unsupported-format, and oversized files
- concurrent  two 1,000-image batches uploaded at the same time
- restart     1,000 images; `restart worker` once ~40% are scored

For every batch it records upload transfer time (the POST, client side)
separately from processing (the job's server timestamps: queued -> started
-> finished), checks directly in PostgreSQL that no image was scored twice and
every event has exactly one decision, and samples container memory on the VM.
Afterwards it measures metadata API latency (client side, over whatever link
--api goes through) and records the server-side latency the API reports.
Machine specifications and the served release are recorded with the results.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import statistics
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

SCENARIOS = ("small", "thousand", "invalid", "concurrent", "restart")


# ------------------------------------------------------------------ files


def variant(data: bytes, tag: str) -> bytes:
    """The same JPEG with a comment segment right after the start-of-image
    marker: new bytes (so a new SHA-256 and no duplicate detection), identical
    decoded pixels (so the model sees exactly the original image)."""
    if data[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG")
    payload = f"wildinbox-loadtest {tag}".encode()
    segment = b"\xff\xfe" + (len(payload) + 2).to_bytes(2, "big") + payload
    return data[:2] + segment + data[2:]


def clean_files(images: list[Path], n: int, tag: str) -> list[tuple[str, bytes]]:
    return [(p.name, variant(p.read_bytes(), tag)) for p in images[:n]]


def invalid_files(images: list[Path], tag: str, max_file_bytes: int) -> list[tuple[str, bytes]]:
    real = variant(images[0].read_bytes(), tag + "-t")
    rng = random.Random(tag)
    return [
        ("corrupt.jpg", b"\xff\xd8\xff\xe0" + tag.encode() + b" not image data" * 40),
        ("truncated.jpg", real[: len(real) // 3]),
        ("empty.jpg", b""),
        ("notes.txt", f"field notes {tag}\n".encode()),
        ("animation.gif", b"GIF89a" + tag.encode() + b"\x00" * 64),
        ("huge.jpg", b"\xff\xd8" + rng.randbytes(max_file_bytes + 1024)),
    ]


# ------------------------------------------------------------------ the VM


class Vm:
    """Runs commands where the stack runs: over SSH, or locally without --ssh."""

    def __init__(self, ssh: str | None, stack: str) -> None:
        self.ssh, self.stack = ssh, stack

    def _argv(self, cmd: str) -> list[str]:
        return [*shlex.split(self.ssh), cmd] if self.ssh else ["bash", "-c", cmd]

    def run(self, cmd: str, timeout: float = 120) -> str:
        out = subprocess.run(
            self._argv(cmd), capture_output=True, text=True, timeout=timeout, check=True
        )
        return out.stdout

    def compose(self, args: str, timeout: float = 300) -> str:
        return self.run(f"{self.stack} {args}", timeout)

    def sql(self, query: str) -> list[list[str]]:
        out = self.compose(
            "exec -T postgres psql -U wildinbox -d wildinbox -At -F '|' -c " + shlex.quote(query)
        )
        return [line.split("|") for line in out.strip().splitlines() if line]

    def machine(self) -> dict[str, Any]:
        script = r"""
t=$(curl -s -m 2 -X PUT http://169.254.169.254/latest/api/token \
      -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' || true)
md() { curl -s -m 2 -H "X-aws-ec2-metadata-token: $t" \
      http://169.254.169.254/latest/meta-data/$1 || true; }
echo "instance_type=$(md instance-type)"
echo "region=$(md placement/region)"
echo "vcpus=$(nproc)"
echo "memory_bytes=$(free -b | awk '/^Mem:/ {print $2}')"
echo "cpu=$(lscpu | sed -n 's/^Model name: *//p')"
echo "kernel=$(uname -r)"
echo "os=$(. /etc/os-release && echo $PRETTY_NAME)"
echo "docker=$(docker version --format '{{.Server.Version}}')"
echo "disk=$(df -h / | awk 'NR==2 {print $2 " total, " $4 " free"}')"
"""
        info = dict(
            line.split("=", 1) for line in self.run(script).strip().splitlines() if "=" in line
        )
        info["containers"] = [
            json.loads(line)
            for line in self.run(
                "docker ps --format '{{json .}}' | jq -c '{Names, Image, Status}'"
            ).splitlines()
        ]
        return info


class MemorySampler:
    """Streams `docker stats` from the VM every ~2 s and keeps each container's peak."""

    def __init__(self, vm: Vm) -> None:
        self.peaks: dict[str, float] = {}  # whole run
        self.window: dict[str, float] = {}  # since the last `take_window`
        self.samples = 0
        cmd = "while true; do docker stats --no-stream --format '{{json .}}'; sleep 1; done"
        self.proc = subprocess.Popen(vm._argv(cmd), stdout=subprocess.PIPE, text=True)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    @staticmethod
    def _bytes(text: str) -> float:
        units = {"B": 1, "KiB": 2**10, "MiB": 2**20, "GiB": 2**30, "kB": 1e3, "MB": 1e6, "GB": 1e9}
        for unit in sorted(units, key=len, reverse=True):
            if text.endswith(unit):
                return float(text[: -len(unit)]) * units[unit]
        return float(text)

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            try:
                row = json.loads(line)
                used = self._bytes(row["MemUsage"].split("/")[0].strip())
            except (ValueError, KeyError):
                continue
            self.samples += 1
            name = row["Name"]
            self.peaks[name] = max(self.peaks.get(name, 0), used)
            self.window[name] = max(self.window.get(name, 0), used)

    def take_window(self) -> dict[str, int]:
        out = {k: round(v / 2**20) for k, v in sorted(self.window.items())}
        self.window = {}
        return out

    def stop(self) -> dict[str, Any]:
        self.proc.terminate()
        return {
            "peak_mib": {k: round(v / 2**20) for k, v in sorted(self.peaks.items())},
            "samples": self.samples,
        }


# ------------------------------------------------------------------ the API


def upload(
    client: httpx.Client, files: list[tuple[str, bytes]], metadata: str | None
) -> dict[str, Any]:
    body = sum(len(d) for _, d in files)
    t0 = time.monotonic()
    res = client.post(
        "/batches",
        files=[("files", (n, d, "application/octet-stream")) for n, d in files],
        data={"metadata": metadata} if metadata else None,
        headers={"Idempotency-Key": str(uuid.uuid4())},
        timeout=1800,
    )
    seconds = time.monotonic() - t0
    if res.status_code != 202:
        raise RuntimeError(f"upload failed: {res.status_code} {res.text[:300]}")
    out: dict[str, Any] = res.json()
    out["_upload"] = {
        "files": len(files),
        "bytes": body,
        "seconds": round(seconds, 2),
        "mbit_per_s": round(8 * body / seconds / 1e6, 1),
    }
    return out


def wait(
    client: httpx.Client,
    batch_id: str,
    timeout: float,
    on_progress: Any = None,
) -> dict[str, Any]:
    t0 = time.monotonic()
    last = -1
    while time.monotonic() - t0 < timeout:
        s: dict[str, Any] = client.get(f"/batches/{batch_id}").json()
        p = s["progress"]
        if p["images_scored"] != last:
            last = p["images_scored"]
            print(f"    {batch_id[:8]} scored {last}/{p['images_to_score']}", flush=True)
        if on_progress:
            on_progress(s)
        if p["finished"]:
            return s
        time.sleep(1)
    raise TimeoutError(f"batch {batch_id} not finished after {timeout}s")


def _ts(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def timing(summary: dict[str, Any]) -> dict[str, Any]:
    job = summary["job"]
    created, started, finished = (_ts(job[k]) for k in ("created_at", "started_at", "finished_at"))
    out: dict[str, Any] = {"attempts": job["attempts"], "worker": job["worker_id"]}
    if created and started:
        out["queued_seconds"] = round((started - created).total_seconds(), 1)
    if created and finished:
        total = (finished - created).total_seconds()
        out["processing_seconds"] = round(total, 1)
        scored = summary["progress"]["images_scored"]
        out["images_per_second"] = round(scored / total, 2) if total else None
    if started and finished:
        out["run_seconds"] = round((finished - started).total_seconds(), 1)
    return out


def integrity(vm: Vm, batch_id: str) -> dict[str, Any]:
    """Counted in PostgreSQL, not trusted from the API."""
    q = f"""
    SELECT
      (SELECT count(*) FROM images WHERE batch_id = '{batch_id}'),
      (SELECT count(*) FROM images WHERE batch_id = '{batch_id}'
         AND validation_status = 'valid'),
      (SELECT count(*) FROM predictions p JOIN images i ON i.id = p.image_id
         WHERE i.batch_id = '{batch_id}'),
      (SELECT count(DISTINCT p.image_id) FROM predictions p JOIN images i ON i.id = p.image_id
         WHERE i.batch_id = '{batch_id}'),
      (SELECT count(*) FROM events WHERE batch_id = '{batch_id}'),
      (SELECT count(*) FROM decisions d JOIN events e ON e.id = d.event_id
         WHERE e.batch_id = '{batch_id}'),
      (SELECT count(DISTINCT d.event_id) FROM decisions d JOIN events e ON e.id = d.event_id
         WHERE e.batch_id = '{batch_id}'),
      (SELECT count(*) FROM images i WHERE i.batch_id = '{batch_id}'
         AND i.validation_status = 'valid'
         AND NOT EXISTS (SELECT 1 FROM predictions p WHERE p.image_id = i.id)
         AND i.processing_error IS NULL)
    """
    (row,) = vm.sql(" ".join(q.split()))
    images, valid, preds, pred_images, events, decisions, decided, unscored = map(int, row)
    return {
        "images": images,
        "valid": valid,
        "predictions": preds,
        "duplicate_predictions": preds - pred_images,
        "valid_without_prediction": unscored,
        "events": events,
        "decisions": decisions,
        "events_without_one_decision": events - decided + (decisions - decided),
        # Images that failed at inference carry an error instead of a prediction.
        "ok": preds == pred_images and unscored == 0 and decisions == decided == events,
    }


def result(vm: Vm, batch: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    return {
        "batch_id": final["id"],
        "status": final["status"],
        "counts": final["counts"],
        "upload": batch["_upload"],
        "processing": timing(final),
        "integrity": integrity(vm, final["id"]),
        "failures": [{k: f[k] for k in ("filename", "stage", "error")} for f in final["failures"]][
            :20
        ],
    }


# ------------------------------------------------------------------ scenarios


def run_scenario(
    name: str,
    client: httpx.Client,
    vm: Vm,
    images: list[Path],
    metadata: str,
    run_tag: str,
    limits: dict[str, int],
    timeout: float,
) -> dict[str, Any]:
    tag = f"{run_tag}-{name}"
    if name == "small":
        b = upload(client, clean_files(images, 24, tag), metadata)
        return {"batches": [result(vm, b, wait(client, b["id"], timeout))]}

    if name == "thousand":
        b = upload(client, clean_files(images, 1000, tag), metadata)
        return {"batches": [result(vm, b, wait(client, b["id"], timeout))]}

    if name == "invalid":
        bad = invalid_files(images, tag, limits["max_file_bytes"])
        b = upload(client, clean_files(images, 20, tag) + bad, metadata)
        final = wait(client, b["id"], timeout)
        out = result(vm, b, final)
        errors = {f["filename"]: f["error"] for f in final["failures"]}
        out["invalid_files"] = {name: errors.get(name) for name, _ in bad}
        out["every_invalid_file_explained"] = all(errors.get(n) for n, _ in bad)
        return {"batches": [out]}

    if name == "concurrent":
        sets = [clean_files(images, 1000, f"{tag}-{k}") for k in "ab"]
        t0 = datetime.now(UTC)
        with ThreadPoolExecutor(2) as pool:
            uploaded = list(pool.map(lambda files: upload(client, files, metadata), sets))
            finals = list(pool.map(lambda b: wait(client, b["id"], timeout), uploaded))
        rows = [result(vm, b, f) for b, f in zip(uploaded, finals, strict=True)]
        created = min(_ts(f["job"]["created_at"]) for f in finals)  # type: ignore[type-var]
        done = max(_ts(f["job"]["finished_at"]) for f in finals)  # type: ignore[type-var]
        makespan = (done - created).total_seconds()  # type: ignore[operator]
        scored = sum(f["progress"]["images_scored"] for f in finals)
        return {
            "batches": rows,
            "started_at": t0.isoformat(),
            "makespan_seconds": round(makespan, 1),
            "images_per_second": round(scored / makespan, 2),
            "distinct_workers": len({r["processing"]["worker"] for r in rows}),
        }

    if name == "restart":
        b = upload(client, clean_files(images, 1000, tag), metadata)
        restarted: dict[str, Any] = {}

        def maybe_restart(s: dict[str, Any]) -> None:
            p = s["progress"]
            if (
                not restarted
                and not p["finished"]
                and p["images_scored"] >= 0.4 * p["images_to_score"]
            ):
                restarted["scored_before"] = p["images_scored"]
                restarted["events_before"] = s["counts"]["events"]
                t = time.monotonic()
                vm.compose("restart worker")
                restarted["restart_seconds"] = round(time.monotonic() - t, 1)
                print(f"    restarted worker at {p['images_scored']} scored", flush=True)

        final = wait(client, b["id"], timeout, maybe_restart)
        out = result(vm, b, final)
        out["restart"] = restarted
        return {"batches": [out]}

    raise ValueError(f"unknown scenario {name}")


# ------------------------------------------------------------------ latency


def percentiles(values: list[float]) -> dict[str, float]:
    v = sorted(values)
    return {
        "requests": len(v),
        "p50_ms": round(1000 * statistics.median(v), 1),
        "p95_ms": round(1000 * v[min(len(v) - 1, int(0.95 * len(v)))], 1),
        "max_ms": round(1000 * v[-1], 1),
    }


def latency(client: httpx.Client, batch_id: str, n: int) -> dict[str, Any]:
    events = client.get("/events", params={"batch_id": batch_id, "limit": 100}).json()["events"]
    job_id = client.get(f"/batches/{batch_id}").json()["job"]["id"]
    targets = {
        "GET /version": ("/version", None),
        "GET /batches/{id}": (f"/batches/{batch_id}", None),
        "GET /jobs/{id}": (f"/jobs/{job_id}", None),
        "GET /events (100 per page)": ("/events", {"batch_id": batch_id, "limit": 100}),
        "GET /events/{id}": (None, None),
    }
    out = {}
    for label, (path, params) in targets.items():
        for _ in range(5):  # warm-up
            client.get(path or f"/events/{events[0]['id']}", params=params)
        times = []
        for i in range(n):
            p = path or f"/events/{events[i % len(events)]['id']}"
            t = time.perf_counter()
            res = client.get(p, params=params)
            times.append(time.perf_counter() - t)
            res.raise_for_status()
        out[label] = percentiles(times)
    return out


# ------------------------------------------------------------------ main


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--label", required=True, help="e.g. workers-1")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--token-env", default="WILDINBOX_TOKEN", help="env var holding a token")
    ap.add_argument("--ssh", default=None, help="SSH command reaching the VM (omit: local)")
    ap.add_argument("--stack", default="docker compose", help="compose command on the VM")
    ap.add_argument("--batch-dir", type=Path, default=Path("data/samples/dev-1000"))
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--latency-requests", type=int, default=200)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    token = os.environ.get(args.token_env)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    client = httpx.Client(base_url=args.api, headers=headers, timeout=60)
    vm = Vm(args.ssh, args.stack)

    ready = client.get("/ready")
    if ready.status_code != 200:
        sys.exit(f"deployment not ready: {ready.text}")
    version = client.get("/version")
    version.raise_for_status()
    release = version.json()["active_release"]
    images = sorted((args.batch_dir / "images").iterdir())
    metadata = (args.batch_dir / "metadata.json").read_text()
    run_tag = f"{args.label}-{uuid.uuid4().hex[:8]}"
    limits = {"max_file_bytes": int(os.environ.get("WILDINBOX_MAX_FILE_BYTES", 20 * 2**20))}

    report: dict[str, Any] = {
        "label": args.label,
        "run_tag": run_tag,
        "started_at": datetime.now(UTC).isoformat(),
        "api": args.api,
        "release": {
            k: release[k]
            for k in ("id", "weights_sha256", "preprocessing_version", "policy_version", "is_test")
        },
        "machine": vm.machine(),
        "image_bytes": {
            "batch_dir": str(args.batch_dir),
            "files": len(images),
            "total": sum(p.stat().st_size for p in images),
            "mean": round(statistics.mean(p.stat().st_size for p in images)),
        },
        "scenarios": {},
    }
    sampler = MemorySampler(vm)
    try:
        for name in args.scenarios.split(","):
            print(f"== {name}", flush=True)
            t = time.monotonic()
            sampler.take_window()
            outcome = run_scenario(
                name, client, vm, images, metadata, run_tag, limits, args.timeout
            )
            outcome["wall_seconds"] = round(time.monotonic() - t, 1)
            outcome["memory_peak_mib"] = sampler.take_window()
            report["scenarios"][name] = outcome
            for b in outcome["batches"]:
                print(
                    f"   {b['status']}: upload {b['upload']['seconds']}s, processing "
                    f"{b['processing'].get('processing_seconds')}s, integrity "
                    f"{'ok' if b['integrity']['ok'] else 'FAILED'}",
                    flush=True,
                )
        largest = next(
            (
                report["scenarios"][s]["batches"][0]["batch_id"]
                for s in ("thousand", "restart", "concurrent")
                if s in report["scenarios"]
            ),
            None,
        )
        if largest and args.latency_requests:
            print("== metadata latency", flush=True)
            report["latency_client"] = latency(client, largest, args.latency_requests)
            report["latency_server"] = client.get("/monitoring", timeout=300).json()["operations"][
                "api_latency"
            ]
    finally:
        report["memory"] = sampler.stop()
        report["finished_at"] = datetime.now(UTC).isoformat()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"results -> {args.out}")


if __name__ == "__main__":
    main()
