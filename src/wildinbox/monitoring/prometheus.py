"""Prometheus text exposition of the monitoring summary (no client library)."""

from __future__ import annotations

from collections import Counter
from typing import Any


def _label(value: str) -> str:
    """Escape a label value per the Prometheus text format."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def render(data: dict[str, Any], latency: dict[str, dict[str, float]]) -> str:
    ops = data["operations"]
    lines: list[str] = []

    def metric(
        name: str, help_: str, kind: str, samples: list[tuple[dict[str, str], float]]
    ) -> None:
        lines.append(f"# HELP wildinbox_{name} {help_}")
        lines.append(f"# TYPE wildinbox_{name} {kind}")
        for labels, value in samples:
            lab = ",".join(f'{k}="{_label(v)}"' for k, v in labels.items())
            lines.append(
                f"wildinbox_{name}{{{lab}}} {value}" if lab else f"wildinbox_{name} {value}"
            )

    metric("jobs", "Jobs by status.", "gauge", [({"status": k}, v) for k, v in ops["jobs"].items()])
    metric(
        "queue_oldest_seconds",
        "Age of the oldest job waiting to run.",
        "gauge",
        [({}, ops["oldest_queued_seconds"])],
    )
    metric("jobs_retrying", "Jobs waiting for a retry.", "gauge", [({}, ops["retrying"])])
    metric(
        "stale_leases",
        "Running jobs whose worker stopped renewing its lease.",
        "gauge",
        [({}, ops["stale_leases"])],
    )
    metric(
        "failed_jobs_window",
        "Terminally failed jobs in the window.",
        "gauge",
        [({}, len(ops["failed_jobs_in_window"]))],
    )
    metric(
        "images_scored_window",
        "Images scored in the window.",
        "gauge",
        [({}, ops["images_scored"])],
    )
    metric(
        "images_scored_last_24h",
        "Images scored in the last 24 hours.",
        "gauge",
        [({}, ops["images_scored_last_24h"])],
    )
    metric(
        "processing_error_rate",
        "Failed frames per image scored in the window.",
        "gauge",
        [({}, ops["processing_error_rate"])],
    )
    metric(
        "unreadable_rate",
        "Unusable files per file uploaded in the window.",
        "gauge",
        [({}, ops["unreadable_rate"])],
    )
    metric(
        "seconds_per_1000_images",
        "Processing wall time per 1,000 images, by release.",
        "gauge",
        [
            ({"release": k}, v["seconds_per_1000_images"])
            for k, v in ops["cost_by_release"].items()
            if v["seconds_per_1000_images"] is not None
        ],
    )
    metric(
        "camera_needs_review_share",
        "Share of a camera's events that need review (label-free).",
        "gauge",
        [
            ({"camera": k}, v["needs_review_share"])
            for k, v in data["signals"].items()
            if v["needs_review_share"] is not None
        ],
    )
    metric(
        "camera_correction_rate",
        "Share of reviewed suggestions that reviewers corrected (needs reviews).",
        "gauge",
        [
            ({"camera": k}, v["correction_rate"])
            for k, v in data["accuracy"]["by_camera"].items()
            if v["correction_rate"] is not None
        ],
    )
    metric(
        "camera_review_coverage",
        "Share of a camera's events with a human review.",
        "gauge",
        [
            ({"camera": k}, v["review_coverage"])
            for k, v in data["accuracy"]["by_camera"].items()
            if v["review_coverage"] is not None
        ],
    )
    work = ops.get("workers")
    if work is not None:
        metric("workers_live", "Worker processes reporting.", "gauge", [({}, len(work["live"]))])
        metric(
            "workers_started_window",
            "Worker process starts in the restart window.",
            "gauge",
            [({}, work["starts_in_window"])],
        )
        metric(
            "workers_died_window",
            "Workers that stopped reporting without shutting down, in the restart window.",
            "gauge",
            [({}, len(work["died_in_window"]))],
        )
        metric(
            "worker_rss_bytes",
            "Resident memory of each live worker.",
            "gauge",
            [
                ({"worker": w["id"]}, round((w["rss_mb"] or w["peak_rss_mb"]) * 2**20))
                for w in work["live"]
                if (w["rss_mb"] or w["peak_rss_mb"]) is not None
            ],
        )
    api = ops.get("api")
    if api is not None:
        metric(
            "http_responses",
            "API responses since the API started, by status class.",
            "gauge",
            [
                ({"class": "4xx"}, api["client_errors"]),
                ({"class": "5xx"}, api["server_errors"]),
                ({"class": "all"}, api["responses"]),
            ],
        )
    store = ops.get("storage")
    if store is not None:
        metric(
            "stored_original_bytes",
            "Bytes of original photos stored.",
            "gauge",
            [({}, store["original_bytes"])],
        )
    latency_ = ops.get("job_latency")
    if latency_ is not None:
        metric(
            "job_upload_to_done_seconds",
            "Upload-to-done time of jobs finished in the window.",
            "gauge",
            [
                ({"quantile": q}, latency_["upload_to_done_seconds"][k])
                for q, k in (("0.50", "p50"), ("0.95", "p95"))
                if latency_["upload_to_done_seconds"][k] is not None
            ],
        )
    beh = data.get("behavior")
    if beh is not None:
        for share, help_ in (
            ("filtered_share", "Share of a camera's events filtered as likely empty."),
            ("auto_labeled_share", "Share of a camera's events labeled automatically."),
        ):
            metric(
                f"camera_{share}",
                help_,
                "gauge",
                [
                    ({"camera": k}, v[share])
                    for k, v in beh["cameras"].items()
                    if v[share] is not None
                ],
            )
    aud = data.get("audits")
    if aud is not None:
        metric(
            "camera_audit_labels",
            "Reviewed audit samples per camera (0: observed quality unknown).",
            "gauge",
            [({"camera": k}, v["audit_reviewed"]) for k, v in aud["by_camera"].items()],
        )
        metric(
            "camera_audit_false_empty",
            "Animals found by audits in automatically filtered events.",
            "gauge",
            [({"camera": k}, v["false_empty"]) for k, v in aud["by_camera"].items()],
        )
    levels = Counter(a["level"] for a in data["alerts"])
    metric(
        "alerts",
        "Active monitoring alerts by level.",
        "gauge",
        [({"level": k}, levels.get(k, 0)) for k in ("critical", "warning", "info")],
    )
    metric(
        "http_request_seconds",
        "API latency over the last requests, by route.",
        "gauge",
        [
            ({"route": r, "quantile": q}, v[f"p{q[2:]}_ms"] / 1000)
            for r, v in latency.items()
            for q in ("0.50", "0.95")
        ],
    )
    return "\n".join(lines) + "\n"
