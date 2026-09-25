"""Deployment settings read from the environment (or a local .env file).

Run configuration (model, classes, thresholds) lives in configs/*.yaml; this
module only holds where services and files are, and service limits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WILDINBOX_", env_file=".env", extra="ignore")

    config_path: Path = Path("configs/example.yaml")
    monitoring_config: Path = Path("configs/monitoring/monitoring.yaml")
    data_dir: Path = Path("data")
    database_url: str = "postgresql+psycopg://wildinbox:wildinbox@localhost:5432/wildinbox"
    redis_url: str = "redis://localhost:6379/0"

    # Object storage: "s3" (any S3-compatible service) or "local" (a directory).
    # An empty URL means AWS S3 itself; empty keys use the default AWS
    # credential chain (on the staging VM, the instance role).
    object_store_backend: Literal["s3", "local"] = "s3"
    object_store_url: str | None = "http://localhost:9000"
    object_store_bucket: str = "wildinbox"
    object_store_access_key: str | None = None
    object_store_secret_key: str | None = None
    object_store_region: str = "us-east-1"
    local_store_dir: Path = Path("data/objects")

    # How batch jobs reach workers: "rq" (Redis queue) or "inline" (same process; tests/dev).
    dispatch: Literal["rq", "inline"] = "rq"
    queue_name: str = "wildinbox"

    # Single-workspace first release.
    workspace: str = "default"
    # Model release used for new batches. "test-predictor-v0" validates plumbing only.
    active_release: str = "test-predictor-v0"
    # The release this deployment must serve. When set, /ready fails unless it
    # is the active release and its weights load; workers load it before
    # taking jobs and exit if they cannot.
    expected_release: str | None = None

    # Access. "tokens": every data endpoint needs `Authorization: Bearer <token>`;
    # `api_tokens` maps a principal name to the SHA-256 of its token (create
    # with `wildinbox token new NAME`). "disabled" is for local development only.
    auth: Literal["tokens", "disabled"] = "tokens"
    api_tokens: dict[str, str] = Field(default_factory=dict)

    # Logs: "json" (one object per line, for staging) or "text".
    log_format: Literal["json", "text"] = "text"
    log_level: str = "INFO"

    # Workers. Images are scored in bounded chunks; each chunk commit renews the
    # job's lease. A lease not renewed within `lease_seconds` is presumed dead and
    # the job is recovered. Failed attempts retry with exponential backoff until
    # the job's `max_attempts`, then fail terminally.
    inference_device: str = "cpu"
    # CPU threads PyTorch may use per worker process (unset: one per core).
    # With several workers on one machine, cores / workers avoids oversubscription.
    torch_threads: int | None = Field(default=None, gt=0)
    inference_chunk: int = Field(default=16, gt=0)
    lease_seconds: int = Field(default=120, gt=0)
    job_timeout_seconds: int = Field(default=3600, gt=0)
    retry_backoff_seconds: int = Field(default=30, ge=0)
    retry_backoff_max_seconds: int = Field(default=900, ge=0)
    recovery_interval_seconds: int = Field(default=30, gt=0)
    # Each worker process records liveness and memory this often (Monitoring).
    worker_heartbeat_seconds: int = Field(default=15, gt=0)

    # Share of automatically handled events (filtered or auto-labeled) sent to
    # the audit queue anyway, so confident mistakes are measured.
    audit_rate: float = Field(default=0.05, ge=0, le=1)
    audit_seed: str = "wildinbox-audit-v1"

    # Input limits. Oversized batches are rejected whole; oversized or unsupported
    # files get individual error records.
    max_files_per_batch: int = Field(default=2000, gt=0)
    max_file_bytes: int = Field(default=20 * 1024 * 1024, gt=0)
    max_batch_bytes: int = Field(default=1024 * 1024 * 1024, gt=0)

    @field_validator(
        "object_store_url", "object_store_access_key", "object_store_secret_key", "expected_release"
    )
    @classmethod
    def _blank_is_none(cls, v: str | None) -> str | None:
        return v or None

    @field_validator("torch_threads", mode="before")
    @classmethod
    def _blank_threads(cls, v: object) -> object:
        return None if v == "" else v

    @field_validator("api_tokens")
    @classmethod
    def _token_hashes(cls, v: dict[str, str]) -> dict[str, str]:
        bad = [name for name, h in v.items() if len(h) != 64 or not _is_hex(h)]
        if bad:
            raise ValueError(f"api_tokens values must be SHA-256 hex digests; check {bad}")
        return {name: h.lower() for name, h in v.items()}


def _is_hex(s: str) -> bool:
    try:
        int(s, 16)
    except ValueError:
        return False
    return True
