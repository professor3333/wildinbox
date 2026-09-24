"""Deployment settings read from the environment (or a local .env file).

Run configuration (model, classes, thresholds) lives in configs/*.yaml; this
module only holds where services and files are, and service limits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WILDINBOX_", env_file=".env", extra="ignore")

    config_path: Path = Path("configs/example.yaml")
    data_dir: Path = Path("data")
    database_url: str = "postgresql+psycopg://wildinbox:wildinbox@localhost:5432/wildinbox"
    redis_url: str = "redis://localhost:6379/0"

    # Object storage: "s3" (any S3-compatible service) or "local" (a directory).
    object_store_backend: Literal["s3", "local"] = "s3"
    object_store_url: str = "http://localhost:9000"
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

    # Input limits. Oversized batches are rejected whole; oversized or unsupported
    # files get individual error records.
    max_files_per_batch: int = Field(default=2000, gt=0)
    max_file_bytes: int = Field(default=20 * 1024 * 1024, gt=0)
    max_batch_bytes: int = Field(default=1024 * 1024 * 1024, gt=0)
