"""Deployment settings read from the environment (or a local .env file).

Run configuration (model, classes, thresholds) lives in configs/*.yaml; this
module only holds where services and files are.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WILDINBOX_", env_file=".env", extra="ignore")

    config_path: Path = Path("configs/example.yaml")
    data_dir: Path = Path("data")
    database_url: str = "postgresql://wildinbox:wildinbox@localhost:5432/wildinbox"
    redis_url: str = "redis://localhost:6379/0"
    object_store_url: str = "http://localhost:9000"
    object_store_bucket: str = "wildinbox"
