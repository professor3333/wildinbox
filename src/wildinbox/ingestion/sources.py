"""Pinned descriptions of public dataset sources (URLs, sizes, checksums)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from wildinbox.config import ConfigError


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Archive(_Strict):
    url: str = Field(pattern=r"^https://")
    filename: str = Field(pattern=r"^[\w.-]+$")
    size: int = Field(gt=0, description="Expected size in bytes.")
    md5: str = Field(pattern=r"^[0-9a-f]{32}$", description="Expected MD5 (hex).")


class Exclusion(_Strict):
    source_id: str
    reason: str = Field(min_length=1)


class SourceConfig(_Strict):
    name: str = Field(pattern=r"^[a-z0-9_]+$")
    images_archive: Archive
    annotations_archive: Archive
    annotation_files: list[str] = Field(
        min_length=1, description="Paths inside the extracted annotations archive."
    )
    exclusions: list[Exclusion] = Field(
        default_factory=list, description="Source records intentionally left out, with reasons."
    )


def load_source(path: str | Path) -> SourceConfig:
    path = Path(path)
    try:
        return SourceConfig.model_validate(yaml.safe_load(path.read_text()))
    except FileNotFoundError:
        raise ConfigError(f"{path}: file not found") from None
    except ValidationError as e:
        raise ConfigError(f"{path}: invalid source config\n{e}") from None
