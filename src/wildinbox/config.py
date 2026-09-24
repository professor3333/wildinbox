"""Run configuration schema: dataset version, model, seed, preprocessing,
supported classes, and decision thresholds.

Every training run and every deployed model is described by one of these.
Unknown fields are rejected so a typo cannot silently fall back to a default.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from wildinbox.class_map import EMPTY_CLASS, ClassMap

_CLASS_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class ConfigError(ValueError):
    """A configuration file is missing, unparsable, or invalid."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetConfig(_Strict):
    name: str = Field(min_length=1, description="Dataset identifier, e.g. 'cct20'.")
    manifest_version: str = Field(
        min_length=1, description="Pinned, versioned manifest the run was built from."
    )


class ModelConfig(_Strict):
    architecture: Literal["efficientnet_b0"]
    pretrained_weights: Literal["IMAGENET1K_V1"] | None = Field(
        description="torchvision weights enum member, or null to train from scratch."
    )


class PreprocessingConfig(_Strict):
    resize_size: int = Field(gt=0, description="Shorter side is resized to this many pixels.")
    crop_size: int = Field(gt=0, description="Square center crop fed to the model.")
    interpolation: Literal["bilinear", "bicubic"]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    @field_validator("std")
    @classmethod
    def _std_positive(cls, v: tuple[float, float, float]) -> tuple[float, float, float]:
        if any(s <= 0 for s in v):
            raise ValueError(f"every std value must be > 0, got {v}")
        return v

    @model_validator(mode="after")
    def _crop_fits(self) -> Self:
        if self.crop_size > self.resize_size:
            raise ValueError(
                f"crop_size ({self.crop_size}) cannot exceed resize_size ({self.resize_size})"
            )
        return self

    def fingerprint(self) -> str:
        """Short stable hash recorded with every prediction as the preprocessing version."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:12]


class ThresholdsConfig(_Strict):
    """Separate limits for filtering empties and accepting species, since the
    consequences of each mistake differ. Automation is off unless evaluation
    supports the operating point."""

    empty_filter: float = Field(
        gt=0, le=1, description="Min calibrated P(empty) every usable frame needs to filter."
    )
    species_accept: float = Field(
        gt=0, le=1, description="Min calibrated P(species) to accept a label automatically."
    )
    unfamiliar_max_distance: float | None = Field(
        default=None, gt=0, description="Max embedding distance before flagging unfamiliar input."
    )
    auto_filter_enabled: bool = False
    auto_accept_enabled: bool = False


class WildInboxConfig(_Strict):
    config_version: Literal[1]
    seed: int = Field(ge=0)
    dataset: DatasetConfig
    model: ModelConfig
    preprocessing: PreprocessingConfig
    classes: list[str] = Field(
        min_length=2,
        description=f"Model output order. Must include {EMPTY_CLASS!r} and >= 1 species.",
    )
    policy_version: str = Field(min_length=1)
    thresholds: ThresholdsConfig

    @field_validator("classes")
    @classmethod
    def _valid_classes(cls, v: list[str]) -> list[str]:
        bad = [c for c in v if not _CLASS_NAME.fullmatch(c)]
        if bad:
            raise ValueError(f"class names must be lowercase snake_case, got {bad}")
        dupes = sorted({c for c in v if v.count(c) > 1})
        if dupes:
            raise ValueError(f"duplicate class names: {dupes}")
        if EMPTY_CLASS not in v:
            raise ValueError(f"classes must include {EMPTY_CLASS!r}")
        return v

    @property
    def class_map(self) -> ClassMap:
        return ClassMap(self.classes)


def _format_errors(path: Path, err: ValidationError) -> str:
    lines = [f"{path}: invalid configuration ({err.error_count()} error(s))"]
    for e in err.errors():
        loc = ".".join(str(p) for p in e["loc"]) or "<root>"
        got = f" (got {e['input']!r})" if e["type"] != "missing" and "input" in e else ""
        lines.append(f"  - {loc}: {e['msg']}{got}")
    return "\n".join(lines)


def load_config(path: str | Path) -> WildInboxConfig:
    """Load and validate a YAML config. Raises ConfigError with readable messages."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except FileNotFoundError:
        raise ConfigError(f"{path}: file not found") from None
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: not valid YAML: {e}") from None
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    try:
        return WildInboxConfig.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(_format_errors(path, e)) from None
