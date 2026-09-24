"""Schemas for the taxonomy mapping and the split specification."""

from __future__ import annotations

import hashlib
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from wildinbox.class_map import EMPTY_CLASS
from wildinbox.config import ConfigError


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Kind(StrEnum):
    ANIMAL = "animal"
    EMPTY = EMPTY_CLASS
    NON_ANIMAL = "non_animal"


class Category(_Strict):
    kind: Kind


class Taxonomy(_Strict):
    name: str
    categories: dict[str, Category]

    @model_validator(mode="after")
    def _one_empty(self) -> Self:
        empties = [k for k, v in self.categories.items() if v.kind is Kind.EMPTY]
        if empties != [EMPTY_CLASS]:
            raise ValueError(
                f"exactly one category must be kind {Kind.EMPTY.value!r} and named "
                f"{EMPTY_CLASS!r}, got {empties}"
            )
        return self

    def kind_of(self, label: str) -> Kind:
        try:
            return self.categories[label].kind
        except KeyError:
            raise ConfigError(
                f"category {label!r} is not in taxonomy {self.name!r}; add it explicitly"
            ) from None


class Partition(StrEnum):
    TRAIN = "train"
    CALIBRATION = "calibration"
    POLICY_VALIDATION = "policy_validation"
    FINAL_TEST = "final_test"
    SEEN_CAMERA_DIAGNOSTIC = "seen_camera_diagnostic"


class CameraAssignment(_Strict):
    final_test: list[str] = Field(min_length=1)
    policy_validation: list[str] = Field(min_length=1)
    calibration: list[str] = Field(min_length=1)
    train: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _disjoint(self) -> Self:
        seen: dict[str, str] = {}
        for part, cams in self.model_dump().items():
            for cam in cams:
                if cam in seen:
                    raise ValueError(f"camera {cam!r} assigned to both {seen[cam]} and {part}")
                seen[cam] = part
        return self

    def partition_of(self) -> dict[str, Partition]:
        return {cam: Partition(part) for part, cams in self.model_dump().items() for cam in cams}


class DiagnosticSample(_Strict):
    fraction: float = Field(gt=0, lt=1, description="Share of training-camera sequences.")
    salt: str = Field(min_length=1, description="Changes the sample deterministically.")

    def contains(self, sequence_id: str) -> bool:
        digest = hashlib.sha256(f"{self.salt}:{sequence_id}".encode()).digest()
        return int.from_bytes(digest[:8], "big") / 2**64 < self.fraction


class SpeciesSelection(_Strict):
    min_train_events: int = Field(gt=0)
    min_train_cameras: int = Field(gt=0)
    min_events_per_camera: int = Field(gt=0)


class SplitSpec(_Strict):
    name: str = Field(pattern=r"^[a-z0-9_.-]+$")
    inventory_manifest_version: str
    taxonomy: Path
    grouping_rule: Literal["sequence_id/v1"]
    cameras: CameraAssignment
    seen_camera_diagnostic: DiagnosticSample | None = None
    species_selection: SpeciesSelection


def _load(path: Path, model: type[_Strict]) -> _Strict:
    try:
        return model.model_validate(yaml.safe_load(path.read_text()))
    except FileNotFoundError:
        raise ConfigError(f"{path}: file not found") from None
    except ValidationError as e:
        raise ConfigError(f"{path}: invalid {model.__name__}\n{e}") from None


def load_split_spec(path: str | Path) -> SplitSpec:
    spec = _load(Path(path), SplitSpec)
    assert isinstance(spec, SplitSpec)
    return spec


def load_taxonomy(path: str | Path) -> Taxonomy:
    tax = _load(Path(path), Taxonomy)
    assert isinstance(tax, Taxonomy)
    return tax
