"""Baseline experiment configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from wildinbox.config import ConfigError
from wildinbox.datasets.spec import Partition


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BackboneSpec(_Strict):
    architecture: Literal["efficientnet_b0"]
    weights: Literal["IMAGENET1K_V1"]
    embedding: Literal["global_avgpool"]


class ClassifierSpec(_Strict):
    kind: Literal["logistic_regression"]
    c: float = Field(gt=0)
    class_weight: Literal["balanced"] | None
    max_iter: int = Field(gt=0)


class ExtractionSpec(_Strict):
    device: Literal["cpu", "mps", "cuda"]
    batch_size: int = Field(gt=0)
    num_workers: int = Field(ge=0)


class EvaluationSpec(_Strict):
    partitions: list[Partition] = Field(min_length=1)
    unseen_camera_partitions: list[Partition] = Field(min_length=1)
    empty_thresholds: list[float] = Field(min_length=1)
    species_thresholds: list[float] = Field(min_length=1)
    reference_empty_threshold: float = Field(gt=0, le=1)
    reference_species_threshold: float = Field(gt=0, le=1)
    small_animal_area: float = Field(gt=0, lt=1)
    gallery_size: int = Field(gt=0)

    @field_validator("partitions", "unseen_camera_partitions")
    @classmethod
    def _no_final_test(cls, v: list[Partition]) -> list[Partition]:
        if Partition.FINAL_TEST in v:
            raise ValueError("the locked final test may not be used for development evaluation")
        return v


class Tolerance(_Strict):
    macro_f1: float = Field(ge=0)
    per_class_recall: float = Field(ge=0)
    false_empty_rate: float = Field(ge=0)


class BaselineConfig(_Strict):
    name: str = Field(pattern=r"^[a-z0-9_.-]+$")
    run_config: Path
    splits: Path
    backbone: BackboneSpec
    classifier: ClassifierSpec
    extraction: ExtractionSpec
    evaluation: EvaluationSpec
    reproducibility: Tolerance


def load_baseline_config(path: str | Path) -> BaselineConfig:
    path = Path(path)
    try:
        return BaselineConfig.model_validate(yaml.safe_load(path.read_text()))
    except FileNotFoundError:
        raise ConfigError(f"{path}: file not found") from None
    except ValidationError as e:
        raise ConfigError(f"{path}: invalid baseline config\n{e}") from None
