from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from wildinbox.cli import main
from wildinbox.config import ConfigError, load_config

from .conftest import EXAMPLE_CONFIG


def test_example_config_is_valid() -> None:
    cfg = load_config(EXAMPLE_CONFIG)
    assert cfg.classes[0] == "empty"
    assert cfg.thresholds.auto_filter_enabled is False
    assert cfg.thresholds.auto_accept_enabled is False


def _with(base: dict[str, Any], dotted: str, value: Any) -> dict[str, Any]:
    data = copy.deepcopy(base)
    *parents, leaf = dotted.split(".")
    node = data
    for p in parents:
        node = node[p]
    node[leaf] = value
    return data


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("classes", ["opossum", "raccoon"], "must include 'empty'"),
        ("classes", ["empty", "raccoon", "raccoon"], "duplicate class names: ['raccoon']"),
        ("classes", ["empty", "Raccoon"], "lowercase snake_case"),
        ("thresholds.empty_filter", 1.5, "thresholds.empty_filter"),
        ("thresholds.species_accept", 0, "thresholds.species_accept"),
        ("preprocessing.crop_size", 300, "crop_size (300) cannot exceed resize_size (256)"),
        ("preprocessing.std", [0.2, 0.0, 0.2], "every std value must be > 0"),
        ("model.architecture", "resnet50", "model.architecture"),
        ("seed", -1, "seed"),
    ],
)
def test_invalid_values_name_the_field(
    example_config_dict: dict[str, Any], write_config: Any, field: str, value: Any, expected: str
) -> None:
    path = write_config(_with(example_config_dict, field, value))
    with pytest.raises(ConfigError) as exc:
        load_config(path)
    assert expected in str(exc.value)
    assert str(path) in str(exc.value)


def test_typo_in_field_name_is_rejected(
    example_config_dict: dict[str, Any], write_config: Any
) -> None:
    data = copy.deepcopy(example_config_dict)
    data["thresholds"]["auto_filter_enable"] = True  # typo of auto_filter_enabled
    with pytest.raises(ConfigError, match=r"thresholds\.auto_filter_enable: Extra inputs"):
        load_config(write_config(data))


def test_missing_section_is_reported(
    example_config_dict: dict[str, Any], write_config: Any
) -> None:
    data = copy.deepcopy(example_config_dict)
    del data["preprocessing"]
    with pytest.raises(ConfigError, match=r"preprocessing: Field required"):
        load_config(write_config(data))


def test_missing_file_and_bad_yaml(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="file not found"):
        load_config(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("classes: [empty\n")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(bad)


def test_cli_exit_codes(
    example_config_dict: dict[str, Any], write_config: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["validate-config", str(EXAMPLE_CONFIG)]) == 0
    bad = write_config(_with(example_config_dict, "classes", ["raccoon", "coyote"]))
    assert main(["validate-config", str(bad)]) == 1
    assert "must include 'empty'" in capsys.readouterr().err
