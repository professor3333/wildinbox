from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO_ROOT / "configs" / "example.yaml"


@pytest.fixture
def example_config_dict() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(EXAMPLE_CONFIG.read_text())
    return data


@pytest.fixture
def write_config(tmp_path: Path):  # type: ignore[no-untyped-def]
    def _write(data: dict[str, Any]) -> Path:
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(data))
        return path

    return _write
