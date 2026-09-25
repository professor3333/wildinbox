"""README for reports/final_evaluation/ (filled in after the first run)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_report(report_dir: Path, out: dict[str, Any]) -> None:
    (report_dir / "README.md").write_text("# Final evaluation\n\nSee metrics.json.\n")
