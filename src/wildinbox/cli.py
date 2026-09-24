"""Command-line entry point: `wildinbox <command>`."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from wildinbox.config import ConfigError, WildInboxConfig, load_config
from wildinbox.schemas import EventDecision, ImageMetadata, Prediction, Review

SCHEMAS: dict[str, type[BaseModel]] = {
    "config": WildInboxConfig,
    "image_metadata": ImageMetadata,
    "prediction": Prediction,
    "event_decision": EventDecision,
    "review": Review,
}


def _validate_config(paths: Sequence[Path]) -> int:
    failed = 0
    for path in paths:
        try:
            cfg = load_config(path)
        except ConfigError as e:
            print(e, file=sys.stderr)
            failed += 1
        else:
            print(f"{path}: ok ({len(cfg.classes)} classes, policy {cfg.policy_version})")
    return 1 if failed else 0


def _export_schemas(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, model in SCHEMAS.items():
        schema = json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n"
        (out_dir / f"{name}.schema.json").write_text(schema)
    print(f"wrote {len(SCHEMAS)} schemas to {out_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wildinbox")
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate-config", help="Validate one or more run config files.")
    p_val.add_argument("paths", nargs="+", type=Path)

    p_exp = sub.add_parser("export-schemas", help="Write JSON Schemas for shared contracts.")
    p_exp.add_argument("out_dir", type=Path, nargs="?", default=Path("docs/schemas"))

    args = parser.parse_args(argv)
    if args.command == "validate-config":
        return _validate_config(args.paths)
    return _export_schemas(args.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
