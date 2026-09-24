"""Command-line entry point: `wildinbox <command>`."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from wildinbox.config import ConfigError, WildInboxConfig, load_config
from wildinbox.schemas import EventDecision, ImageMetadata, Prediction, Review
from wildinbox.settings import Settings

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


def _data(args: argparse.Namespace) -> int:
    # Imported lazily so config/schema commands stay fast.
    from wildinbox.ingestion.download import DownloadError
    from wildinbox.ingestion.inventory import Paths, ingest
    from wildinbox.ingestion.pipeline import (
        LockMismatchError,
        check_or_write_lock,
        fetch,
        lock_payload,
    )
    from wildinbox.ingestion.report import write_report
    from wildinbox.ingestion.sources import load_source

    source = load_source(args.source)
    paths = Paths(root=Settings().data_dir, name=source.name)
    try:
        if args.data_command in ("download", "acquire"):
            fetch(source, paths, images=not args.annotations_only)
        if args.data_command in ("ingest", "acquire"):
            result = ingest(source, paths, workers=args.workers)
            lock = Path("manifests") / f"{source.name}.lock.json"
            state = check_or_write_lock(lock, lock_payload(source, result), update=args.update_lock)
            print(json.dumps(result.counts, indent=2))
            print(f"manifest {result.version} -> {result.manifest_path} (lock {state})")
            if args.data_command == "acquire":
                out = write_report(paths, result.version, Path(args.report_dir) / source.name)
                print(f"report -> {out}")
        if args.data_command == "report":
            lock = json.loads((Path("manifests") / f"{source.name}.lock.json").read_text())
            out = write_report(paths, lock["manifest_version"], Path(args.report_dir) / source.name)
            print(f"report -> {out}")
    except (DownloadError, LockMismatchError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="wildinbox")
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate-config", help="Validate one or more run config files.")
    p_val.add_argument("paths", nargs="+", type=Path)

    p_exp = sub.add_parser("export-schemas", help="Write JSON Schemas for shared contracts.")
    p_exp.add_argument("out_dir", type=Path, nargs="?", default=Path("docs/schemas"))

    p_data = sub.add_parser("data", help="Acquire and validate a public dataset.")
    data_sub = p_data.add_subparsers(dest="data_command", required=True)
    for name, help_ in [
        ("download", "Download and extract archives (resumable)."),
        ("ingest", "Validate every record and write the versioned inventory."),
        ("report", "Write the data-quality report for the current inventory."),
        ("acquire", "download + ingest + report."),
    ]:
        p = data_sub.add_parser(name, help=help_)
        p.add_argument("--source", type=Path, default=Path("configs/sources/cct20.yaml"))
        p.add_argument("--annotations-only", action="store_true", help="Skip the image archive.")
        p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
        p.add_argument(
            "--update-lock",
            action="store_true",
            help="Accept an inventory that differs from manifests/<name>.lock.json.",
        )
        p.add_argument("--report-dir", default="reports/data_quality")

    args = parser.parse_args(argv)
    if args.command == "validate-config":
        return _validate_config(args.paths)
    if args.command == "export-schemas":
        return _export_schemas(args.out_dir)
    return _data(args)


if __name__ == "__main__":
    raise SystemExit(main())
