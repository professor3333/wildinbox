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


def _dataset(args: argparse.Namespace) -> int:
    from wildinbox.datasets.build import BuildError, build, lock_payload
    from wildinbox.datasets.report import write_split_report
    from wildinbox.datasets.spec import load_split_spec, load_taxonomy
    from wildinbox.ingestion.inventory import Paths
    from wildinbox.ingestion.pipeline import LockMismatchError, check_or_write_lock

    spec = load_split_spec(args.spec)
    taxonomy = load_taxonomy(spec.taxonomy)
    data_dir = Settings().data_dir
    dataset = spec.inventory_manifest_version.split("-")[0]
    try:
        result = build(spec, taxonomy, Paths(data_dir, dataset).database, data_dir / "splits")
    except BuildError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    report = write_split_report(spec, taxonomy, result, Path(args.report_dir) / dataset)
    for c in result.checks:
        print(f"{'PASS' if c.passed else 'FAIL'}  {c.name}: {c.detail}")
    print(f"supported classes: {result.supported_classes}")
    print(f"splits {result.version} -> {result.events_path.parent}  report -> {report}")
    if not result.passed:
        print("error: leakage checks failed; lock not updated", file=sys.stderr)
        return 1
    try:
        state = check_or_write_lock(
            Path("manifests") / f"{spec.name}.lock.json",
            lock_payload(spec, result),
            update=args.update_lock,
        )
    except LockMismatchError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"lock {state}")
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

    p_ds = sub.add_parser("dataset", help="Build events, labels, and evaluation splits.")
    ds_sub = p_ds.add_subparsers(dest="dataset_command", required=True)
    p_build = ds_sub.add_parser("build", help="Build split manifests and the leakage report.")
    p_build.add_argument("--spec", type=Path, default=Path("configs/splits/cct20.yaml"))
    p_build.add_argument(
        "--update-lock",
        action="store_true",
        help="Accept splits that differ from manifests/<name>.lock.json.",
    )
    p_build.add_argument("--report-dir", default="reports/splits")

    p_base = sub.add_parser("baseline", help="Frozen-embedding baseline.")
    base_sub = p_base.add_subparsers(dest="baseline_command", required=True)
    p_train = base_sub.add_parser("train", help="Embed, fit, and save the baseline artifact.")
    p_train.add_argument("--config", type=Path, default=Path("configs/experiments/baseline.yaml"))
    p_train.add_argument("--models-dir", type=Path, default=Path("models"))
    p_train.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    p_train.add_argument("--no-cache", action="store_true", help="Recompute all embeddings.")

    p_eval = sub.add_parser("evaluate", help="Evaluate a model on development partitions.")
    p_eval.add_argument(
        "--model", type=Path, default=Path("models/baseline-frozen-effnetb0-logreg-v1")
    )
    p_eval.add_argument("--config", type=Path, default=Path("configs/experiments/baseline.yaml"))
    p_eval.add_argument("--report-dir", type=Path, default=Path("reports/baseline"))
    p_eval.add_argument(
        "--compare-to",
        type=Path,
        default=None,
        help="Reference metrics.json; fail if results differ beyond tolerance.",
    )
    p_eval.add_argument("--no-benchmark", action="store_true", help="Skip the latency benchmark.")

    p_api = sub.add_parser("api", help="Serve the HTTP API.")
    p_api.add_argument("--host", default="127.0.0.1")
    p_api.add_argument("--port", type=int, default=8000)
    sub.add_parser("worker", help="Run a batch-processing worker (Redis/RQ).")

    args = parser.parse_args(argv)
    if args.command == "baseline":
        from wildinbox.training.run import train_baseline

        out = train_baseline(
            args.config,
            Settings().data_dir,
            args.models_dir,
            device=args.device,
            use_cache=not args.no_cache,
        )
        print(f"baseline artifact -> {out}")
        return 0
    if args.command == "evaluate":
        from wildinbox.evaluation.run import evaluate_cli

        return evaluate_cli(args)
    if args.command == "api":
        import uvicorn

        from wildinbox.api.app import create_app

        uvicorn.run(create_app(), host=args.host, port=args.port)
        return 0
    if args.command == "worker":
        from wildinbox.workers.dispatch import run_worker

        run_worker(Settings())
        return 0
    if args.command == "dataset":
        return _dataset(args)
    if args.command == "validate-config":
        return _validate_config(args.paths)
    if args.command == "export-schemas":
        return _export_schemas(args.out_dir)
    return _data(args)


if __name__ == "__main__":
    raise SystemExit(main())
