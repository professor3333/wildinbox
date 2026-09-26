"""Command-line entry point: `wildinbox <command>`."""

from __future__ import annotations

import argparse
import json
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


def _release(args: argparse.Namespace) -> int:
    from sqlalchemy import select

    from wildinbox.inference.releases import (
        ReleaseError,
        activate,
        active_release_id,
        register_release,
    )
    from wildinbox.storage.db import session_factory
    from wildinbox.storage.models import ModelRelease
    from wildinbox.storage.objects import store_from_settings

    settings = Settings()
    with session_factory(settings.database_url)() as s:
        try:
            if args.release_command == "register":
                release, created = register_release(
                    s, store_from_settings(settings), args.model_dir, args.policy, args.note
                )
                print(f"{'registered' if created else 'already registered'}: {release.id}")
                print(f"  weights sha256 {release.weights_sha256}")
                print(f"  preprocessing {release.preprocessing_version}")
                print(f"  policy {release.policy_version}")
                if args.activate:
                    activate(s, release.id, args.note)
                    print(f"active release: {release.id}")
            elif args.release_command == "activate":
                activate(s, args.release_id, args.note)
                print(f"active release: {args.release_id}")
            else:
                active = active_release_id(s, settings)
                for r in s.scalars(select(ModelRelease).order_by(ModelRelease.created_at)):
                    mark = "*" if r.id == active else " "
                    print(f"{mark} {r.id}  {r.kind}  policy {r.policy_version}")
            s.commit()
        except ReleaseError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    from wildinbox.logs import configure

    # Read directly: a full Settings() would also validate unrelated fields.
    configure(
        os.environ.get("WILDINBOX_LOG_FORMAT", "text"),
        os.environ.get("WILDINBOX_LOG_LEVEL", "INFO"),
    )
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

    p_ft = sub.add_parser("finetune", help="Fine-tune EfficientNet-B0.")
    ft_sub = p_ft.add_subparsers(dest="finetune_command", required=True)
    p_ft_train = ft_sub.add_parser("train", help="Train a fine-tuning experiment.")
    p_ft_train.add_argument("--config", type=Path, required=True)
    p_ft_train.add_argument("--models-dir", type=Path, default=Path("models"))
    p_ft_aug = ft_sub.add_parser(
        "inspect-augmentation", help="Render augmented training samples with their boxes."
    )
    p_ft_aug.add_argument("--config", type=Path, required=True)
    p_ft_aug.add_argument("--out", type=Path, default=Path("reports/experiments/augmentation"))

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

    p_cmp = sub.add_parser("compare", help="Compare evaluated models; apply the selection rule.")
    p_cmp.add_argument("reports", type=Path, nargs="+", help="Report directories.")
    p_cmp.add_argument("--rule", type=Path, default=Path("configs/experiments/selection.yaml"))
    p_cmp.add_argument("--out", type=Path, default=None, help="Write the comparison here.")

    p_cal = sub.add_parser(
        "calibrate", help="Fit calibration and choose the operating point (pre-registered rule)."
    )
    p_cal.add_argument(
        "--rule", type=Path, default=Path("configs/experiments/operating_point_v2.yaml")
    )
    p_cal.add_argument("--config", type=Path, default=Path("configs/experiments/baseline.yaml"))
    p_cal.add_argument("--report-dir", type=Path, default=Path("reports/calibration"))
    p_cal.add_argument(
        "--deviation",
        type=Path,
        default=Path("configs/experiments/operating_point_deviation.yaml"),
        help="Recorded decision to release less automation than the rule allows (if present).",
    )

    p_unf = sub.add_parser(
        "unfamiliar", help="Evaluate the unfamiliar-input score (pre-registered rule)."
    )
    p_unf.add_argument("--rule", type=Path, default=Path("configs/experiments/unfamiliar.yaml"))
    p_unf.add_argument("--config", type=Path, default=Path("configs/experiments/baseline.yaml"))
    p_unf.add_argument("--report-dir", type=Path, default=Path("reports/unfamiliar"))

    p_rep = sub.add_parser(
        "replay", help="Reproduce every saved event decision from predictions + policy."
    )
    p_rep.add_argument("--policy", type=Path, default=Path("reports/calibration/policy.json"))
    p_rep.add_argument(
        "--decisions", type=Path, default=Path("reports/calibration/decisions.jsonl.gz")
    )

    p_rel = sub.add_parser("release", help="Register, activate, and list model releases.")
    rel_sub = p_rel.add_subparsers(dest="release_command", required=True)
    p_reg = rel_sub.add_parser("register", help="Register an immutable release.")
    p_reg.add_argument("--model-dir", type=Path, default=Path("models/finetune-e3-deep-balanced"))
    p_reg.add_argument("--policy", type=Path, default=Path("reports/calibration/policy.json"))
    p_reg.add_argument("--note", default=None)
    p_reg.add_argument("--activate", action="store_true", help="Also make it the default.")
    p_act = rel_sub.add_parser("activate", help="Make a release the default for new batches.")
    p_act.add_argument("release_id")
    p_act.add_argument("--note", default=None)
    rel_sub.add_parser("list", help="List releases and the active one.")

    p_jobs = sub.add_parser("jobs", help="Job maintenance.")
    jobs_sub = p_jobs.add_subparsers(dest="jobs_command", required=True)
    jobs_sub.add_parser("recover", help="Requeue stale jobs and dispatch due ones now.")

    p_ft = sub.add_parser(
        "final-test", help="Open the locked final test once, under the pre-registered protocol."
    )
    p_ft.add_argument("--protocol", type=Path, default=Path("configs/experiments/final_test.yaml"))
    p_ft.add_argument("--config", type=Path, default=Path("configs/experiments/baseline.yaml"))
    p_ft.add_argument("--report-dir", type=Path, default=Path("reports/final_test"))

    p_fe = sub.add_parser(
        "final-evaluation",
        help="Stage 13: frozen release on the locked final test (pre-registered plan).",
    )
    p_fe.add_argument(
        "--plan", type=Path, default=Path("configs/experiments/final_evaluation.yaml")
    )
    p_fe.add_argument("--config", type=Path, default=Path("configs/experiments/baseline.yaml"))
    p_fe.add_argument("--report-dir", type=Path, default=Path("reports/final_evaluation"))
    p_fe.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    p_fe.add_argument(
        "--skip-reproduction",
        action="store_true",
        help="Do not re-run the Stage 10 final test first (it takes several minutes).",
    )

    p_snap = sub.add_parser("snapshot", help="Training snapshots from reviewed events.")
    snap_sub = p_snap.add_subparsers(dest="snapshot_command", required=True)
    p_sb = snap_sub.add_parser(
        "build",
        help="Build a versioned snapshot from approved reviews through the API, "
        "excluding protected evaluation records.",
    )
    p_sb.add_argument(
        "--protocol", type=Path, default=Path("configs/experiments/update_cycle.yaml")
    )
    p_sb.add_argument(
        "--api-url",
        default=os.environ.get("WILDINBOX_API_URL", "http://localhost:8000"),
        help="API to read from; sends WILDINBOX_TOKEN as a bearer token when set.",
    )
    p_sb.add_argument("--out", type=Path, default=Path("data/snapshots"))

    p_upd = sub.add_parser(
        "update", help="Update cycle: compare a candidate with the deployed model."
    )
    upd_sub = p_upd.add_subparsers(dest="update_command", required=True)
    p_gate = upd_sub.add_parser("gate", help="Apply the pre-registered promotion gate.")
    p_gate.add_argument(
        "--protocol", type=Path, default=Path("configs/experiments/update_cycle.yaml")
    )
    p_gate.add_argument("--candidate", type=Path, default=Path("models/finetune-e3-update1"))
    p_gate.add_argument("--config", type=Path, default=Path("configs/experiments/baseline.yaml"))
    p_gate.add_argument(
        "--report-dir", type=Path, default=Path("reports/update/finetune-e3-update1")
    )

    p_mon = sub.add_parser("monitoring", help="Monitoring maintenance.")
    mon_sub = p_mon.add_subparsers(dest="monitoring_command", required=True)
    mon_sub.add_parser("backfill-quality", help="Record image quality for older images.")
    mon_sub.add_parser("summary", help="Print the monitoring summary as JSON.")

    p_study = sub.add_parser("study", help="Timed review study (organizer).")
    study_sub = p_study.add_subparsers(dest="study_command", required=True)
    p_sp = study_sub.add_parser("plan", help="Create a study plan from uploaded batches.")
    p_sp.add_argument("--name", required=True)
    p_sp.add_argument("--batch-id", action="append", required=True)
    p_sp.add_argument("--truth", type=Path, required=True, help="truth.csv of the study batch")
    p_sp.add_argument(
        "--api-url", default=os.environ.get("WILDINBOX_API_URL", "http://localhost:8000")
    )
    p_sa = study_sub.add_parser("analyze", help="Apply the pre-registered analysis.")
    p_sa.add_argument("--plan-id", required=True)
    p_sa.add_argument("--out", type=Path, required=True)
    p_sa.add_argument(
        "--api-url", default=os.environ.get("WILDINBOX_API_URL", "http://localhost:8000")
    )

    p_pol = sub.add_parser("policy", help="Decision-policy artifacts.")
    pol_sub = p_pol.add_subparsers(dest="policy_command", required=True)
    p_up = pol_sub.add_parser("upgrade", help="Same calibration and thresholds, newer policy.")
    p_up.add_argument(
        "--from", dest="source", type=Path, default=Path("reports/calibration/policy.json")
    )
    p_up.add_argument(
        "--decisions", type=Path, default=Path("reports/calibration/decisions.jsonl.gz")
    )
    p_up.add_argument("--to", dest="policy_name", default="conservative/v2")
    p_up.add_argument("--out", type=Path, required=True)

    p_tok = sub.add_parser("token", help="API access tokens.")
    tok_sub = p_tok.add_subparsers(dest="token_command", required=True)
    p_new = tok_sub.add_parser(
        "new", help="Create tokens; prints each token and the WILDINBOX_API_TOKENS line."
    )
    p_new.add_argument("names", nargs="+", help="Principal names recorded in logs, e.g. alice.")

    p_api = sub.add_parser("api", help="Serve the HTTP API.")
    p_api.add_argument("--host", default="127.0.0.1")
    p_api.add_argument("--port", type=int, default=8000)
    sub.add_parser("worker", help="Run a batch-processing worker (Redis/RQ).")
    p_ui = sub.add_parser("ui", help="Serve the review interface (Streamlit).")
    p_ui.add_argument("--host", default="127.0.0.1")
    p_ui.add_argument("--port", type=int, default=8501)
    p_ui.add_argument(
        "--api-url", default=os.environ.get("WILDINBOX_API_URL", "http://localhost:8000")
    )

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
    if args.command == "finetune":
        from wildinbox.training import finetune

        if args.finetune_command == "train":
            out = finetune.train(args.config, Settings().data_dir, args.models_dir)
            print(f"fine-tuned model -> {out}")
        else:
            print(finetune.inspect_augmentation(args.config, Settings().data_dir, args.out))
        return 0
    if args.command == "evaluate":
        from wildinbox.evaluation.run import evaluate_cli

        return evaluate_cli(args)
    if args.command == "compare":
        from wildinbox.evaluation.compare import compare

        text, _ = compare(args.reports, args.rule)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text)
        print(text)
        return 0
    if args.command == "calibrate":
        from wildinbox.evaluation.calibration import run

        r = run(args.rule, args.config, args.report_dir, args.deviation)
        op = r["operating_point"]
        print(
            f"temperature {r['calibration']['temperature']:.3f}; rule: empty filter "
            f"{op['empty_filter'] or 'disabled'}, species acceptance "
            f"{op['species_accept'] or 'disabled'}; released: filtering "
            f"{'on' if op['auto_filter_enabled'] else 'off'}, acceptance "
            f"{'on' if op['auto_accept_enabled'] else 'off'}"
        )
        print(f"report -> {args.report_dir / 'README.md'}")
        return 0
    if args.command == "unfamiliar":
        from wildinbox.evaluation.unfamiliar import run as run_unfamiliar

        u = run_unfamiliar(args.rule, args.config, args.report_dir)
        print(
            f"distance score {'adopted' if u['adopted'] else 'not adopted'} "
            f"(detection gain {u['detection_gain_on_fit']:+.3f} on {u['artifact']['method']})"
        )
        print(f"report -> {args.report_dir / 'README.md'}")
        return 0
    if args.command == "replay":
        from wildinbox.policy.replay import replay

        replayed = replay(args.decisions, json.loads(args.policy.read_text()))
        for name, counts in replayed["dispositions"].items():
            print(f"{name}: {counts}")
        for problem in replayed["problems"][:20]:
            print(f"MISMATCH {problem}", file=sys.stderr)
        print(f"{replayed['events']} events replayed, {len(replayed['problems'])} mismatches")
        return 1 if replayed["problems"] else 0
    if args.command == "final-test":
        from wildinbox.evaluation.final_test import FinalTestError
        from wildinbox.evaluation.final_test import run as run_final_test

        try:
            measured = run_final_test(args.protocol, args.config, args.report_dir)
        except FinalTestError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        e3 = measured["results"]["image_level"]["e3"]
        print(
            f"final test: E3 macro-F1 {e3['macro_f1']:.3f}; report -> {args.report_dir}/README.md"
        )
        return 0
    if args.command == "final-evaluation":
        from wildinbox.evaluation.final_eval import FinalEvaluationError
        from wildinbox.evaluation.final_eval import run as run_final_eval
        from wildinbox.evaluation.final_eval_report import write_report

        try:
            evaluation = run_final_eval(
                args.plan, args.config, args.report_dir, args.device, args.skip_reproduction
            )
        except FinalEvaluationError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        write_report(args.report_dir, evaluation)
        for policy in ("released", "rule"):
            for name, t in evaluation["results"][policy]["targets"].items():
                v = t["value"]
                shown = "undefined" if v is None else f"{v:.4f}"
                print(f"{policy:8s} {name}: {shown} (target {t['target']}, met {t['met']})")
        print(f"report -> {args.report_dir / 'README.md'}")
        return 0
    if args.command == "snapshot":
        from wildinbox.training.snapshot import SnapshotAPIError, build

        try:
            out = build(args.protocol, args.api_url, args.out)
        except SnapshotAPIError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        summary = json.loads((out / "snapshot.json").read_text())
        print(f"snapshot {summary['version']} -> {out}")
        print(f"  train: {summary['train']}")
        print(f"  holdout: {summary['holdout']}; cutoffs {summary['cutoffs']}")
        print(
            f"  approved reviewers: {summary['approved_reviewers']} "
            f"({summary['reviews_not_approved']} reviewed events by others skipped)"
        )
        print(f"  protected records excluded: {summary['excluded']['by_reason'] or 'none'}")
        print(f"  provenance: {summary['provenance']['events']} events in labels.jsonl")
        return 0
    if args.command == "update":
        from wildinbox.training.gate import GateError
        from wildinbox.training.gate import run as run_gate

        try:
            verdict = run_gate(args.protocol, args.candidate, args.config, args.report_dir)
        except GateError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        for name, check in verdict["checks"].items():
            print(f"{'pass' if check['pass'] else 'FAIL'}  {name}: {check['value']}")
        decision = "PROMOTE" if verdict["promote"] else "DO NOT PROMOTE"
        print(f"{decision} {verdict['candidate_release']}")
        print(f"report -> {args.report_dir / 'README.md'}")
        return 0
    if args.command == "monitoring":
        from wildinbox.monitoring.metrics import backfill_quality, load_config, summary
        from wildinbox.storage.db import session_factory
        from wildinbox.storage.objects import store_from_settings

        settings = Settings()
        with session_factory(settings.database_url)() as s:
            if args.monitoring_command == "backfill-quality":
                n = backfill_quality(s, store_from_settings(settings))
                print(f"recorded quality for {n} image(s)")
            else:
                data = summary(s, settings.lease_seconds, load_config(settings.monitoring_config))
                print(json.dumps(data, indent=2, default=str))
        return 0
    if args.command == "policy":
        from wildinbox.policy.upgrade import upgrade

        up = upgrade(args.source, args.decisions, args.policy_name, args.out)
        print(f"{args.policy_name} artifact {up['artifact_version']} -> {args.out / 'policy.json'}")
        print(f"  decisions whose outcome or reasons changed: {up['decisions_changed']}")
        return 0
    if args.command == "study":
        from wildinbox.study.cli import create_plan, write_analysis

        if args.study_command == "plan":
            plan_id = create_plan(args.api_url, args.name, args.batch_id, args.truth)
            print(f"study plan {plan_id}: participants join it on the review UI's Study page")
        else:
            result = write_analysis(args.api_url, args.plan_id, args.out)
            print(result["summary"]["verdict"])
            print(f"report -> {args.out / 'README.md'}")
        return 0
    if args.command == "release":
        return _release(args)
    if args.command == "jobs":
        from wildinbox.storage.db import session_factory
        from wildinbox.storage.objects import store_from_settings
        from wildinbox.workers.dispatch import dispatcher_from_settings
        from wildinbox.workers.process import recover_stale

        settings = Settings()
        store = store_from_settings(settings)
        ids = recover_stale(
            session_factory(settings.database_url),
            dispatcher_from_settings(settings, store),
            settings,
        )
        print(f"dispatched {len(ids)} due job(s)")
        return 0
    if args.command == "token":
        from wildinbox.api.auth import new_token, token_hash

        if len(set(args.names)) != len(args.names):
            print("error: names must be distinct", file=sys.stderr)
            return 2
        hashes = {}
        for name in args.names:
            token = new_token()
            hashes[name] = token_hash(token)
            print(f"token for {name} (give it to them; it is not stored anywhere):")
            print(f"  {token}")
        print("the deployment's env line (hashes only; replaces any earlier line):")
        print(f"WILDINBOX_API_TOKENS='{json.dumps(hashes)}'")
        return 0
    if args.command == "api":
        import uvicorn

        from wildinbox.api.app import create_app

        # Logging is already configured; uvicorn's own config would replace it.
        uvicorn.run(create_app(), host=args.host, port=args.port, log_config=None, access_log=False)
        return 0
    if args.command == "ui":
        import subprocess

        app = Path(__file__).parent / "ui" / "app.py"
        env = {**os.environ, "WILDINBOX_API_URL": args.api_url}
        cmd = [sys.executable, "-m", "streamlit", "run", str(app)]
        cmd += ["--server.address", args.host, "--server.port", str(args.port)]
        cmd += ["--server.headless", "true", "--server.maxUploadSize", "1024"]
        cmd += ["--browser.gatherUsageStats", "false", "--client.toolbarMode", "minimal"]
        return subprocess.call(cmd, env=env)
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
