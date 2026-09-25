"""Stage 13: final evaluation of the frozen release on the locked final test.

Applies configs/experiments/final_evaluation.yaml, which was committed before
any number here was computed:

1. verifies every frozen artifact by hash;
2. re-runs the Stage 10 final test in a scratch copy and requires it to
   reproduce the recorded results exactly (the committed record is untouched);
3. rebuilds each final-test event with the same scoring and policy code,
   checks that its decisions equal the recorded ones and that the deployed
   policy (conservative/v2) decides every event exactly as v1 did;
4. computes the capstone metrics with camera-night and camera cluster
   bootstraps (events, never frames, are the unit), per camera;
5. writes the error gallery by a fixed selection rule.

Nothing is chosen from these results.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import sqlite3
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from wildinbox.class_map import EMPTY_CLASS
from wildinbox.policy import audit
from wildinbox.policy.conservative import POLICY_NAME, POLICY_V2, Frame, decide

AUTOMATIC = ("likely_empty", "species_identified")
ANIMAL_ROLES = ("supported_species", "unsupported_animal", "mixed_species")


class FinalEvaluationError(RuntimeError):
    pass


# ------------------------------------------------------------------ records


@dataclass
class EventRecord:
    event_id: str
    camera_id: str
    night: bool
    camera_night: str
    role: str
    label: str | None
    suggested: str | None
    confidence: float | None
    dispositions: dict[str, str]  # policy name -> disposition
    audited: dict[str, bool] = field(default_factory=dict)
    # (source_id, calibrated probabilities) per frame, for the gallery only.
    frames: list[tuple[str, dict[str, float]]] = field(default_factory=list, repr=False)

    @property
    def is_animal(self) -> bool:
        return self.role in ANIMAL_ROLES


def night_key(camera_id: str, captured_at: str | None) -> str:
    """Camera x night: a night runs noon to noon, so one night's captures
    either side of midnight share a cluster. Unknown times cluster per camera."""
    if not captured_at:
        return f"{camera_id}|unknown"
    ts = datetime.fromisoformat(captured_at)
    return f"{camera_id}|{(ts - timedelta(hours=12)).date().isoformat()}"


# ------------------------------------------------------------------ metrics


@dataclass(frozen=True)
class Ratio:
    """A metric as per-event numerator and denominator indicators, so any
    resample of clusters recomputes it from summed counts."""

    num: np.ndarray
    den: np.ndarray
    complement: bool = False  # report 1 - num/den (review reduction)

    def value(self, w: np.ndarray | None = None) -> float | None:
        n = float(self.num @ w) if w is not None else float(self.num.sum())
        d = float(self.den @ w) if w is not None else float(self.den.sum())
        if d == 0:
            return None
        return 1 - n / d if self.complement else n / d


def ratios(records: Sequence[EventRecord], policy: str) -> dict[str, Ratio]:
    disp = np.array([r.dispositions[policy] for r in records])
    role = np.array([r.role for r in records])
    animal = np.isin(role, ANIMAL_ROLES)
    kept = disp != "likely_empty"
    accepted = disp == "species_identified"
    correct = np.array([r.suggested == r.label for r in records])
    automatic = np.isin(disp, AUTOMATIC)
    audited = np.array([r.audited.get(policy, False) for r in records])
    reviewed = (disp == "needs_review") | (automatic & audited)
    ones = np.ones(len(records))
    unsupported = role == "unsupported_animal"
    f = np.asarray
    out = {
        "animal_event_retention": Ratio(f(animal & kept, float), f(animal, float)),
        "accepted_species_precision": Ratio(f(accepted & correct, float), f(accepted, float)),
        "automatic_coverage": Ratio(f(automatic, float), ones),
        "review_reduction": Ratio(f(reviewed, float), ones, complement=True),
        "unsupported_false_acceptance": Ratio(
            f(unsupported & accepted, float), f(unsupported, float)
        ),
    }
    for group in ANIMAL_ROLES:
        g = role == group
        out[f"retention_{group}"] = Ratio(f(g & kept, float), f(g, float))
    return out


def cluster_bootstrap(
    ratio: Ratio, clusters: Sequence[str], resamples: int, seed: int, level: float
) -> tuple[float, float] | None:
    """Percentile interval from resampling whole clusters with replacement."""
    names, idx = np.unique(np.asarray(clusters), return_inverse=True)
    k = len(names)
    num = np.bincount(idx, weights=ratio.num, minlength=k)
    den = np.bincount(idx, weights=ratio.den, minlength=k)
    if den.sum() == 0:
        return None
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, k, size=(resamples, k))
    counts = np.apply_along_axis(np.bincount, 1, draws, minlength=k)
    n, d = counts @ num, counts @ den
    ok = d > 0
    vals = n[ok] / d[ok]
    if ratio.complement:
        vals = 1 - vals
    a = (1 - level) / 2
    lo, hi = np.quantile(vals, [a, 1 - a])
    return float(lo), float(hi)


def wilson(successes: float, n: float, level: float = 0.95) -> tuple[float, float] | None:
    if n == 0:
        return None
    from statistics import NormalDist

    z = NormalDist().inv_cdf(1 - (1 - level) / 2)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def summarize(
    records: Sequence[EventRecord], policy: str, unc: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    nights = [r.camera_night for r in records]
    cams = [r.camera_id for r in records]
    for name, ratio in ratios(records, policy).items():
        num, den = float(ratio.num.sum()), float(ratio.den.sum())
        naive = wilson(num, den, unc["level"])
        if naive and ratio.complement:
            naive = (1 - naive[1], 1 - naive[0])
        out[name] = {
            "value": ratio.value(),
            "numerator": int(den - num) if ratio.complement else int(num),
            "denominator": int(den),
            "ci_camera_night": cluster_bootstrap(
                ratio, nights, unc["resamples"], unc["seed"], unc["level"]
            ),
            "ci_camera": cluster_bootstrap(
                ratio, cams, unc["resamples"], unc["seed"], unc["level"]
            ),
            "ci_naive_events": naive,
            "clusters": {"camera_nights": len(set(nights)), "cameras": len(set(cams))},
        }
    return out


def per_camera(records: Sequence[EventRecord], policy: str) -> dict[str, dict[str, Any]]:
    by_cam: dict[str, list[EventRecord]] = defaultdict(list)
    for r in records:
        by_cam[r.camera_id].append(r)
    out = {}
    for cam in sorted(by_cam, key=lambda c: (len(c), c)):
        rs = by_cam[cam]
        out[cam] = {
            "events": len(rs),
            "animal_events": sum(r.is_animal for r in rs),
            **{name: ratio.value() for name, ratio in ratios(rs, policy).items()},
        }
    return out


def spread(cams: dict[str, dict[str, Any]], metric: str) -> dict[str, Any]:
    vals = [c[metric] for c in cams.values() if c[metric] is not None]
    if not vals:
        return {"min": None, "median": None, "max": None, "cameras": 0}
    return {
        "min": min(vals),
        "median": float(np.median(vals)),
        "max": max(vals),
        "cameras": len(vals),
    }


def against_targets(
    summary: dict[str, dict[str, Any]], targets: dict[str, float]
) -> dict[str, dict[str, Any]]:
    out = {}
    for name, target in targets.items():
        s = summary[name]
        ci = s["ci_camera_night"]
        out[name] = {
            "target": target,
            "value": s["value"],
            "met": None if s["value"] is None else s["value"] >= target,
            "met_with_confidence": None if ci is None else ci[0] >= target,
        }
    return out


# ------------------------------------------------------------------ freeze


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_freeze(plan: dict[str, Any], models_dir: Path = Path("models")) -> dict[str, str]:
    fz, ft = plan["freeze"], plan["final_test"]
    checks = {
        "policy artifact (deployed, v2)": (
            Path(fz["policy_artifact"]),
            fz["policy_artifact_sha256"],
        ),
        "policy artifact (measured, v1)": (
            Path(fz["measured_policy_artifact"]),
            fz["measured_policy_artifact_sha256"],
        ),
        "E3 weights": (models_dir / "finetune-e3-deep-balanced" / "model.pt", fz["weights_sha256"]),
        "split lock": (Path("manifests/cct20-splits-v1.lock.json"), fz["split_lock_sha256"]),
        "taxonomy": (Path("configs/taxonomy/cct20.yaml"), fz["taxonomy_sha256"]),
        "grouping (datasets/events.py)": (
            Path("src/wildinbox/datasets/events.py"),
            fz["grouping_source_sha256"],
        ),
        "final-test protocol": (Path(ft["protocol"]), ft["protocol_sha256"]),
        "recorded final-test metrics": (
            Path("reports/final_test/metrics.json"),
            ft["recorded_metrics_sha256"],
        ),
        "recorded final-test decisions": (
            Path("reports/final_test/decisions.jsonl.gz"),
            ft["recorded_decisions_sha256"],
        ),
    }
    found = {name: _sha(path) for name, (path, _) in checks.items()}
    bad = [name for name, (_, want) in checks.items() if found[name] != want]
    if bad:
        raise FinalEvaluationError(f"frozen artifacts differ from the plan: {bad}")
    policy = json.loads(Path(fz["policy_artifact"]).read_text())
    meta = json.loads((models_dir / "finetune-e3-deep-balanced" / "meta.json").read_text())
    versions = {
        "released": (policy["released"]["policy_version"], fz["released_policy_version"]),
        "rule": (policy["rule"]["policy_version"], fz["rule_policy_version"]),
        "calibration": (policy["calibration"]["version"], fz["calibration_version"]),
        "preprocessing": (meta["preprocessing_version"], fz["preprocessing_version"]),
        "class map": (meta["class_map_fingerprint"], fz["class_map_fingerprint"]),
        "release": (f"{meta['name']}@{policy['artifact_version']}", fz["release_id"]),
    }
    wrong = {k: v for k, v in versions.items() if v[0] != v[1]}
    if wrong:
        raise FinalEvaluationError(f"frozen versions differ from the plan: {wrong}")
    return found


# ------------------------------------------------------------------ reproduce


def reproduce_final_test(plan: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """Re-run Stage 10 under its unchanged protocol in a scratch copy of its
    report directory; final_test.run refuses unless the results are identical."""
    from wildinbox.evaluation.final_test import run

    ft = plan["final_test"]
    recorded = Path("reports/final_test")
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp) / "final_test"
        shutil.copytree(recorded, scratch)
        start = time.perf_counter()
        out = run(Path(ft["protocol"]), config_path, scratch)
        seconds = time.perf_counter() - start

        def lines(p: Path) -> list[str]:
            with gzip.open(p, "rt") as f:
                return f.read().splitlines()

        same_decisions = lines(scratch / "decisions.jsonl.gz") == lines(
            recorded / "decisions.jsonl.gz"
        )
    if not same_decisions:
        raise FinalEvaluationError("re-run decisions differ from the recorded final test")
    recorded_results = json.loads((recorded / "metrics.json").read_text())["results"]
    return {
        "results_identical": out["results"] == recorded_results,
        "decisions_identical": same_decisions,
        "rerun_code": out["code"],
        "seconds": round(seconds, 1),
    }


# ------------------------------------------------------------------ events


def build_records(
    plan: dict[str, Any], config_path: Path, device: str = "mps"
) -> tuple[list[EventRecord], Any]:
    """Every final-test event, scored and decided by the frozen code paths."""
    from dataclasses import replace

    from wildinbox.evaluation.data import box_areas
    from wildinbox.evaluation.final_test import _policy_config, open_final_test
    from wildinbox.evaluation.predictors import FinetunedPredictor
    from wildinbox.evaluation.run import _score
    from wildinbox.inference.calibration import apply_temperature
    from wildinbox.settings import Settings
    from wildinbox.training.run import load_context

    fz = plan["freeze"]
    v2 = json.loads(Path(fz["policy_artifact"]).read_text())
    v1 = json.loads(Path(fz["measured_policy_artifact"]).read_text())
    classes = list(v2["classes"])
    ctx = load_context(config_path, Settings().data_dir)
    rows, events = open_final_test(ctx.split_dir, box_areas(ctx.inventory_db))
    e3 = FinetunedPredictor(ctx, Path("models/finetune-e3-deep-balanced"), device)
    if e3.weights_digest != fz["weights_sha256"][:12]:
        raise FinalEvaluationError("loaded E3 weights differ from the plan")
    scored = _score(e3, rows)
    raw = np.array([[s.probs[c] for c in classes] for s in scored])
    cal = apply_temperature(raw, float(v2["calibration"]["temperature"]))
    by_image = {
        s.row.source_id: replace(s, probs={c: float(v) for c, v in zip(classes, p, strict=True)})
        for s, p in zip(scored, cal, strict=True)
    }
    conn = sqlite3.connect(ctx.inventory_db)
    captured = dict(conn.execute("SELECT source_id, captured_at FROM records"))
    conn.close()

    configs = {
        ("v1", "released"): (_policy_config(v1["released"]), POLICY_NAME),
        ("v1", "rule"): (_policy_config(v1["rule"]), POLICY_NAME),
        ("v2", "released"): (_policy_config(v2["released"]), POLICY_V2),
        ("v2", "rule"): (_policy_config(v2["rule"]), POLICY_V2),
    }
    rate, seed = plan["freeze"]["audit"]["rate"], plan["freeze"]["audit"]["seed"]
    records = []
    for ev in events.values():
        frames = [(i, by_image[i]) for i in ev.image_ids if i in by_image]
        if not frames:
            continue
        policy_frames = [Frame(s.probs) for _, s in frames]
        outcomes = {k: decide(policy_frames, cfg, name) for k, (cfg, name) in configs.items()}
        released = outcomes[("v2", "released")]
        nights = [s.night for _, s in frames]
        times = sorted(t for t in (captured.get(i) for i, _ in frames) if t)
        disp = {f"{v}_{p}": o.disposition.value for (v, p), o in outcomes.items()}
        records.append(
            EventRecord(
                event_id=ev.event_id,
                camera_id=ev.camera_id,
                night=sum(nights) * 2 >= len(nights),
                camera_night=night_key(ev.camera_id, times[0] if times else None),
                role=ev.role,
                label=ev.label,
                suggested=released.label,
                confidence=released.confidence,
                dispositions={"released": disp["v2_released"], "rule": disp["v2_rule"], **disp},
                audited={
                    p: audit.selected(ev.event_id, disp[f"v2_{p}"], rate, seed)
                    for p in ("released", "rule")
                },
                frames=[(i, s.probs) for i, s in frames],
            )
        )
    return records, ctx


def check_decisions(records: Sequence[EventRecord], recorded_path: Path) -> dict[str, Any]:
    """The rebuilt events must be the recorded ones, decided the same way, and
    the deployed v2 policy must decide every event as v1 did."""
    with gzip.open(recorded_path, "rt") as f:
        recorded = {r["event_id"]: r for r in map(json.loads, f)}
    ours = {r.event_id: r for r in records}
    if set(ours) != set(recorded):
        raise FinalEvaluationError("rebuilt events differ from the recorded final-test events")
    mismatches = [
        eid
        for eid, r in ours.items()
        if (r.dispositions["v1_released"], r.dispositions["v1_rule"])
        != (recorded[eid]["released"], recorded[eid]["rule"])
        or (r.role, r.label) != (recorded[eid]["role"], recorded[eid]["label"])
    ]
    v2_vs_v1 = [
        r.event_id
        for r in records
        if r.dispositions["v2_released"] != r.dispositions["v1_released"]
        or r.dispositions["v2_rule"] != r.dispositions["v1_rule"]
    ]
    if mismatches:
        raise FinalEvaluationError(f"{len(mismatches)} events decided differently than recorded")
    if v2_vs_v1:
        raise FinalEvaluationError(f"v2 decides {len(v2_vs_v1)} events differently than v1")
    return {"events": len(records), "match_recorded": True, "v2_equals_v1": True}


# ------------------------------------------------------------------ gallery


def _order(records: Sequence[EventRecord]) -> list[EventRecord]:
    return sorted(records, key=lambda r: hashlib.sha256(r.event_id.encode()).hexdigest())


GALLERY_RULES: dict[str, Callable[[EventRecord], bool]] = {
    "animal_filtered_as_empty": lambda r: r.is_animal and r.dispositions["rule"] == "likely_empty",
    "confident_wrong_species": lambda r: (
        r.role == "supported_species"
        and r.suggested not in (None, r.label, EMPTY_CLASS)
        and (r.confidence or 0) >= 0.9
    ),
    "unsupported_as_known": lambda r: (
        r.role == "unsupported_animal"
        and r.suggested not in (None, EMPTY_CLASS)
        and (r.confidence or 0) >= 0.9
    ),
    "animal_suggested_empty_night": lambda r: (
        r.is_animal and r.night and r.suggested == EMPTY_CLASS
    ),
}


def select_gallery(
    records: Sequence[EventRecord], categories: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    out = {}
    for cat in categories:
        pool = [r for r in records if GALLERY_RULES[cat["name"]](r)]
        out[cat["name"]] = {
            "rule": cat["rule"],
            "matching_events": len(pool),
            "shown": _order(pool)[: cat["count"]],
        }
    return out


def _frame_for(r: EventRecord) -> tuple[str, float]:
    target = r.suggested or EMPTY_CLASS
    sid, probs = max(r.frames, key=lambda f: f[1].get(target, 0.0))
    return sid, probs.get(target, 0.0)


def write_gallery(
    gallery: dict[str, dict[str, Any]], ctx: Any, rows_by_id: dict[str, Any], out: Path, px: int
) -> dict[str, list[dict[str, Any]]]:
    from PIL import Image

    out.mkdir(parents=True, exist_ok=True)
    listing: dict[str, list[dict[str, Any]]] = {}
    for name, g in gallery.items():
        listing[name] = []
        for r in g["shown"]:
            sid, p = _frame_for(r)
            src = ctx.images_root / rows_by_id[sid].storage_path
            fname = f"{name}-{hashlib.sha256(r.event_id.encode()).hexdigest()[:10]}.jpg"
            with Image.open(src) as original:
                thumb = original.convert("RGB")
            thumb.thumbnail((px, px))
            thumb.save(out / fname, "JPEG", quality=82)
            listing[name].append(
                {
                    "file": fname,
                    "event_id": r.event_id,
                    "camera": r.camera_id,
                    "night": r.night,
                    "true": r.label if r.role != "mixed_species" else "mixed",
                    "role": r.role,
                    "suggested": r.suggested,
                    "confidence": None if r.confidence is None else round(r.confidence, 3),
                    "frame_probability": round(p, 3),
                    "frames": len(r.frames),
                    "released": r.dispositions["released"],
                    "rule": r.dispositions["rule"],
                }
            )
    return listing


# ------------------------------------------------------------------ run


def load_plan(path: Path) -> dict[str, Any]:
    plan: dict[str, Any] = yaml.safe_load(path.read_text())
    return plan


def run(
    plan_path: Path,
    config_path: Path,
    report_dir: Path,
    device: str = "mps",
    skip_reproduction: bool = False,
) -> dict[str, Any]:
    from wildinbox.evaluation.data import box_areas
    from wildinbox.evaluation.final_test import open_final_test
    from wildinbox.evaluation.run import _round
    from wildinbox.training.run import git_state

    plan = load_plan(plan_path)
    frozen = verify_freeze(plan)
    reproduction = (
        {"skipped": True} if skip_reproduction else reproduce_final_test(plan, config_path)
    )
    records, ctx = build_records(plan, config_path, device)
    consistency = check_decisions(records, Path("reports/final_test/decisions.jsonl.gz"))

    unc = plan["uncertainty"]
    results: dict[str, Any] = {
        "images": sum(len(r.frames) for r in records),
        "events": len(records),
        "roles": dict(Counter(r.role for r in records)),
        "camera_nights": len({r.camera_night for r in records}),
        "cameras": sorted({r.camera_id for r in records}, key=lambda c: (len(c), c)),
    }
    for policy in ("released", "rule"):
        summary = summarize(records, policy, unc)
        cams = per_camera(records, policy)
        results[policy] = {
            "dispositions": dict(Counter(r.dispositions[policy] for r in records)),
            "audited_events": sum(r.audited[policy] for r in records),
            "metrics": summary,
            "targets": against_targets(summary, plan["targets"]),
            "per_camera": cams,
            "camera_spread": {
                m: spread(cams, m)
                for m in (
                    "animal_event_retention",
                    "automatic_coverage",
                    "review_reduction",
                    "retention_supported_species",
                    "retention_unsupported_animal",
                )
            },
        }
    results["day_night_rule"] = {
        cond: summarize([r for r in records if r.night == (cond == "night")], "rule", unc)[
            "animal_event_retention"
        ]
        for cond in ("day", "night")
    }

    rows, _ = open_final_test(ctx.split_dir, box_areas(ctx.inventory_db))
    gallery = select_gallery(records, plan["gallery"]["categories"])
    listing = write_gallery(
        gallery,
        ctx,
        {r.source_id: r for r in rows},
        report_dir / "gallery",
        plan["gallery"]["thumbnail_px"],
    )
    results["gallery"] = {
        name: {"rule": g["rule"], "matching_events": g["matching_events"], "shown": listing[name]}
        for name, g in gallery.items()
    }

    out = {
        "plan": str(plan_path),
        "plan_sha256": _sha(plan_path),
        "frozen_artifacts": frozen,
        "code": git_state(),
        "reproduction": reproduction,
        "consistency": consistency,
        "results": _round(results),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "metrics.json").write_text(json.dumps(out, indent=2) + "\n")
    with gzip.open(report_dir / "events.jsonl.gz", "wt") as f:
        for r in sorted(records, key=lambda r: r.event_id):
            f.write(
                json.dumps(
                    {
                        "event_id": r.event_id,
                        "camera_id": r.camera_id,
                        "camera_night": r.camera_night,
                        "night": r.night,
                        "role": r.role,
                        "label": r.label,
                        "suggested": r.suggested,
                        "confidence": None if r.confidence is None else round(r.confidence, 6),
                        "frames": len(r.frames),
                        "released": r.dispositions["released"],
                        "rule": r.dispositions["rule"],
                        "audited_released": r.audited["released"],
                        "audited_rule": r.audited["rule"],
                    }
                )
                + "\n"
            )
    return out
