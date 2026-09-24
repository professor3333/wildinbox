"""`wildinbox compare`: experiment comparison on the decisions the product needs,
and the pre-registered selection rule (configs/experiments/selection.yaml)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

UNSEEN = "unseen_cameras"


class SelectionRule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    reference: Path
    min_macro_f1_gain: float = Field(ge=0)
    max_false_empty_increase: float = Field(ge=0)
    max_cpu_latency_ratio: float = Field(gt=0)


def load_rule(path: Path) -> SelectionRule:
    return SelectionRule.model_validate(yaml.safe_load(path.read_text()))


@dataclass
class Summary:
    name: str
    report_dir: Path
    metrics: dict[str, Any]

    @property
    def unseen(self) -> dict[str, Any]:
        u: dict[str, Any] = self.metrics["groups"][UNSEEN]
        return u

    @property
    def macro_f1(self) -> float:
        return float(self.unseen["image"]["macro_f1"])

    @property
    def false_empty(self) -> int:
        return int(self.unseen["event_reference"]["false_empty"])

    def false_empty_at(self, t: float) -> int:
        return next(
            int(m["false_empty"])
            for m in self.unseen["event_empty_sweep"]
            if m["thresholds"]["empty_filter"] == t
        )

    @property
    def cpu_p50(self) -> float | None:
        lat = self.metrics.get("latency") or []
        return next((float(x["p50_ms"]) for x in lat if x["device"] == "cpu"), None)


def load_summary(report_dir: Path) -> Summary:
    metrics = json.loads((report_dir / "metrics.json").read_text())
    return Summary(metrics["model"], report_dir, metrics)


def verdict(candidate: Summary, ref: Summary, rule: SelectionRule) -> tuple[bool, list[str]]:
    reasons, ok = [], True
    gain = candidate.macro_f1 - ref.macro_f1
    if gain < rule.min_macro_f1_gain:
        ok = False
        reasons.append(f"macro-F1 gain {gain:+.3f} < {rule.min_macro_f1_gain}")
    else:
        reasons.append(f"macro-F1 gain {gain:+.3f}")
    limit = ref.false_empty * (1 + rule.max_false_empty_increase)
    if candidate.false_empty > limit:
        ok = False
        reasons.append(f"false-empty events {candidate.false_empty} > limit {limit:.0f}")
    else:
        reasons.append(f"false-empty events {candidate.false_empty} (limit {limit:.0f})")
    if candidate.cpu_p50 and ref.cpu_p50:
        ratio = candidate.cpu_p50 / ref.cpu_p50
        if ratio > rule.max_cpu_latency_ratio:
            ok = False
            reasons.append(f"CPU latency {ratio:.2f}x > {rule.max_cpu_latency_ratio}x")
        else:
            reasons.append(f"CPU latency {ratio:.2f}x")
    return ok, reasons


def _t(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _link(report_dir: Path) -> str:
    """Link from reports/experiments/README.md to a report directory."""
    parts = report_dir.resolve().parts
    rel = Path(*parts[parts.index("reports") + 1 :])
    return str(Path("..") / rel / "README.md")


def _dn(s: Summary, cond: str, key: str) -> Any:
    return s.metrics["groups"]["slices_unseen"]["day_night"][cond][key]


def compare(report_dirs: list[Path], rule_path: Path) -> tuple[str, Summary | None]:
    rule = load_rule(rule_path)
    ref = load_summary(rule.reference)
    models = [ref] + [
        load_summary(d) for d in report_dirs if d.resolve() != ref.report_dir.resolve()
    ]
    passing = []
    rows = []
    for s in models:
        u = s.unseen
        lin = s.metrics.get("lineage", {})
        species = u["event_species_sweep"]
        acc = next(m for m in species if m["thresholds"]["species_accept"] == 0.9)
        is_ref = s is ref
        ok, why = (None, ["reference"]) if is_ref else verdict(s, ref, rule)
        if ok:
            passing.append(s)
        rows.append(
            [
                f"[{s.name}]({_link(s.report_dir)})",
                f"{s.macro_f1:.3f}",
                f"{u['image']['min_species_recall']:.3f}",
                f"{100 * u['image']['animal_image_recall']:.1f}%",
                f"{s.false_empty_at(0.9)} / {s.false_empty_at(0.99)}",
                f"{acc['accepted']} @ {100 * (acc['accepted_precision'] or 0):.1f}%",
                acc["unsupported_accepted_as_known"],
                f"{s.metrics['groups']['seen_camera_diagnostic']['image']['macro_f1']:.3f}",
                f"{s.cpu_p50:.0f}" if s.cpu_p50 else "n/a",
                f"{lin.get('train_seconds', 0) / 60:.0f}" if lin.get("train_seconds") else "-",
                "reference" if is_ref else ("**pass**" if ok else "fail") + ": " + "; ".join(why),
            ]
        )
    best = max(passing, key=lambda s: (s.macro_f1, -s.false_empty), default=None)
    classes = [c for c in ref.metrics["classes"]]
    md = [
        "## Comparison (unseen development cameras)",
        "",
        _t(
            [
                "Model",
                "Macro-F1",
                "Min species recall",
                "Animal image recall",
                "False-empty events @0.9 / @0.99",
                "Accepted @0.9 (precision)",
                "Unsupported accepted",
                "Seen-camera macro-F1",
                "CPU p50 ms",
                "Train min",
                "Selection rule",
            ],
            rows,
        ),
        "",
        "Per-class recall:",
        "",
        _t(
            ["Model", *classes],
            [
                [s.name, *(f"{s.unseen['image']['per_class'][c]['recall']:.2f}" for c in classes)]
                for s in models
            ],
        ),
        "",
        "False-empty events per camera at the reference thresholds:",
        "",
        _t(
            ["Model", *sorted(ref.metrics["groups"]["slices_unseen"]["camera"], key=int)],
            [
                [
                    s.name,
                    *(
                        f"{v['false_empty']} / {v['animal_events']}"
                        for _, v in sorted(
                            s.metrics["groups"]["slices_unseen"]["camera"].items(),
                            key=lambda kv: int(kv[0]),
                        )
                    ),
                ]
                for s in models
            ],
        ),
        "",
        "Night vs day (unseen cameras):",
        "",
        _t(
            ["Model", "Day macro-F1", "Night macro-F1", "Night false-empty"],
            [
                [
                    s.name,
                    f"{_dn(s, 'day', 'macro_f1'):.3f}",
                    f"{_dn(s, 'night', 'macro_f1'):.3f}",
                    f"{_dn(s, 'night', 'false_empty')} / {_dn(s, 'night', 'animal_events')}",
                ]
                for s in models
            ],
        ),
        "",
        f"**Selected by the rule:** {best.name if best else 'none: the baseline is retained'}.",
        "",
    ]
    return "\n".join(md), best
