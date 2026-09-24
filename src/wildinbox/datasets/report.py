"""Split report: leakage checks, partitions, taxonomy, species selection,
benchmark deviations, grouping-rule evaluation, and limitations."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from wildinbox.datasets.build import BuildResult
from wildinbox.datasets.events import Role
from wildinbox.datasets.grouping import (
    DEFAULT_GAP_SECONDS,
    GroupableImage,
    group_by_time_gap,
    time_gap_rule_id,
)
from wildinbox.datasets.spec import Partition, SplitSpec, Taxonomy

ORDER = [
    Partition.TRAIN,
    Partition.SEEN_CAMERA_DIAGNOSTIC,
    Partition.CALIBRATION,
    Partition.POLICY_VALIDATION,
    Partition.FINAL_TEST,
]
PURPOSE = {
    Partition.TRAIN: "Fit model parameters and training-derived statistics",
    Partition.SEEN_CAMERA_DIAGNOSTIC: "Diagnostic only: seen-camera vs unseen-camera comparison",
    Partition.CALIBRATION: "Fit score calibration",
    Partition.POLICY_VALIDATION: "Choose thresholds, aggregation rules, operating points",
    Partition.FINAL_TEST: "Measure the finished system once, after decisions are frozen",
}


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _camera_key(cam: str) -> tuple[int, int, str]:
    """Numeric camera ids in numeric order, then any others alphabetically."""
    return (0, int(cam), "") if cam.isdigit() else (1, 0, cam)


def _short(source_file: str) -> str:
    return Path(source_file).stem.replace("_annotations", "")


def grouping_agreement(result: BuildResult, gap: float = DEFAULT_GAP_SECONDS) -> dict[str, Any]:
    """How the upload rule (time gaps) compares with true sequence ids on CCT20."""
    seq_of: dict[str, str] = {}
    images = []
    for e in result.events:
        for f in e.frames:
            seq_of[f.source_id] = f.sequence_id
            ts = datetime.fromisoformat(f.captured_at) if f.captured_at else None
            images.append(GroupableImage(f.source_id, f.camera_id, ts))
    groups = group_by_time_gap(images, gap)
    split_seqs = {
        s for s, n in Counter(s for g in groups for s in {seq_of[i] for i in g}).items() if n > 1
    }
    seq_size = Counter(seq_of.values())
    exact = sum(
        1 for g in groups if len({seq_of[i] for i in g}) == 1 and len(g) == seq_size[seq_of[g[0]]]
    )
    merged = sum(1 for g in groups if len({seq_of[i] for i in g}) > 1)
    return {
        "rule": time_gap_rule_id(gap),
        "events": len(groups),
        "sequences": len(set(seq_of.values())),
        "exact": exact,
        "merged": merged,
        "split_sequences": len(split_seqs),
    }


def write_split_report(spec: SplitSpec, taxonomy: Taxonomy, result: BuildResult, out: Path) -> Path:
    kept = [e for e in result.events if not e.excluded_reason]
    excluded = [e for e in result.events if e.excluded_reason]
    md: list[str] = [
        f"# Split report: `{result.version}`",
        "",
        f"Inventory `{spec.inventory_manifest_version}` · grouping rule `{spec.grouping_rule}` · "
        f"spec `configs/splits/cct20.yaml` · rules [`docs/dataset.md`](../../../docs/dataset.md)",
        "",
        "## Leakage checks",
        "",
        _table(
            ["Check", "Result", "Detail"],
            [[c.name, "PASS" if c.passed else "**FAIL**", c.detail] for c in result.checks],
        ),
        "",
    ]

    # Partitions
    cams: dict[Partition, set[str]] = defaultdict(set)
    roles: dict[Partition, Counter[str]] = defaultdict(Counter)
    imgs: Counter[Partition] = Counter()
    fit: Counter[Partition] = Counter()
    for e in kept:
        assert e.partition is not None and e.role is not None
        cams[e.partition].add(e.camera_id)
        roles[e.partition][e.role.value] += 1
        imgs[e.partition] += len(e.images)
    for line in _image_fit_counts(result):
        fit[line[0]] += line[1]
    role_names = [r.value for r in Role]
    md += [
        "## Partitions",
        "",
        _table(
            ["Partition", "Permitted use", "Cameras", "Events", "Images", "Fit images"],
            [
                [
                    p.value,
                    PURPOSE[p],
                    ", ".join(sorted(cams[p], key=_camera_key)),
                    sum(roles[p].values()),
                    imgs[p],
                    fit[p],
                ]
                for p in ORDER
            ],
        ),
        "",
        "Events by role:",
        "",
        _table(
            ["Partition", *role_names],
            [[p.value, *(roles[p][r] for r in role_names)] for p in ORDER],
        ),
        "",
    ]

    labels = sorted({e.label or "(mixed)" for e in kept})
    per_label: dict[str, Counter[Partition]] = defaultdict(Counter)
    for e in kept:
        assert e.partition is not None
        per_label[e.label or "(mixed)"][e.partition] += 1
    supported = set(result.supported_classes)
    md += [
        "## Events per label",
        "",
        "**Bold** labels are the supported classes.",
        "",
        _table(
            ["Label", *(p.value for p in ORDER)],
            [
                [f"**{lab}**" if lab in supported else lab, *(per_label[lab][p] for p in ORDER)]
                for lab in sorted(labels, key=lambda x: -sum(per_label[x].values()))
            ],
        ),
        "",
    ]

    # Species selection
    rule = spec.species_selection
    md += [
        "## Supported species selection",
        "",
        f"Rule, applied to single-species events in the **training partition only**: at least "
        f"{rule.min_train_events} training events and at least {rule.min_train_cameras} training "
        f"cameras with >= {rule.min_events_per_camera} events each.",
        "",
        f"Supported classes (model output order): `{result.supported_classes}`",
        "",
        _table(
            ["Species", "Training events", "Training cameras", "Selected", "Reason"],
            [
                [
                    s.species,
                    s.train_events,
                    s.train_cameras,
                    "yes" if s.selected else "no",
                    s.reason,
                ]
                for s in result.species
            ],
        ),
        "",
        "Animals not selected stay identifiable as `unsupported_animal` events. They are "
        "never used as empty (negative) examples; they are evaluation cases for unsupported "
        "inputs. Species absent from training appear in no row above.",
        "",
    ]

    # Taxonomy
    md += [
        "## Taxonomy mapping",
        "",
        _table(
            ["Source category", "Kind", "Role in this build"],
            [
                [
                    name,
                    c.kind.value,
                    "supported class"
                    if name in supported
                    else ("unsupported animal" if c.kind.value == "animal" else c.kind.value),
                ]
                for name, c in sorted(taxonomy.categories.items())
            ],
        ),
        "",
    ]

    # Benchmark deviations
    multi_file = sum(1 for e in result.events if len({f.source_file for f in e.frames}) > 1)
    dev_from_cis = sorted(
        {
            e.camera_id
            for e in kept
            if e.partition in (Partition.CALIBRATION, Partition.POLICY_VALIDATION)
            and not any("trans_" in f.source_file for f in e.frames)
        },
        key=_camera_key,
    )
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for e in kept:
        assert e.partition is not None
        matrix["+".join(_short(f) for f in sorted({f.source_file for f in e.frames}))][
            e.partition.value
        ] += 1
    md += [
        "## Published benchmark partitions and deviations",
        "",
        "Events by the published CCT20 file(s) their frames came from:",
        "",
        _table(
            ["Published file(s)", *(p.value for p in ORDER)],
            [[k, *(v[p.value] for p in ORDER)] for k, v in sorted(matrix.items())],
        ),
        "",
        "Deviations from the published split:",
        "",
        f"1. **{multi_file} sequences have frames in more than one published file** "
        "(`train` and `cis_val`): sequence leakage in the published split. Partitions here "
        "are assigned per whole sequence, so those files are pooled on training cameras.",
        f"2. **Development cameras {', '.join(dev_from_cis)} come from the published cis "
        "cameras**, because the published `trans_val` has a single camera. All of their "
        "published train/val/test images move to development.",
        "3. **Published `cis_test` is not used as a test**: its cameras are training "
        "cameras. Instead a deterministic sample of whole training-camera sequences is a "
        "seen-camera diagnostic (in-distribution performance only). Using all of `cis_test` "
        "would have removed about half of the training events.",
        "4. **Policy validation uses cameras "
        f"{', '.join(sorted(spec.cameras.policy_validation, key=_camera_key))}**: the "
        "published `trans_val` camera(s) plus development cameras listed above.",
        "5. **`trans_test` is the locked final test, unchanged.**",
        "",
    ]

    # Excluded
    md += ["## Excluded events", ""]
    if excluded:
        md += [
            _table(
                ["Event", "Reason", "Frames"],
                [[e.event_id, e.excluded_reason, len(e.frames)] for e in excluded],
            ),
            "",
        ]
    else:
        md += ["None.", ""]
    notes = [e for e in kept if e.notes]
    if notes:
        md += [
            f"{len(notes)} kept event(s) carry notes, e.g. "
            f"`{notes[0].event_id}`: {notes[0].notes[0]}. Their empty-annotated frames are "
            "not used as fit examples.",
            "",
        ]

    # Grouping rule for uploads
    g = grouping_agreement(result)
    md += [
        "## Upload grouping rule evaluated on CCT20",
        "",
        f"Uploads without sequence ids use `{g['rule']}`. Applied to all "
        f"{sum(len(e.frames) for e in result.events)} CCT20 images (ignoring their sequence ids):",
        "",
        _table(
            ["", "Count"],
            [
                ["True sequences", g["sequences"]],
                ["Time-gap events", g["events"]],
                ["Events matching exactly one whole sequence", g["exact"]],
                ["Events merging several sequences (back-to-back triggers)", g["merged"]],
                ["Sequences split across events", g["split_sequences"]],
            ],
        ),
        "",
        "The rule never splits a trigger burst but merges re-triggers that start within the "
        "gap. For uploads this errs toward fewer, longer events to review; event counts from "
        "time grouping are not comparable to sequence counts.",
        "",
    ]

    # Limitations
    n = {p: len(cams[p]) for p in ORDER}
    md += [
        "## Limitations",
        "",
        f"- **Few locations.** {sum(n.values()) - n[Partition.SEEN_CAMERA_DIAGNOSTIC]} cameras "
        f"in total: {n[Partition.TRAIN]} train, {n[Partition.CALIBRATION]} calibration, "
        f"{n[Partition.POLICY_VALIDATION]} policy validation, {n[Partition.FINAL_TEST]} final "
        "test. Calibration and threshold choices rest on 2 cameras each and may not transfer; "
        "final-test results must be reported per camera, with camera-level variation.",
        "- **Low empty rate.** CCT20 events are mostly animals; empty-filtering numbers here "
        "will not reflect a real memory card (~70% empty in full Caltech Camera Traps).",
        "- **Uneven species coverage.** Some supported species are rare on development or "
        "test cameras (see Events per label), so per-species results there will be noisy.",
        "- **Non-animal triggers.** `car` events occur on one training and one test camera.",
        "",
    ]
    out.mkdir(parents=True, exist_ok=True)
    path = out / "README.md"
    path.write_text("\n".join(md))
    return path


def _image_fit_counts(result: BuildResult) -> list[tuple[Partition, int]]:
    from wildinbox.datasets.events import fit_exclusion

    out = []
    for e in result.events:
        if e.excluded_reason or e.partition is None:
            continue
        out.append((e.partition, sum(1 for f in e.images if fit_exclusion(e, f) is None)))
    return out
