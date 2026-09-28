"""Apply a fresh-camera rule (configs/experiments/fresh_cameras_v2.yaml; v1 is
kept for the record) to the full Caltech Camera Traps metadata and write the
selection to manifests/fresh-cameras-v2.json.

Labels are read only to apply eligibility. The output records, per selected
camera, sequence and image counts only; per-class counts are written for the
adaptation-development cameras and never for fresh-test cameras.

    curl -o data/cct_full/caltech_camera_traps.json.zip <metadata_url>
    unzip -d data/cct_full data/cct_full/caltech_camera_traps.json.zip
    uv run python scripts/select_fresh_cameras.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# The released model's classes (models/<release>/meta.json), without empty.
SUPPORTED = {"bobcat", "cat", "coyote", "dog", "opossum", "rabbit", "raccoon"}


def _usable_time(ts: str | None) -> bool:
    try:
        return ts is not None and datetime.fromisoformat(ts).year >= 2000
    except ValueError:
        return False


def cameras(meta: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per location: sequences, images, timestamped sequences, class per sequence."""
    names = {c["id"]: c["name"] for c in meta["categories"]}
    labels: dict[str, set[str]] = defaultdict(set)
    for a in meta["annotations"]:
        labels[a["image_id"]].add(names[a["category_id"]])
    seqs: dict[str, dict[str, Any]] = {}
    for im in meta["images"]:
        s = seqs.setdefault(
            im["seq_id"],
            {"location": str(im["location"]), "images": 0, "labels": set(), "timed": True},
        )
        s["images"] += 1
        s["labels"] |= labels[im["id"]]
        s["timed"] &= _usable_time(im.get("date_captured"))
    out: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"sequences": 0, "images": 0, "timed": 0, "classes": Counter()}
    )
    for s in seqs.values():
        c = out[s["location"]]
        c["sequences"] += 1
        c["images"] += s["images"]
        c["timed"] += s["timed"]
        animals = s["labels"] - {"empty"}
        key = next(iter(animals)) if len(animals) == 1 else ("empty" if not animals else "mixed")
        c["classes"][key] += 1
    return dict(out)


def select(rule: dict[str, Any], cams: dict[str, dict[str, Any]]) -> dict[str, Any]:
    excluded = {str(x) for x in rule["exclude_locations"]["cct20"]}
    el = rule["eligibility"]

    def eligible(loc: str) -> bool:
        c = cams[loc]
        supported = sum(n for k, n in c["classes"].items() if k in SUPPORTED)
        return (
            loc not in excluded
            and c["sequences"] >= el["min_sequences"]
            and supported >= el["min_supported_species_sequences"]
            and c["timed"] / c["sequences"] >= el["require_timestamps"]
            and c["images"] / c["sequences"] >= el.get("min_images_per_sequence", 0)
        )

    sel = rule["selection"]
    salt = sel.get("salt", "wildinbox-fresh-v1:")
    order = sorted(
        (loc for loc in cams if eligible(loc)),
        key=lambda loc: hashlib.sha256((salt + loc).encode()).hexdigest(),
    )
    budget, used = sel.get("max_images_total", float("inf")), 0
    groups: dict[str, list[str]] = {"fresh_test": [], "adaptation_development": []}
    rest = iter(order)
    for group in ("fresh_test", "adaptation_development"):
        for loc in rest:
            if used + cams[loc]["images"] > budget:
                break
            groups[group].append(loc)
            used += cams[loc]["images"]
            if len(groups[group]) == sel[group]["cameras"]:
                break

    def record(loc: str, with_classes: bool) -> dict[str, Any]:
        c = cams[loc]
        r = {"location": loc, "sequences": c["sequences"], "images": c["images"]}
        if with_classes:
            r["sequences_by_class"] = dict(sorted(c["classes"].items()))
        return r

    return {
        "locations_total": len(cams),
        "locations_outside_cct20": sum(loc not in excluded for loc in cams),
        "eligible": len(order),
        "images_selected": used,
        "fresh_test": [record(loc, False) for loc in groups["fresh_test"]],
        "adaptation_development": [record(loc, True) for loc in groups["adaptation_development"]],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rule", type=Path, default=Path("configs/experiments/fresh_cameras_v2.yaml"))
    ap.add_argument(
        "--metadata-zip", type=Path, default=Path("data/cct_full/caltech_camera_traps.json.zip")
    )
    ap.add_argument(
        "--metadata", type=Path, default=Path("data/cct_full/caltech_images_20210113.json")
    )
    ap.add_argument("--out", type=Path, default=Path("manifests/fresh-cameras-v2.json"))
    args = ap.parse_args()
    rule = yaml.safe_load(args.rule.read_text())
    size = args.metadata_zip.stat().st_size
    if size != rule["source"]["metadata_size"]:
        raise SystemExit(
            f"metadata archive is {size} bytes, rule pins {rule['source']['metadata_size']}"
        )
    meta = json.loads(args.metadata.read_text())
    result = {
        "name": rule["name"],
        "rule": str(args.rule),
        "rule_sha256": hashlib.sha256(args.rule.read_bytes()).hexdigest(),
        "metadata": {
            "url": rule["source"]["metadata_url"],
            "zip_sha256": hashlib.sha256(args.metadata_zip.read_bytes()).hexdigest(),
            "json": args.metadata.name,
            "version": meta["info"]["version"],
        },
        **select(rule, cameras(meta)),
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"{result['eligible']} eligible of {result['locations_outside_cct20']} locations "
        f"outside CCT20; fresh test {len(result['fresh_test'])} cameras, "
        f"adaptation development {len(result['adaptation_development'])}, "
        f"{result['images_selected']:,} images -> {args.out}"
    )


if __name__ == "__main__":
    main()
