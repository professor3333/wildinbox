"""Build an upload-ready sample batch from the development cameras.

    uv run python scripts/make_sample_batch.py --images 1000 --out data/samples/dev-1000
    uv run python scripts/make_sample_batch.py --images 36 --diverse --out samples/cct-dev

Takes whole capture sequences (never splitting one) from the calibration and
policy-validation partitions, in a fixed pseudo-random order, until the image
count is reached (with --diverse, one sequence per label in turn, so every
supported species, empty frames, and unsupported species appear). The locked
final test is never read (load_rows refuses it).
Writes the images, `metadata.json` (the API's metadata field: camera, sequence,
and capture time per file, as a camera's memory card would carry them), and
`truth.csv` (ground truth and rights holder, not uploaded), and `LICENSE.md`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

from wildinbox.datasets.spec import Partition
from wildinbox.evaluation.data import load_rows
from wildinbox.settings import Settings
from wildinbox.training.run import load_context

LICENSE = """# License and attribution

These images come from the **Caltech Camera Traps** dataset (CCT20 subset),
distributed by LILA BC under the **Community Data License Agreement -
Permissive, Version 1.0** (https://cdla.dev/permissive-1-0/).

- Dataset: https://lila.science/datasets/caltech-camera-traps
- Reference: Beery, S., Van Horn, G., and Perona, P. "Recognition in Terra
  Incognita." ECCV 2018.
- Rights holders of these images: {holders} (per image in `truth.csv`).

Files are unchanged apart from their names (`cam<camera>_<image id>`). They
come from WildInbox's development cameras, never from its locked final test.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("data/samples/dev-1000"))
    parser.add_argument("--seed", default="wildinbox-sample-v1")
    parser.add_argument("--diverse", action="store_true", help="round-robin over labels")
    parser.add_argument(
        "--partition",
        action="append",
        choices=["calibration", "policy_validation"],
        help="development partition(s) to draw from (default: both)",
    )
    args = parser.parse_args()

    ctx = load_context(Path("configs/experiments/baseline.yaml"), Settings().data_dir)
    parts = [Partition(p) for p in (args.partition or ["calibration", "policy_validation"])]
    rows, events = load_rows(ctx.split_dir, parts)
    conn = sqlite3.connect(ctx.inventory_db)
    raw_images = {
        sid: json.loads(raw)
        for sid, raw in conn.execute("SELECT source_id, raw_image FROM records")
    }
    captured = {sid: r.get("date_captured") for sid, r in raw_images.items()}
    holder = {sid: r.get("rights_holder") for sid, r in raw_images.items()}
    conn.close()
    by_event: dict[str, list] = {}
    for r in rows:
        by_event.setdefault(r.event_id, []).append(r)
    order = sorted(by_event, key=lambda e: hashlib.sha256(f"{args.seed}:{e}".encode()).hexdigest())

    chosen = []
    if args.diverse:
        by_label: dict[str, list[str]] = {}
        for eid in order:
            first = by_event[eid][0]
            by_label.setdefault(first.event_label or first.event_role, []).append(eid)
        queues = [by_label[k] for k in sorted(by_label)]
        while len(chosen) < args.images and any(queues):
            for q in queues:
                if q and len(chosen) < args.images:
                    chosen.extend(by_event[q.pop(0)])
    else:
        for eid in order:
            if len(chosen) >= args.images:
                break
            chosen.extend(by_event[eid])

    if args.out.exists():
        shutil.rmtree(args.out)
    (args.out / "images").mkdir(parents=True)
    files: dict[str, dict[str, str]] = {}
    with (args.out / "truth.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "filename",
                "sequence_id",
                "camera_id",
                "captured_at",
                "partition",
                "image_label",
                "event_role",
                "event_label",
                "rights_holder",
            ]
        )
        for r in chosen:
            name = f"cam{r.camera_id}_{r.source_id}{Path(r.storage_path).suffix.lower()}"
            shutil.copyfile(ctx.images_root / r.storage_path, args.out / "images" / name)
            files[name] = {"camera_id": f"cct-{r.camera_id}", "sequence_id": r.event_id}
            if captured.get(r.source_id):
                files[name]["captured_at"] = str(captured[r.source_id]).replace(" ", "T")
            w.writerow(
                [
                    name,
                    r.event_id,
                    r.camera_id,
                    captured.get(r.source_id),
                    r.partition.value,
                    r.image_label,
                    r.event_role,
                    events[r.event_id].label,
                    holder.get(r.source_id),
                ]
            )
    (args.out / "metadata.json").write_text(json.dumps({"files": files}, indent=1) + "\n")
    holders = sorted({h for r in chosen if (h := holder.get(r.source_id))})
    (args.out / "LICENSE.md").write_text(
        LICENSE.format(holders=", ".join(holders) or "see truth.csv")
    )
    size = sum(p.stat().st_size for p in (args.out / "images").iterdir())
    print(
        f"{len(chosen)} images from {len({r.event_id for r in chosen})} sequences, "
        f"{size / 1e6:.1f} MB -> {args.out}"
    )


if __name__ == "__main__":
    main()
