"""Build an upload-ready sample batch from the development cameras.

    uv run python scripts/make_sample_batch.py --images 1000 --out data/samples/dev-1000

Takes whole capture sequences (never splitting one) from the calibration and
policy-validation partitions, in a fixed pseudo-random order, until the image
count is reached. The locked final test is never read (load_rows refuses it).
Writes the images, `metadata.json` (the API's metadata field: camera, sequence,
and capture time per file, as a camera's memory card would carry them), and
`truth.csv` (ground truth, not uploaded).
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("data/samples/dev-1000"))
    parser.add_argument("--seed", default="wildinbox-sample-v1")
    args = parser.parse_args()

    ctx = load_context(Path("configs/experiments/baseline.yaml"), Settings().data_dir)
    rows, _ = load_rows(ctx.split_dir, [Partition.CALIBRATION, Partition.POLICY_VALIDATION])
    conn = sqlite3.connect(ctx.inventory_db)
    captured = {
        sid: json.loads(raw).get("date_captured")
        for sid, raw in conn.execute("SELECT source_id, raw_image FROM records")
    }
    conn.close()
    by_event: dict[str, list] = {}
    for r in rows:
        by_event.setdefault(r.event_id, []).append(r)
    order = sorted(by_event, key=lambda e: hashlib.sha256(f"{args.seed}:{e}".encode()).hexdigest())

    chosen = []
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
            ["filename", "sequence_id", "camera_id", "partition", "image_label", "event_role"]
        )
        for r in chosen:
            name = f"cam{r.camera_id}_{r.source_id}{Path(r.storage_path).suffix.lower()}"
            shutil.copyfile(ctx.images_root / r.storage_path, args.out / "images" / name)
            files[name] = {"camera_id": f"cct-{r.camera_id}", "sequence_id": r.event_id}
            if captured.get(r.source_id):
                files[name]["captured_at"] = str(captured[r.source_id]).replace(" ", "T")
            w.writerow(
                [name, r.event_id, r.camera_id, r.partition.value, r.image_label, r.event_role]
            )
    (args.out / "metadata.json").write_text(json.dumps({"files": files}, indent=1) + "\n")
    size = sum(p.stat().st_size for p in (args.out / "images").iterdir())
    print(
        f"{len(chosen)} images from {len({r.event_id for r in chosen})} sequences, "
        f"{size / 1e6:.1f} MB -> {args.out}"
    )


if __name__ == "__main__":
    main()
