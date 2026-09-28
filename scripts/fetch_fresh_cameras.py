"""Download one group of fresh cameras (manifests/fresh-cameras-v2.json) as a
local source for `wildinbox data ingest`.

Writes `data/raw/<name>/annotations/<name>.json` (the full Caltech Camera
Traps metadata restricted to the group's locations) and downloads each image
to `data/raw/<name>/images/`. Resumable: files already present are kept, and
a download is written to a temporary name and renamed when complete.
Validation (decoding, sha256, duplicates) is left to `wildinbox data ingest`.

The fresh-test group is refused unless `--protocol` names a protocol file
that is committed and unchanged, so the test cannot be downloaded before its
protocol exists.

    uv run python scripts/fetch_fresh_cameras.py --group adaptation_development \\
        --name cct_fresh_dev
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml

GROUPS = ("adaptation_development", "fresh_test")


def _committed_and_clean(path: Path) -> bool:
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path)], capture_output=True
        )
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", str(path)])
    except OSError:
        return False
    return tracked.returncode == 0 and not dirty.strip()


def _fetch(url: str, dest: Path, attempts: int = 4) -> str:
    if dest.exists() and dest.stat().st_size > 0:
        return "kept"
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    for i in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, tmp.open("wb") as f:
                while chunk := r.read(1 << 16):
                    f.write(chunk)
            tmp.replace(dest)
            return "downloaded"
        except OSError:
            if i == attempts - 1:
                raise
            time.sleep(2**i)
    raise AssertionError("unreachable")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--group", choices=GROUPS, required=True)
    ap.add_argument("--name", required=True, help="Source name, e.g. cct_fresh_dev.")
    ap.add_argument("--rule", type=Path, default=Path("configs/experiments/fresh_cameras_v2.yaml"))
    ap.add_argument("--selection", type=Path, default=Path("manifests/fresh-cameras-v2.json"))
    ap.add_argument(
        "--metadata-zip", type=Path, default=Path("data/cct_full/caltech_camera_traps.json.zip")
    )
    ap.add_argument(
        "--metadata", type=Path, default=Path("data/cct_full/caltech_images_20210113.json")
    )
    ap.add_argument("--protocol", type=Path, help="Required for the fresh-test group.")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    if args.group == "fresh_test" and not (
        args.protocol and args.protocol.exists() and _committed_and_clean(args.protocol)
    ):
        raise SystemExit("the fresh test is downloaded only under a committed, unchanged protocol")
    rule = yaml.safe_load(args.rule.read_text())
    zip_sha = hashlib.sha256(args.metadata_zip.read_bytes()).hexdigest()
    if zip_sha != rule["source"]["metadata_zip_sha256"]:
        raise SystemExit(f"metadata archive sha256 {zip_sha} is not the one the rule pins")
    selection = json.loads(args.selection.read_text())
    locations = {c["location"] for c in selection[args.group]}

    meta = json.loads(args.metadata.read_text())
    images = [im for im in meta["images"] if str(im["location"]) in locations]
    ids = {im["id"] for im in images}
    subset: dict[str, Any] = {
        "info": {
            **meta["info"],
            "wildinbox_subset": {
                "group": args.group,
                "locations": sorted(locations, key=int),
                "selection": str(args.selection),
                "metadata_zip_sha256": zip_sha,
            },
        },
        "categories": meta["categories"],
        "images": images,
        "annotations": [a for a in meta["annotations"] if a["image_id"] in ids],
    }
    root = args.data_dir / "raw" / args.name
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "annotations" / f"{args.name}.json").write_text(json.dumps(subset))

    base = rule["source"]["image_base_url"]
    outcomes: dict[str, int] = {}
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [
            pool.submit(
                _fetch, base + im["file_name"], root / "images" / Path(im["file_name"]).name
            )
            for im in images
        ]
        for i, fut in enumerate(as_completed(futures), 1):
            outcome = fut.result()
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            if i % 500 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)} {outcomes}", flush=True)
    print(f"{args.group}: {len(images)} images from {len(locations)} cameras -> {root}")


if __name__ == "__main__":
    main()
