"""Model releases: the immutable unit that is deployed, audited, and rolled back.

A release bundles everything that determines a prediction and its decision:
weights (content-addressed by SHA-256), taxonomy (class names and their
fingerprint), preprocessing, calibration, and the decision policy with its
thresholds. Jobs are pinned to one release when they are created, so a retry
runs the same release even after another one becomes the default.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from wildinbox.class_map import ClassMap
from wildinbox.config import PreprocessingConfig, WildInboxConfig
from wildinbox.inference.plumbing import TEST_NOTICE, TEST_RELEASE_ID
from wildinbox.policy.review_all import POLICY_VERSION
from wildinbox.settings import Settings
from wildinbox.storage.models import ModelRelease, ReleaseActivation
from wildinbox.storage.objects import ObjectStore

FINETUNED_KIND = "finetuned_efficientnet_b0"
# Fields that define a release; two registrations with the same id must agree on all.
DEFINING = (
    "kind",
    "weights_key",
    "weights_sha256",
    "class_names",
    "class_map_fingerprint",
    "preprocessing",
    "preprocessing_version",
    "calibration",
    "policy",
    "policy_version",
)


class ReleaseError(RuntimeError):
    pass


def ensure_test_release(session: Session, cfg: WildInboxConfig) -> ModelRelease:
    """Create the test-predictor release if missing. Idempotent."""
    session.execute(
        insert(ModelRelease)
        .values(
            id=TEST_RELEASE_ID,
            kind="test_predictor",
            is_test=True,
            weights_key=None,
            class_names=list(cfg.classes),
            class_map_fingerprint=cfg.class_map.fingerprint(),
            preprocessing=cfg.preprocessing.model_dump(),
            preprocessing_version=cfg.preprocessing.fingerprint(),
            calibration=None,
            policy={"automation": False},
            policy_version=POLICY_VERSION,
            notes=TEST_NOTICE,
        )
        .on_conflict_do_nothing(index_elements=["id"])
    )
    release = session.get(ModelRelease, TEST_RELEASE_ID)
    assert release is not None
    return release


def release_notice(release: ModelRelease) -> str | None:
    return TEST_NOTICE if release.is_test else None


def weights_key(sha256: str) -> str:
    return f"releases/weights/{sha256}.pt"


def build_release(model_dir: Path, policy_path: Path) -> tuple[dict[str, Any], bytes]:
    """The release row for a trained model and its policy artifact, plus the
    weights bytes. Every cross-reference is checked before anything is stored."""
    meta = json.loads((model_dir / "meta.json").read_text())
    policy = json.loads(policy_path.read_text())
    weights = (model_dir / "model.pt").read_bytes()
    sha = hashlib.sha256(weights).hexdigest()

    classes = list(meta["classes"])
    if policy["model"] != meta["name"]:
        raise ReleaseError(f"policy is for {policy['model']!r}, model dir is {meta['name']!r}")
    if policy["classes"] != classes:
        raise ReleaseError("policy and model disagree on the class order")
    if policy["calibration"]["weights_digest"] != sha[:12]:
        raise ReleaseError("calibration was fitted on different weights than model.pt")
    class_map = ClassMap(classes)
    if class_map.fingerprint() != meta["class_map_fingerprint"]:
        raise ReleaseError("class map fingerprint does not match the model's metadata")
    pre = PreprocessingConfig.model_validate(meta["preprocessing"])
    if pre.fingerprint() != meta["preprocessing_version"]:
        raise ReleaseError("preprocessing settings do not match their recorded version")

    released = {k: v for k, v in policy["released"].items() if k != "policy_version"}
    row = {
        "id": f"{meta['name']}@{policy['artifact_version']}",
        "kind": FINETUNED_KIND,
        "is_test": False,
        "weights_key": weights_key(sha),
        "weights_sha256": sha,
        "class_names": classes,
        "class_map_fingerprint": class_map.fingerprint(),
        "preprocessing": pre.model_dump(mode="json"),
        "preprocessing_version": pre.fingerprint(),
        "calibration": policy["calibration"],
        "policy": {
            "name": policy["policy"],
            "config": released,
            "unfamiliar": policy.get("unfamiliar"),
        },
        "policy_version": policy["released"]["policy_version"],
        "provenance": {
            "model": meta["name"],
            "training_code": meta.get("code"),
            "split_version": meta.get("split_version"),
            "mlflow_run_id": meta.get("mlflow_run_id"),
            "policy_artifact_version": policy["artifact_version"],
        },
    }
    return row, weights


def register_release(
    session: Session, store: ObjectStore, model_dir: Path, policy_path: Path, note: str | None
) -> tuple[ModelRelease, bool]:
    """Store the weights (content-addressed) and insert the release. Registering
    the same release again is a no-op; a different release under the same id is
    refused. Returns (release, created)."""
    row, weights = build_release(model_dir, policy_path)
    existing = session.get(ModelRelease, row["id"])
    if existing is not None:
        differs = [k for k in DEFINING if getattr(existing, k) != row[k]]
        if differs:
            raise ReleaseError(f"release {row['id']} already exists with different {differs}")
        return existing, False
    if not store.exists(row["weights_key"]):
        store.put(row["weights_key"], weights, "application/octet-stream")
    session.add(ModelRelease(**row, notes=note))
    session.flush()
    release = session.get(ModelRelease, row["id"])
    assert release is not None
    return release, True


def activate(session: Session, release_id: str, note: str | None = None) -> ReleaseActivation:
    """Make `release_id` the default for NEW batches. Existing jobs keep theirs."""
    if session.get(ModelRelease, release_id) is None:
        raise ReleaseError(f"release {release_id} is not registered")
    row = ReleaseActivation(release_id=release_id, note=note)
    session.add(row)
    session.flush()
    return row


def active_release_id(session: Session, settings: Settings) -> str:
    latest = session.scalar(
        select(ReleaseActivation.release_id).order_by(ReleaseActivation.id.desc()).limit(1)
    )
    return latest or settings.active_release
