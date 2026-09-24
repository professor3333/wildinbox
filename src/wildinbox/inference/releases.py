"""Model release records: the unit that is deployed, audited, and rolled back."""

from __future__ import annotations

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from wildinbox.config import WildInboxConfig
from wildinbox.inference.plumbing import TEST_NOTICE, TEST_RELEASE_ID
from wildinbox.policy.review_all import POLICY_VERSION
from wildinbox.storage.models import ModelRelease


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
