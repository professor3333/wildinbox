from __future__ import annotations

import uuid

from wildinbox.policy.audit import rule, selected


def test_only_automatic_decisions_are_audited_at_the_configured_rate() -> None:
    ids = [uuid.UUID(int=i * 7919 + 1) for i in range(20000)]
    picked = sum(selected(e, "likely_empty", 0.05, "seed") for e in ids)
    assert 0.045 < picked / len(ids) < 0.055
    assert not any(selected(e, "needs_review", 1.0, "seed") for e in ids[:100])
    assert all(selected(e, "species_identified", 1.0, "seed") for e in ids[:100])
    assert not any(selected(e, "likely_empty", 0.0, "seed") for e in ids[:100])


def test_selection_is_reproducible_from_the_recorded_rule() -> None:
    e = uuid.uuid4()
    assert selected(e, "likely_empty", 0.3, "a") == selected(str(e), "likely_empty", 0.3, "a")
    assert rule(0.05, "wildinbox-audit-v1") == "sha256-uniform(rate=0.05, seed=wildinbox-audit-v1)"
    different_seed = sum(
        selected(uuid.UUID(int=i + 1), "likely_empty", 0.5, "a")
        != selected(uuid.UUID(int=i + 1), "likely_empty", 0.5, "b")
        for i in range(1000)
    )
    assert different_seed > 300  # the seed changes which events are sampled
