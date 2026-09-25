"""Audit sampling of automatically handled events.

Reviewing only uncertain events hides confident mistakes (CLAUDE.md, biased
feedback), so a fixed share of events that automation filtered or labeled is
sent to a person anyway. Selection hashes the event id with a seed: event ids
are random UUIDs, so this is a uniform random sample, and anyone can recompute
exactly which events were chosen from the recorded rule.
"""

from __future__ import annotations

import hashlib
import uuid

AUTOMATIC = ("likely_empty", "species_identified")


def rule(rate: float, seed: str) -> str:
    return f"sha256-uniform(rate={rate:g}, seed={seed})"


def selected(event_id: uuid.UUID | str, disposition: str, rate: float, seed: str) -> bool:
    """Whether an event goes to the audit queue. Only automatic dispositions
    are ever audited; events needing review are reviewed anyway."""
    if disposition not in AUTOMATIC or rate <= 0:
        return False
    digest = hashlib.sha256(f"{seed}:{event_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < rate
