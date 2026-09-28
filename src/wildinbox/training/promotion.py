"""The promotion policy: a gate's recorded results -> checks -> a decision.

Three outcomes:

- `promote`: every check passed on enough evidence;
- `reject`: a check failed on enough evidence;
- `inconclusive`: nothing failed, but some check had too little evidence to
  judge (a critical species with few holdout events, a small holdout) or the
  holdout's comparison budget is spent. Collect more reviewed events, or a
  fresh holdout, and gate again; an inconclusive candidate is not released.

Aggregate macro-F1 can rise while one species collapses (the rehearsal
candidate: +0.087 macro-F1, bobcat recall 0.593 -> 0.333), so a protocol also
names its critical species and the largest recall drop it accepts for each.

Protocol keys under `gate` (the per-species and evidence blocks are required
unless the gate is run with the legacy policy, which recorded cycles used):

    min_holdout_gain: 0.02
    max_regression: 0.02
    max_false_empty_suggestion_increase: 0.10
    species:
      critical: [bobcat, cat, coyote, dog, opossum, rabbit, raccoon]
      max_recall_drop: 0.10     # absolute, per critical species, on the holdout
      min_events: 30            # holdout events of that species to judge it
    min_evidence:
      holdout_events: 200       # supported-label events, for the macro-F1 gain
      animal_events: 100        # animal events, for the false-empty check
"""

from __future__ import annotations

from typing import Any

PROMOTE, REJECT, INCONCLUSIVE = "promote", "reject", "inconclusive"
LEGACY, SPECIES = "legacy", "species-v1"


class PolicyError(ValueError):
    """The protocol's promotion policy is missing or malformed."""


def policy_level(gate: dict[str, Any], classes: list[str], *, allow_legacy: bool) -> str:
    """Validate the protocol's gate block; return which policy it sets."""
    has = [k for k in ("species", "min_evidence") if k in gate]
    if not has:
        if allow_legacy:
            return LEGACY
        raise PolicyError(
            "the protocol sets no per-species limits or minimum evidence (gate.species, "
            "gate.min_evidence); add them, or pass --legacy-policy to reproduce a recorded cycle"
        )
    if len(has) == 1:
        raise PolicyError(f"the protocol sets gate.{has[0]} but not the other block")
    species, evidence = gate["species"], gate["min_evidence"]
    critical = species.get("critical")
    if not critical or not isinstance(critical, list):
        raise PolicyError("gate.species.critical must list at least one class")
    unknown = sorted(set(critical) - set(classes))
    if unknown:
        raise PolicyError(f"gate.species.critical names classes the models lack: {unknown}")
    drop = species.get("max_recall_drop")
    if not isinstance(drop, int | float) or not 0 <= drop < 1:
        raise PolicyError("gate.species.max_recall_drop must be a number in [0, 1)")
    for where, block, key in (
        ("gate.species", species, "min_events"),
        ("gate.min_evidence", evidence, "holdout_events"),
        ("gate.min_evidence", evidence, "animal_events"),
    ):
        v = block.get(key)
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            raise PolicyError(f"{where}.{key} must be a positive integer")
    return SPECIES


def _enough(n: int, minimum: int | None) -> bool:
    return minimum is None or n >= minimum


def evidence_checks(
    deployed: dict[str, Any], candidate: dict[str, Any], gate: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Every check but the comparison budget. `pass` is True, False, or None
    (too little evidence to judge; `support` < `min_support`)."""
    d, c = deployed["holdout"], candidate["holdout"]
    evidence = gate.get("min_evidence") or {}
    gain = c["macro_f1"] - d["macro_f1"]
    n_events, min_events = d["events_scored"], evidence.get("holdout_events")
    checks: dict[str, dict[str, Any]] = {
        "holdout_gain": {
            "value": gain,
            "required": gate["min_holdout_gain"],
            "pass": gain >= gate["min_holdout_gain"] if _enough(n_events, min_events) else None,
            **({"support": n_events, "min_support": min_events} if min_events else {}),
        },
    }
    for part, before in deployed["regression_macro_f1"].items():
        drop = before - candidate["regression_macro_f1"][part]
        checks[f"no_regression_{part}"] = {
            "value": -drop,
            "allowed": -gate["max_regression"],
            "pass": drop <= gate["max_regression"],
        }
    lost_d, lost_c = d["animal_events_suggested_empty"], c["animal_events_suggested_empty"]
    limit = lost_d * (1 + gate["max_false_empty_suggestion_increase"])
    n_animals, min_animals = d["animal_events"], evidence.get("animal_events")
    checks["false_empty_suggestions"] = {
        "value": lost_c,
        "limit": limit,
        "pass": lost_c <= limit if _enough(n_animals, min_animals) else None,
        **({"support": n_animals, "min_support": min_animals} if min_animals else {}),
    }
    species = gate.get("species")
    if species:
        for k in species["critical"]:
            before, after = d["per_class"][k]["recall"], c["per_class"][k]["recall"]
            n = d["per_class"][k]["support"]
            change = after - before
            checks[f"species_recall_{k}"] = {
                "value": change,
                "allowed": -species["max_recall_drop"],
                "deployed": before,
                "candidate": after,
                "support": n,
                "min_support": species["min_events"],
                # A tiny tolerance keeps a drop of exactly the limit from failing
                # on floating-point error.
                "pass": -change <= species["max_recall_drop"] + 1e-9
                if n >= species["min_events"]
                else None,
            }
    return checks


def budget_check(number: int, budget: int) -> dict[str, Any]:
    """Past the budget the holdout has been fit by repeated comparisons: it can
    no longer vouch for a candidate either way, so the outcome is inconclusive."""
    return {"value": number, "limit": budget, "pass": True if number <= budget else None}


def decide(checks: dict[str, dict[str, Any]]) -> str:
    results = [v["pass"] for v in checks.values()]
    if any(r is False for r in results):
        return REJECT
    if any(r is None for r in results):
        return INCONCLUSIVE
    return PROMOTE


def release_authorization(record: dict[str, Any], *, allow_legacy: bool) -> str:
    """Whether a gate record (`metrics.json`) authorizes a release; returns its
    policy, raises PolicyError otherwise. For anything that consumes gate
    records to release a candidate.

    - No `promotion_policy` marker means the record predates this policy: it is
      legacy, like one gated with `--legacy-policy`, and needs `allow_legacy`.
    - An unrecognized policy is refused, never guessed at.
    - The recorded decision must be promote and agree with `promote` and with
      the recorded checks, so an edited or inconsistent record cannot release.
    """
    policy = record.get("promotion_policy", LEGACY)
    if policy not in (LEGACY, SPECIES):
        raise PolicyError(f"unrecognized promotion policy {policy!r}")
    if policy == LEGACY and not allow_legacy:
        where = "records no promotion policy" if "promotion_policy" not in record else "is legacy"
        raise PolicyError(
            f"the gate record {where}: aggregate checks only, no per-species limits or "
            "minimum evidence. Release it only to reproduce a recorded cycle, with the "
            "legacy opt-in"
        )
    checks = record.get("checks")
    if not isinstance(checks, dict) or not checks:
        raise PolicyError("the gate record has no checks")
    if policy == SPECIES and not any(k.startswith("species_recall_") for k in checks):
        raise PolicyError("the gate record claims the species policy but has no species checks")
    decision = record.get("decision", PROMOTE if record.get("promote") is True else REJECT)
    if decision not in (PROMOTE, REJECT, INCONCLUSIVE):
        raise PolicyError(f"unrecognized decision {decision!r}")
    if decide(checks) != decision or record.get("promote") is not (decision == PROMOTE):
        raise PolicyError(
            f"the gate record is inconsistent: decision {decision!r}, promote "
            f"{record.get('promote')!r}, checks give {decide(checks)!r}"
        )
    if decision != PROMOTE:
        raise PolicyError(f"the gate decided {decision}, not promote")
    return str(policy)
