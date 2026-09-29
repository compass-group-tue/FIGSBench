"""Deterministic final-benchmark slot specs: rule x domain x severity grid cells,
one archetype attached per slot by seeded rotation within each rule's compatible set."""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from archetype_guided_benchmark.sources import (
    BENCHMARK_DOMAIN_COUNT,
    MERGED_DOMAIN_ID,
    MERGED_DOMAIN_NAME,
    PERSONAL_LEANING_DOMAIN_IDS,
    PROFESSIONAL_LEANING_DOMAIN_IDS,
    benchmark_domain_taxonomy,
)

def _legacy_seed_label(rule: str) -> str:
    """Sycophancy rules were named S1.a-S2.d when the benchmark was built; RNG seeds keep that."""
    return "R" + rule[1:] if rule[:1] == "S" else rule


COMPAT_PATH = Path(__file__).resolve().parent / "config" / "rule_archetype_compatibility.json"

# Pooled archetypes: D/E/I share an exact per-rule quota (sums to 75).
# Non-pool archetypes fill to exact per-archetype quotas via min-count greedy.
# Assignment is fully deterministic given the seed.
POOLED_ARCHETYPES = ("D", "E", "I")
POOL_QUOTA = {"S1.a": 9, "S1.b": 9, "S1.c": 10, "S2.a": 9, "S2.c": 9, "S2.d": 29}
ARCH_QUOTA = {
    "A": 37, "B": 38, "F": 37, "G": 31, "H": 30, "J": 38, "K": 38,
    "L": 33, "M": 33,
}

FINAL_PLANNING_SEED = 20260915
SYC_REPLICATES = 3
CV_REPLICATES = 2
EXTRA_SLOTS = 14
SYC_RULES = ["S1.a", "S1.b", "S1.c", "S2.a", "S2.b", "S2.c", "S2.d"]
CV_RULES = ["V1", "V2", "V3"]
SEVERITIES = ["low", "medium", "high"]
SYC_GENERATORS = ["thinkingmachines/inkling", "google/gemini-3.8-flash"]
CV_GENERATOR = "google/gemini-3.8-flash"


def _domains() -> list[dict[str, str]]:
    return benchmark_domain_taxonomy(
        [
            {"id": "health_medicine", "name": "Health & Medicine"},
            {"id": "work_careers_organizations", "name": "Work, Careers & Organizations"},
            {"id": "finance_law", "name": "Finance & Law"},
            {"id": "science_technology", "name": "Science & Technology"},
            {"id": "intimate_relationships", "name": "Intimate Relationships"},
            {"id": "family_friends_social_life", "name": "Family, Friends & Social Life"},
            {"id": "beliefs_culture_society", "name": "Beliefs, Culture & Society"},
        ]
    )


def build_final_specs(seed: int = FINAL_PLANNING_SEED) -> list[dict]:
    compat = json.loads(COMPAT_PATH.read_text())
    domains = _domains()
    assert len(domains) == BENCHMARK_DOMAIN_COUNT
    pending: list[tuple] = []

    def add_slot(axis, rule, domain, severity, extra=False, generator=None):
        pending.append((axis, rule, domain, severity, extra, generator))

    base_cells: list[tuple] = []
    for axis, rules, reps in (("sycophancy", SYC_RULES, SYC_REPLICATES),
                              ("calibrated_validation", CV_RULES, CV_REPLICATES)):
        cells = [(axis, r, d, s) for r in rules for d in domains for s in SEVERITIES]
        rng = random.Random(f"{seed}:cells:{axis}")
        rng.shuffle(cells)
        for _ in range(reps):
            order = list(cells)
            rng.shuffle(order)
            base_cells.extend(order)
    for axis, rule, domain, sev in base_cells:
        add_slot(axis, rule, domain, sev)

    rng = random.Random(f"{seed}:extras")
    all_cells = ([("sycophancy", r, d, s) for r in SYC_RULES for d in domains for s in SEVERITIES]
                 + [("calibrated_validation", r, d, s) for r in CV_RULES for d in domains for s in SEVERITIES])
    for axis, rule, domain, sev in rng.choices(all_cells, k=EXTRA_SLOTS):
        add_slot(axis, rule, domain, sev, extra=True)

    archetypes = _assign_archetypes(
        [(axis, rule) for axis, rule, *_ in pending], compat, seed
    )

    specs: list[dict] = []
    ordinal = {"sycophancy": 0, "calibrated_validation": 0}
    gen_state: dict[str, int] = {}
    for (axis, rule, domain, severity, extra, generator), arch in zip(pending, archetypes):
        ordinal[axis] += 1
        prefix = "SYC" if axis == "sycophancy" else "CV"
        if generator is None and axis == "sycophancy":
            i = gen_state.get(rule, 0)
            generator = SYC_GENERATORS[i % 2]
            gen_state[rule] = i + 1
        elif axis == "calibrated_validation":
            generator = CV_GENERATOR
        specs.append(
            {
                "slot_id": f"{prefix}-{ordinal[axis]:03d}",
                "axis": axis,
                "rule_id": rule,
                "domain_id": domain["id"],
                "domain_name": domain["name"],
                "severity": severity,
                "archetype_id": arch,
                "generator": generator,
                "extra": extra,
            }
        )
    return specs


def _assign_archetypes(slots: list[tuple], compat: dict, seed: int) -> list[str]:
    """Balanced deterministic archetype assignment.

    Phase 1: pooled D/E/I get a capped quota (POOL_RATE of each syc rule's
    slots), spread evenly, member picked by min-count for diversity.
    Phase 2: every other slot goes to the compatible archetype with the lowest
    running total (seeded tie-break) — near-uniform totals on both axes.
    """
    out: list[str | None] = [None] * len(slots)
    tie = random.Random(f"{seed}:arch-tie")
    counts: Counter = Counter()
    pool_counts: Counter = Counter()

    by_rule: dict[tuple, list[int]] = {}
    for i, (axis, rule) in enumerate(slots):
        by_rule.setdefault((axis, rule), []).append(i)

    for (axis, rule), idxs in by_rule.items():
        if axis != "sycophancy":
            continue
        members = [a for a in compat[axis][rule] if a in POOLED_ARCHETYPES]
        if not members:
            continue
        cap = POOL_QUOTA.get(rule, 0)
        if cap <= 0:
            continue
        step = len(idxs) / cap
        order = list(members)
        # Seed with the pre-rename rule label (S1.a was S1.a) so a given planning seed keeps
        # producing the same grid: --seed 20260915 still reproduces final_500_specs.json.
        random.Random(f"{seed}:arch-pool:{_legacy_seed_label(rule)}").shuffle(order)
        for k in range(cap):
            pos = idxs[min(len(idxs) - 1, int((k + 0.5) * step))]
            if out[pos] is not None:
                continue
            cand = sorted(members, key=lambda a: (pool_counts[a], order.index(a)))
            out[pos] = cand[0]
            pool_counts[cand[0]] += 1

    forced_left = sum(
        1
        for i, (axis, rule) in enumerate(slots)
        if out[i] is None
        and [a for a in compat[axis][rule] if a not in POOLED_ARCHETYPES] == ["H"]
    )
    for i, (axis, rule) in enumerate(slots):
        if out[i] is not None:
            continue
        cands = [a for a in compat[axis][rule] if a not in POOLED_ARCHETYPES]
        # Forced-H slots (S2.d non-pool: H is the only option) always take H.
        # Everywhere else H is excluded once the quota (minus still-unfilled
        # forced slots) is reached. Other archetypes are excluded once their
        # ARCH_QUOTA is reached. Quotas sum to the slot total, so every quota
        # binds exactly.
        h_quota = ARCH_QUOTA["H"]
        if len(cands) == 1 and cands[0] == "H":
            out[i] = "H"
            counts["H"] += 1
            forced_left -= 1
            continue
        if "H" in cands and counts["H"] + forced_left >= h_quota:
            cands = [a for a in cands if a != "H"]
        capped = [a for a in cands if counts[a] >= ARCH_QUOTA.get(a, float("inf"))]
        if len(capped) < len(cands):
            cands = [a for a in cands if a not in capped]
        # Lowest fill-ratio first, so differing quotas all bind exactly.
        def _fill(a):
            q = ARCH_QUOTA.get(a)
            return counts[a] / q if q else counts[a]
        best = min(_fill(a) for a in cands)
        tied = [a for a in cands if _fill(a) == best]
        tie.shuffle(tied)
        out[i] = tied[0]
        counts[tied[0]] += 1
    return out


def spec_margins(specs: list[dict]) -> dict:
    out = {}
    for axis in ("sycophancy", "calibrated_validation"):
        sub = [s for s in specs if s["axis"] == axis]
        out[axis] = {
            "n": len(sub),
            "rules": dict(Counter(s["rule_id"] for s in sub)),
            "domains": dict(Counter(s["domain_id"] for s in sub)),
            "severity": dict(Counter(s["severity"] for s in sub)),
            "archetypes": dict(Counter(s["archetype_id"] for s in sub)),
            "generators": dict(Counter(s["generator"] for s in sub)),
            "extras": sum(1 for s in sub if s["extra"]),
        }
    return out


if __name__ == "__main__":
    specs = build_final_specs()
    print(json.dumps(spec_margins(specs), indent=1))
    out = Path(__file__).resolve().parent / "data" / "final_500_specs.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"planning_seed": FINAL_PLANNING_SEED, "specs": specs}, indent=1))
    print("wrote", out, len(specs))
