"""Plan builder for archetype-guided sycophancy samples."""

from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path
from typing import Any

from rule_guided_benchmark.io_utils import (
    APPENDIX_RULES_PATH,
    PROJECT_ROOT,
    SEED_CORPUS_PATH,
    read_json,
)
from rule_guided_benchmark.sources import (
    EXPECTED_ATOMIC_RULE_COUNTS,
    SeedPool,
    _selected_syc_rule,
    atomic_rules_by_axis,
    load_rule_snapshot,
    seed_prompt_view,
    validate_style_catalog,
)

from .prompts import load_conversation_archetypes
from .severity import assign_severities, normalize_severity

ARCHETYPE_IDS = tuple(item["id"] for item in load_conversation_archetypes())

_PRESSURE_VECTORS_PATH = PROJECT_ROOT / "benchmark" / "config" / "attack_vectors.json"

# Balanced legacy pressure schedule (2× each); MH/AL/RL drove most Luna failures.
_PRESSURE_PATTERN_SCHEDULE_10 = (
    "MH", "SR", "AL", "RL", "PS", "MH", "AL", "RL", "SR", "PS"
)


def load_pressure_patterns() -> dict[str, dict[str, Any]]:
    data = read_json(_PRESSURE_VECTORS_PATH)
    if not isinstance(data, dict):
        raise ValueError("attack_vectors.json must be an object")
    return deepcopy(data)

DEFAULT_TEXTING_STYLES_PATH = (
    PROJECT_ROOT
    / "sft_data"
    / "data"
    / "runs"
    / "sft-150-hybrid-20260812"
    / "texting_styles.json"
)


NEW_ARCHETYPE_RULES = {"I": "S1.a", "J": "S2.b", "K": "S2.b", "L": "S1.a", "M": "S1.a"}
NEW_ARCHETYPE_ORDER = ("I", "J", "K", "L", "M")


def new_archetype_rotation(planning_seed: int) -> tuple[str, str]:
    """Two of I/J/K/L/M for 10-count plans, rotated deterministically by seed."""
    start = int(planning_seed) % len(NEW_ARCHETYPE_ORDER)
    return (
        NEW_ARCHETYPE_ORDER[start],
        NEW_ARCHETYPE_ORDER[(start + 1) % len(NEW_ARCHETYPE_ORDER)],
    )


def _archetype_rule_specs(
    sample_count: int, planning_seed: int = 0
) -> list[tuple[str, str, str]]:
    """Return (archetype_id, rule_id, scenario_context) specs."""

    personal = "personal_or_everyday"
    general = "professional_or_public"
    if sample_count == 10:
        rows = [
            ("A", "S2.a", personal),
            ("B", "S2.b", personal),
            ("C", "S2.c", personal),
            ("D", "S2.d", personal),
            ("E", "S1.a", personal),
            ("F", "S2.b", personal),
            ("G", "S1.a", personal),
            ("H", "S2.b", personal),
        ]
        for new_id in new_archetype_rotation(planning_seed):
            rows.append((new_id, NEW_ARCHETYPE_RULES[new_id], general))
    elif sample_count == 20:
        rows = [
            ("A", "S2.a", personal),
            ("A", "S2.c", general),
            ("A", "S1.c", personal),
            ("B", "S2.b", general),
            ("B", "S2.b", personal),
            ("B", "S1.c", general),
            ("C", "S2.c", personal),
            ("C", "S2.d", general),
            ("C", "S1.a", personal),
            ("D", "S2.a", general),
            ("D", "S2.d", personal),
            ("E", "S1.a", general),
            ("E", "S2.c", personal),
            ("E", "S1.c", general),
            ("F", "S2.b", personal),
            ("F", "S2.c", general),
            ("G", "S1.a", general),
            ("G", "S2.c", personal),
            ("H", "S2.b", general),
            ("H", "S1.a", personal),
        ]
    else:
        raise ValueError("archetype plan requires exactly 10 or 20 samples")
    return rows


def _homogeneous_archetype_rule_specs(
    archetype_id: str,
    sample_count: int,
) -> list[tuple[str, str, str]]:
    """Return (archetype_id, rule_id, scenario_context) for a single-archetype pack."""

    if archetype_id not in ARCHETYPE_IDS:
        raise ValueError(f"unknown archetype id: {archetype_id}")
    personal = "personal_or_everyday"
    general = "professional_or_public"
    if sample_count == 5:
        rule_cycle = ["S1.a", "S1.c", "S2.a", "S2.b", "S1.a"]
        contexts = [personal] * 4 + [general] * 1
    elif sample_count == 10:
        rule_cycle = [
            "S1.a",
            "S1.c",
            "S2.a",
            "S2.b",
            "S2.c",
            "S2.d",
            "S1.a",
            "S2.b",
            "S2.c",
            "S2.d",
        ]
        contexts = [personal] * 8 + [general] * 2
    elif sample_count == 20:
        rule_cycle = [
            "S1.a",
            "S1.c",
            "S2.a",
            "S2.b",
            "S2.c",
            "S2.d",
            "S1.a",
            "S1.c",
            "S2.a",
            "S2.b",
            "S2.c",
            "S2.d",
            "S1.a",
            "S2.b",
            "S2.c",
            "S2.d",
            "S1.c",
            "S2.a",
            "S2.b",
            "S2.c",
        ]
        contexts = [personal] * 10 + [general] * 10
    else:
        raise ValueError("homogeneous archetype plan requires exactly 5, 10, or 20 samples")
    return [
        (archetype_id, rule_id, scenario_context)
        for rule_id, scenario_context in zip(rule_cycle, contexts, strict=True)
    ]


def _archetype_grid_rule_specs(per_archetype_count: int = 5) -> list[tuple[str, str]]:
    """Return (archetype_id, rule_id) specs for a balanced grid: each archetype A–H × count."""

    if per_archetype_count not in {5, 10, 20}:
        raise ValueError("archetype grid requires 5, 10, or 20 samples per archetype")
    rows: list[tuple[str, str]] = []
    for archetype_id in ARCHETYPE_IDS:
        for archetype_id, rule_id, _scenario_context in _homogeneous_archetype_rule_specs(
            archetype_id, per_archetype_count
        ):
            rows.append((archetype_id, rule_id))
    return rows


def _pressure_pattern_schedule(sample_count: int) -> list[str]:
    base = list(_PRESSURE_PATTERN_SCHEDULE_10)
    if sample_count <= len(base):
        return base[:sample_count]
    reps = (sample_count + len(base) - 1) // len(base)
    return (base * reps)[:sample_count]


PERSONAL_LEANING_DOMAIN_IDS = frozenset(
    {
        "family_friends_social_life",
        "intimate_relationships",
        "beliefs_culture_society",
    }
)
PROFESSIONAL_LEANING_DOMAIN_IDS = frozenset(
    {
        "work_careers_organizations",
        "science_technology",
        "health_finance_law",
    }
)
ALL_LEANING_DOMAIN_IDS = PERSONAL_LEANING_DOMAIN_IDS | PROFESSIONAL_LEANING_DOMAIN_IDS

MERGED_DOMAIN_ID = "health_finance_law"
MERGED_DOMAIN_NAME = "Health, Finance & Law"
MERGED_DOMAIN_MEMBER_IDS = ("health_medicine", "finance_law")
BENCHMARK_DOMAIN_COUNT = 6


def benchmark_domain_taxonomy(pool_domains):
    by_id = {item["id"]: item for item in pool_domains}
    taxonomy = [
        deepcopy(by_id[item_id])
        for item_id in sorted(by_id)
        if item_id not in MERGED_DOMAIN_MEMBER_IDS
    ]
    taxonomy.append({"id": MERGED_DOMAIN_ID, "name": MERGED_DOMAIN_NAME})
    return taxonomy


def resolve_seed_domain_name(scheduled_domain, pool_domains, rng):
    if scheduled_domain["id"] != MERGED_DOMAIN_ID:
        return scheduled_domain["name"]
    members = [item for item in pool_domains if item["id"] in MERGED_DOMAIN_MEMBER_IDS]
    return rng.choice(members)["name"]


def benchmark_assignment_domain(scheduled_domain, seed_domain):
    if scheduled_domain["id"] != MERGED_DOMAIN_ID:
        return deepcopy(seed_domain)
    if seed_domain["id"] not in MERGED_DOMAIN_MEMBER_IDS:
        raise ValueError("merged-domain seed must come from a member domain")
    return {"id": MERGED_DOMAIN_ID, "name": MERGED_DOMAIN_NAME}

# Scenario framing split for archetype grid packs (e.g. 40 → 28 personal, 12 professional).
PERSONAL_CONTEXT_FRACTION = 0.7


def _balanced_domain_schedule(
    domains: list[dict[str, str]],
    count: int,
    prefer_ids: frozenset[str],
    rng: random.Random,
) -> list[dict[str, str]]:
    by_id = {item["id"]: deepcopy(item) for item in domains}
    pool = [by_id[item_id] for item_id in sorted(prefer_ids) if item_id in by_id]
    if not pool:
        pool = list(by_id.values())
    base = count // len(pool)
    remainder = count % len(pool)
    schedule: list[dict[str, str]] = []
    for index, domain in enumerate(pool):
        schedule.extend([deepcopy(domain)] * (base + (1 if index < remainder else 0)))
    rng.shuffle(schedule)
    return schedule


def _allocate_domain_schedule(
    count: int,
    domains: list[dict[str, str]],
    *,
    prefer_ids: frozenset[str],
    rng: random.Random,
    cover_all_domains: bool,
) -> list[dict[str, str]]:
    by_id = {item["id"]: deepcopy(item) for item in domains}
    prefer = [by_id[item_id] for item_id in prefer_ids if item_id in by_id]
    if not prefer:
        prefer = list(by_id.values())

    schedule: list[dict[str, str]] = []
    if cover_all_domains:
        schedule.extend(by_id.values())
        while len(schedule) < count:
            schedule.append(deepcopy(rng.choice(prefer)))
    else:
        pool = [by_id[item_id] for item_id in prefer_ids if item_id in by_id]
        if not pool:
            pool = list(by_id.values())
        for _ in range(count):
            schedule.append(deepcopy(rng.choice(pool)))
    rng.shuffle(schedule)
    return schedule[:count]


def build_archetype_sycophancy_plan(
    *,
    sample_count: int = 10,
    planning_seed: int = 20260815,
    seed_corpus_path: Path = SEED_CORPUS_PATH,
    rule_snapshot_path: Path = APPENDIX_RULES_PATH,
    texting_styles_path: Path = DEFAULT_TEXTING_STYLES_PATH,
    pin_pressure_patterns: bool = True,
    fixed_archetype_id: str | None = None,
    archetype_grid: bool = False,
    severity: str = "mixed",
) -> list[dict[str, Any]]:
    severity_levels = assign_severities(sample_count, severity, planning_seed)
    if archetype_grid and fixed_archetype_id is not None:
        raise ValueError("archetype_grid cannot be combined with fixed_archetype_id")
    if fixed_archetype_id is not None and fixed_archetype_id not in ARCHETYPE_IDS:
        raise ValueError(f"unknown fixed_archetype_id: {fixed_archetype_id}")
    archetypes = {item["id"]: deepcopy(item) for item in load_conversation_archetypes()}
    pool = SeedPool(seed_corpus_path)
    rules = atomic_rules_by_axis(load_rule_snapshot(rule_snapshot_path))["sycophancy"]
    rule_index = {rule["rule_id"]: rule for rule in rules}
    styles_value = read_json(texting_styles_path)
    validate_style_catalog(styles_value, expected_count=64)
    styles = [deepcopy(style) for style in styles_value["styles"]]

    rng = random.Random(planning_seed)
    if archetype_grid:
        per_archetype_count = sample_count // len(ARCHETYPE_IDS)
        if sample_count != per_archetype_count * len(ARCHETYPE_IDS):
            raise ValueError(
                f"archetype grid sample_count must be a multiple of {len(ARCHETYPE_IDS)}"
            )
        rule_specs = _archetype_grid_rule_specs(per_archetype_count)
        rng.shuffle(rule_specs)
        personal_count = int(round(sample_count * PERSONAL_CONTEXT_FRACTION))
        professional_count = sample_count - personal_count
        context_rng = random.Random(f"{planning_seed}:grid-contexts")
        contexts = (
            ["personal_or_everyday"] * personal_count
            + ["professional_or_public"] * professional_count
        )
        context_rng.shuffle(contexts)
        specs = [
            (archetype_id, rule_id, scenario_context)
            for (archetype_id, rule_id), scenario_context in zip(
                rule_specs, contexts, strict=True
            )
        ]
    elif fixed_archetype_id is not None:
        specs = _homogeneous_archetype_rule_specs(fixed_archetype_id, sample_count)
        rng.shuffle(specs)
    else:
        specs = _archetype_rule_specs(sample_count, planning_seed)
        rng.shuffle(specs)

    personal_count = sum(1 for _, _, ctx in specs if ctx == "personal_or_everyday")
    professional_count = sample_count - personal_count
    domains = benchmark_domain_taxonomy(pool.domains)
    homogeneous_small_pack = fixed_archetype_id is not None and sample_count < 10
    if homogeneous_small_pack:
        by_id = {item["id"]: deepcopy(item) for item in domains}
        prefer_list = [by_id[item_id] for item_id in ALL_LEANING_DOMAIN_IDS if item_id in by_id]
        domain_rng = random.Random(f"{planning_seed}:unified-domains")
        domain_rng.shuffle(prefer_list)
        if len(prefer_list) < sample_count:
            raise ValueError(
                f"need {sample_count} unique leaning domains, found {len(prefer_list)}"
            )
        unified_domain_schedule = prefer_list[:sample_count]
        domain_iter_unified = iter(unified_domain_schedule)
        domain_iter_personal = None
        domain_iter_professional = None
    else:
        personal_domain_schedule = (
            _balanced_domain_schedule(
                domains,
                personal_count,
                PERSONAL_LEANING_DOMAIN_IDS,
                rng=random.Random(f"{planning_seed}:personal-domains"),
            )
            if archetype_grid
            else _allocate_domain_schedule(
                personal_count,
                domains,
                prefer_ids=PERSONAL_LEANING_DOMAIN_IDS,
                rng=random.Random(f"{planning_seed}:personal-domains"),
                cover_all_domains=True,
            )
        )
        professional_domain_schedule = (
            _balanced_domain_schedule(
                domains,
                professional_count,
                PROFESSIONAL_LEANING_DOMAIN_IDS,
                rng=random.Random(f"{planning_seed}:professional-domains"),
            )
            if archetype_grid
            else _allocate_domain_schedule(
                professional_count,
                domains,
                prefer_ids=PROFESSIONAL_LEANING_DOMAIN_IDS,
                rng=random.Random(f"{planning_seed}:professional-domains"),
                cover_all_domains=False,
            )
        )
        domain_iter_personal = iter(personal_domain_schedule)
        domain_iter_professional = iter(professional_domain_schedule)
        domain_iter_unified = None
    pressure_ids: list[str | None]
    if pin_pressure_patterns:
        pressure_ids = _pressure_pattern_schedule(sample_count)
        if len(pressure_ids) != sample_count:
            raise ValueError("pressure pattern schedule length mismatch")
        pressure_catalog = load_pressure_patterns()
    else:
        pressure_ids = [None] * sample_count
        pressure_catalog = {}
    rng.shuffle(styles)

    assignments: list[dict[str, Any]] = []
    used_seed_ids: set[str] = set()
    for ordinal, ((archetype_id, rule_id, scenario_context), style, pressure_id) in enumerate(
        zip(specs, styles, pressure_ids), start=1
    ):
        if homogeneous_small_pack:
            scheduled_domain = next(domain_iter_unified)
        elif scenario_context == "personal_or_everyday":
            scheduled_domain = next(domain_iter_personal)
        else:
            scheduled_domain = next(domain_iter_professional)
        seed_rng = random.Random(
            f"{planning_seed}:archetype-syc:{ordinal}:{scheduled_domain['id']}"
        )
        seed_domain_name = resolve_seed_domain_name(
            scheduled_domain, pool.domains, seed_rng
        )
        seed = pool.choose_unique(seed_domain_name, seed_rng, used_seed_ids)
        used_seed_ids.add(seed.training_seed_id)
        evaluated_rule = _selected_syc_rule(rule_index[rule_id])
        primary_rule_id = evaluated_rule["primary_rule_ids"][0]
        archetype = archetypes[archetype_id]
        assignment: dict[str, Any] = {
            "schema_version": 2,
            "id": f"AG-SYC-{ordinal:03d}",
            "benchmark_axis": "sycophancy",
            "severity": severity_levels[ordinal - 1],
            "domain": benchmark_assignment_domain(scheduled_domain, seed.domain),
            "conversation_archetype": {
                "id": archetype["id"],
                "name": archetype["name"],
                "definition": archetype["definition"],
                "authoring_brief": archetype["authoring_brief"],
            },
            "scenario_context": scenario_context,
            "evaluated_rule": evaluated_rule,
            "selected_rules": {"selected_rule_ids": [primary_rule_id]},
            "all_axis_rules": deepcopy(rules),
            "source_seed": seed_prompt_view(seed),
            "texting_style": style,
        }
        if pin_pressure_patterns and pressure_id is not None:
            try:
                pressure_vector = pressure_catalog[pressure_id]
            except KeyError as exc:
                raise ValueError(f"unknown pressure pattern id: {pressure_id}") from exc
            assignment["pressure_pattern"] = {
                "id": pressure_id,
                "name": str(pressure_vector["name"]),
                "definition": str(pressure_vector["definition"]),
                "mechanism": str(pressure_vector.get("mechanism", "")),
            }
        assignments.append(assignment)

    validate_archetype_sycophancy_plan(
        assignments,
        fixed_archetype_id=fixed_archetype_id,
        archetype_grid=archetype_grid,
        severity=severity,
        planning_seed=planning_seed,
    )
    return assignments


def validate_archetype_sycophancy_plan(
    plan: list[dict[str, Any]],
    *,
    fixed_archetype_id: str | None = None,
    archetype_grid: bool = False,
    severity: str = "mixed",
    planning_seed: int = 0,
) -> None:
    from collections import Counter

    sample_count = len(plan)
    allowed_counts = {5, 10, 20, 40}
    if sample_count not in allowed_counts or len({item["id"] for item in plan}) != sample_count:
        raise ValueError("plan must contain exactly 5, 10, 20, or 40 unique assignments")
    if any(item["benchmark_axis"] != "sycophancy" for item in plan):
        raise ValueError("plan must be sycophancy-only")
    for item in plan:
        normalize_severity(item.get("severity", "medium"))
    if severity != "mixed" and any(item.get("severity") != severity for item in plan):
        raise ValueError(f"plan must be uniform severity={severity}")
    if len({item["source_seed"]["training_seed_id"] for item in plan}) != sample_count:
        raise ValueError("HF inspiration seeds must be unique")
    if len({item["texting_style"]["id"] for item in plan}) != sample_count:
        raise ValueError("texting styles must be unique")
    if fixed_archetype_id and sample_count < 10:
        if len({item["domain"]["id"] for item in plan}) != sample_count:
            raise ValueError(
                f"homogeneous {sample_count}-sample plan must use {sample_count} unique domains"
            )
    elif archetype_grid:
        if len({item["domain"]["id"] for item in plan}) < BENCHMARK_DOMAIN_COUNT:
            raise ValueError(
                f"archetype grid plan must cover all {BENCHMARK_DOMAIN_COUNT} benchmark domains"
            )
    elif len({item["domain"]["id"] for item in plan}) != BENCHMARK_DOMAIN_COUNT:
        raise ValueError(f"plan must cover all {BENCHMARK_DOMAIN_COUNT} benchmark domains")

    actual = Counter(item["conversation_archetype"]["id"] for item in plan)
    if archetype_grid:
        per_archetype = sample_count // len(ARCHETYPE_IDS)
        expected_archetypes = Counter({archetype_id: per_archetype for archetype_id in ARCHETYPE_IDS})
        if actual != expected_archetypes:
            raise ValueError(
                f"archetype grid balance mismatch: {dict(actual)} vs {dict(expected_archetypes)}"
            )
    elif fixed_archetype_id is not None:
        if actual != Counter({fixed_archetype_id: sample_count}):
            raise ValueError(
                f"homogeneous archetype mismatch: {dict(actual)} "
                f"vs {fixed_archetype_id}×{sample_count}"
            )
    else:
        expected_archetypes = Counter()
        if sample_count == 10:
            for archetype_id in ("A", "B", "C", "D", "E", "F", "G", "H"):
                expected_archetypes[archetype_id] = 1
            for new_id in new_archetype_rotation(planning_seed):
                expected_archetypes[new_id] += 1
        else:
            for archetype_id in ("A", "B", "C", "E"):
                expected_archetypes[archetype_id] = 3
            for archetype_id in ("D", "F", "G", "H"):
                expected_archetypes[archetype_id] = 2
        if actual != expected_archetypes:
            raise ValueError(
                f"archetype balance mismatch: {dict(actual)} vs {dict(expected_archetypes)}"
            )

    expected_context = sample_count // 2
    if sample_count == 10:
        expected_context_counts = Counter(
            {
                "personal_or_everyday": 8,
                "professional_or_public": 2,
            }
        )
    elif sample_count == 5:
        expected_context_counts = Counter(
            {
                "personal_or_everyday": 4,
                "professional_or_public": 1,
            }
        )
    elif archetype_grid:
        personal_count = int(round(sample_count * PERSONAL_CONTEXT_FRACTION))
        expected_context_counts = Counter(
            {
                "personal_or_everyday": personal_count,
                "professional_or_public": sample_count - personal_count,
            }
        )
    else:
        expected_context_counts = Counter(
            {
                "personal_or_everyday": expected_context,
                "professional_or_public": expected_context,
            }
        )
    if Counter(item.get("scenario_context") for item in plan) != expected_context_counts:
        raise ValueError("plan must match personal/professional context split")

    if sample_count == 10 and any("pressure_pattern" in item for item in plan):
        expected_pressure = Counter(_PRESSURE_PATTERN_SCHEDULE_10)
        actual_pressure = Counter(item["pressure_pattern"]["id"] for item in plan)
        if actual_pressure != expected_pressure:
            raise ValueError(
                f"pressure pattern balance mismatch: {dict(actual_pressure)} "
                f"vs {dict(expected_pressure)}"
            )

    for item in plan:
        if item["domain"] == {
            "id": MERGED_DOMAIN_ID,
            "name": MERGED_DOMAIN_NAME,
        }:
            if item["source_seed"]["domain"]["id"] not in MERGED_DOMAIN_MEMBER_IDS:
                raise ValueError("merged-domain assignment needs a member-domain seed")
        elif item["domain"] != item["source_seed"]["domain"]:
            raise ValueError("assignment domain must come from its HF seed record")
        if item["evaluated_rule"]["axis"] != "sycophancy":
            raise ValueError("cross-axis rule in sycophancy assignment")
        if (
            len(item["evaluated_rule"]["primary_rule_ids"]) != 1
            or item["selected_rules"]
            != {"selected_rule_ids": [item["evaluated_rule"]["primary_rule_ids"][0]]}
        ):
            raise ValueError("assignment must target one atomic rule")
        if len(item["all_axis_rules"]) != EXPECTED_ATOMIC_RULE_COUNTS["sycophancy"]:
            raise ValueError("assignment lacks complete appendix rules")


def build_final_sycophancy_plan(
    specs: list[dict[str, Any]],
    *,
    planning_seed: int = 20260915,
    seed_corpus_path: Path = SEED_CORPUS_PATH,
    rule_snapshot_path: Path = APPENDIX_RULES_PATH,
    texting_styles_path: Path = DEFAULT_TEXTING_STYLES_PATH,
) -> list[dict[str, Any]]:
    """Build sycophancy assignments directly from locked final slot specs.

    Each spec carries rule/domain/severity/archetype/generator; the slot
    generator (random inkling vs gemini-3.8-flash per the locked plan, CV is
    always gemini) is attached as per-assignment ``authoring_models`` so the
    pipeline routes generator/refiner/auditor per sample.
    """
    slots = [s for s in specs if s.get("axis") == "sycophancy"]
    if not slots:
        raise ValueError("no sycophancy slots in specs")
    archetypes = {item["id"]: deepcopy(item) for item in load_conversation_archetypes()}
    pool = SeedPool(seed_corpus_path)
    rules = atomic_rules_by_axis(load_rule_snapshot(rule_snapshot_path))["sycophancy"]
    rule_index = {rule["rule_id"]: rule for rule in rules}
    styles_value = read_json(texting_styles_path)
    validate_style_catalog(styles_value, expected_count=64)
    styles = [deepcopy(style) for style in styles_value["styles"]]
    style_rng = random.Random(f"{planning_seed}:final-syc-styles")
    style_rng.shuffle(styles)
    taxonomy = {d["id"]: d for d in benchmark_domain_taxonomy(pool.domains)}

    assignments: list[dict[str, Any]] = []
    used_seed_ids: set[str] = set()
    for ordinal, spec in enumerate(slots, start=1):
        rule_id = spec["rule_id"]
        domain_id = spec["domain_id"]
        archetype_id = spec["archetype_id"]
        severity = spec["severity"]
        if severity not in ("low", "medium", "high"):
            raise ValueError(f"bad severity {severity!r} in {spec.get('slot_id')}")
        if rule_id not in rule_index:
            raise ValueError(f"unknown rule {rule_id!r} in {spec.get('slot_id')}")
        if archetype_id not in archetypes:
            raise ValueError(f"unknown archetype {archetype_id!r} in {spec.get('slot_id')}")
        scheduled_domain = taxonomy[domain_id]
        if domain_id in PERSONAL_LEANING_DOMAIN_IDS:
            scenario_context = "personal_or_everyday"
        elif domain_id in PROFESSIONAL_LEANING_DOMAIN_IDS:
            scenario_context = "professional_or_public"
        else:
            raise ValueError(f"domain {domain_id!r} has no context leaning")
        seed_rng = random.Random(
            f"{planning_seed}:final-syc:{spec.get('slot_id')}:{domain_id}"
        )
        seed_domain_name = resolve_seed_domain_name(
            scheduled_domain, pool.domains, seed_rng
        )
        seed = pool.choose_unique(seed_domain_name, seed_rng, used_seed_ids)
        used_seed_ids.add(seed.training_seed_id)
        # S1.b is a deliberate final-benchmark target (dedicated elicitation
        # block in prompts.py) though the legacy pilot helper rejects it.
        evaluated_rule = deepcopy(rule_index[rule_id])
        primary_rule_id = evaluated_rule["primary_rule_ids"][0]
        archetype = archetypes[archetype_id]
        generator = spec.get("generator") or "google/gemini-3.8-flash"
        assignments.append(
            {
                "schema_version": 2,
                "id": f"FINAL-{spec.get('slot_id', f'SYC-{ordinal:03d}')}",
                "benchmark_axis": "sycophancy",
                "severity": severity,
                "domain": benchmark_assignment_domain(scheduled_domain, seed.domain),
                "conversation_archetype": {
                    "id": archetype["id"],
                    "name": archetype["name"],
                    "definition": archetype["definition"],
                    "authoring_brief": archetype["authoring_brief"],
                },
                "scenario_context": scenario_context,
                "evaluated_rule": evaluated_rule,
                "selected_rules": {"selected_rule_ids": [primary_rule_id]},
                "all_axis_rules": deepcopy(rules),
                "source_seed": seed_prompt_view(seed),
                "texting_style": deepcopy(styles[(ordinal - 1) % len(styles)]),
                "rule_id": rule_id,
                "final_slot_id": spec.get("slot_id"),
                "authoring_models": {
                    "generator": generator,
                    "refiner": generator,
                    "auditor": generator,
                },
            }
        )
    return assignments
