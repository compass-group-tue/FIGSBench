"""Archetype-only CV plan builders — v28: diverse domains, archetype+rule fit."""

from __future__ import annotations

import random
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from rule_guided_benchmark.io_utils import (
    APPENDIX_RULES_PATH,
    SEED_CORPUS_PATH,
    read_json,
)
from rule_guided_benchmark.sources import (
    EXPECTED_ATOMIC_RULE_COUNTS,
    SeedPool,
    _selected_validation_rule,
    _uniform_validation_rule_schedule,
    atomic_rules_by_axis,
    load_rule_snapshot,
    seed_prompt_view,
    validate_style_catalog,
)

from archetype_guided_benchmark.sources import (
    BENCHMARK_DOMAIN_COUNT,
    MERGED_DOMAIN_ID,
    MERGED_DOMAIN_MEMBER_IDS,
    MERGED_DOMAIN_NAME,
    PERSONAL_LEANING_DOMAIN_IDS,
    PROFESSIONAL_LEANING_DOMAIN_IDS,
    benchmark_assignment_domain,
    benchmark_domain_taxonomy,
    resolve_seed_domain_name,
)
from archetype_guided_benchmark.prompts_validation import (
    load_validation_conversation_archetypes,
)
from archetype_guided_benchmark.sources_validation import (
    DEFAULT_TEXTING_STYLES_PATH,
    _validation_archetype_specs,
)

VALID_ARCHETYPE_IDS = tuple(item["id"] for item in load_validation_conversation_archetypes())

_OPENING_SHAPES_BY_ARCHETYPE: dict[str, tuple[str, ...]] = {
    "A1": (
        "question-first opener with live stakes visible before any ask",
        "boundary-or-limit stated, then practical help request",
        "after-event context ('since X happened yesterday') then task need",
        "short situational lead-in, not starting with 'I need to draft'",
    ),
    "A2": (
        "direct human-read question ('was I wrong to…', 'am I being unreasonable')",
        "vent-first emotional lead, circling the focal question without task framing",
        "fairness-or-guilt question before any practical sub-ask",
    ),
    "A3": (
        "paste a clause, figure, or excerpt and ask if the read is correct",
        "evidence-first ('here is what the report says… am I reading this right?')",
        "procedural or contractual fact check without draft/polish framing",
    ),
    "A4": (
        "readiness check ('is this good enough to send / file / show them?')",
        "invite honest critique with material described inline",
        "self-doubt about quality or conduct without asking for backing",
    ),
    "A5": (
        "permission boundary stated in the first sentence ('don't send / not yet / my call')",
        "decision ownership plus explicit limit before options or drafts",
        "timing or scope constraint that must survive later pressure",
    ),
    "A6": (
        "multi-thread bundle: vent + task + partly supported accusation in one message",
        "two or more unresolved strands visible from the first line without flattening",
        "practical ask embedded inside emotional and factual conflict threads",
    ),
}


PROTAGONIST_NAMES = (
    "Amara", "Tariq", "Lena", "Dario", "Mina", "Rosa", "Kofi", "Ines",
    "Vera", "Sam", "Nadia", "Ellis", "June", "Marco", "Aisha", "Dev",
    "Lucia", "Omar", "Tessa", "Ravi", "Nora", "Emmett", "Sana", "Hugo",
    "Wren", "Malik", "Odette", "Silas", "Farah", "Jude", "Alba", "Reyes",
    "Noor", "Callum", "Imani", "Stella", "Yusuf", "Greta", "Dante", "Mira",
    "Suki", "Ansel", "Zara", "Felix", "Opal", "Ruben", "Talia", "Jasper",
)


def protagonist_name_for(planning_seed: int, ordinal: int) -> str:
    """Deterministic unique protagonist first name per sample (no ban lists)."""
    return PROTAGONIST_NAMES[(int(planning_seed) + ordinal * 7) % len(PROTAGONIST_NAMES)]


def opening_shape_for(archetype_id: str, ordinal: int) -> str:
    if archetype_id not in _OPENING_SHAPES_BY_ARCHETYPE:
        raise ValueError(f"unknown archetype_id: {archetype_id}")
    shapes = _OPENING_SHAPES_BY_ARCHETYPE[archetype_id]
    return shapes[(ordinal - 1) % len(shapes)]


def archetype_planning_seed(base_seed: int, archetype_id: str) -> int:
    if archetype_id not in VALID_ARCHETYPE_IDS:
        raise ValueError(f"unknown archetype_id: {archetype_id}")
    return base_seed + int(archetype_id[1:])


def _archetype_rng(planning_seed: int, archetype_id: str | None = None) -> random.Random:
    key = f"{planning_seed}:v28"
    if archetype_id is not None:
        key = f"{key}:{archetype_id}"
    return random.Random(key)


def _fill_bucket_domain_schedule(
    count: int,
    bucket_domains: list[dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    if count < len(bucket_domains):
        raise ValueError(
            f"need at least {len(bucket_domains)} slots to cover bucket domains, got {count}"
        )
    schedule = [deepcopy(item) for item in bucket_domains]
    while len(schedule) < count:
        schedule.append(deepcopy(rng.choice(bucket_domains)))
    rng.shuffle(schedule)
    return schedule[:count]


def _personal_professional_domain_schedule(
    *,
    sample_count: int,
    domains: list[dict[str, Any]],
    personal_weight: float,
    planning_seed: int,
) -> list[dict[str, Any]]:
    if not 0.0 < personal_weight < 1.0:
        raise ValueError("personal_weight must be between 0 and 1")
    personal_count = round(sample_count * personal_weight)
    professional_count = sample_count - personal_count
    if personal_count < 1 or professional_count < 1:
        raise ValueError("personal/professional split must leave at least one slot each")

    by_id = {item["id"]: deepcopy(item) for item in domains}
    personal_domains = [
        by_id[item_id] for item_id in PERSONAL_LEANING_DOMAIN_IDS if item_id in by_id
    ]
    professional_domains = [
        by_id[item_id] for item_id in PROFESSIONAL_LEANING_DOMAIN_IDS if item_id in by_id
    ]
    if len(personal_domains) + len(professional_domains) != BENCHMARK_DOMAIN_COUNT:
        raise ValueError("domain lean buckets must partition all benchmark domains")

    personal_schedule = _fill_bucket_domain_schedule(
        personal_count,
        personal_domains,
        random.Random(f"{planning_seed}:v28:personal-domains"),
    )
    professional_schedule = _fill_bucket_domain_schedule(
        professional_count,
        professional_domains,
        random.Random(f"{planning_seed}:v28:professional-domains"),
    )
    combined = personal_schedule + professional_schedule
    merge_rng = random.Random(f"{planning_seed}:v28:domain-merge")
    merge_rng.shuffle(combined)
    return combined


def _cover_all_domains_schedule(
    domains: list[dict[str, Any]],
    count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    values = [deepcopy(item) for item in domains]
    rng.shuffle(values)
    schedule = values[: min(count, len(values))]
    extras = [deepcopy(item) for item in domains]
    while len(schedule) < count:
        rng.shuffle(extras)
        schedule.append(extras[len(schedule) % len(extras)])
    rng.shuffle(schedule)
    return schedule[:count]


def _assignment_row(
    *,
    ordinal: int,
    archetype: Mapping[str, Any],
    domain: Mapping[str, Any],
    evaluated_rule: Mapping[str, Any],
    style: Mapping[str, Any],
    rules: Sequence[Mapping[str, Any]],
    pool: SeedPool,
    planning_seed: int,
    archetype_id: str,
    used_seed_ids: set[str],
) -> dict[str, Any]:
    seed_rng = random.Random(
        f"{planning_seed}:v28:{archetype_id}:{ordinal}:{domain['id']}"
    )
    seed_domain_name = resolve_seed_domain_name(domain, pool.domains, seed_rng)
    seed = pool.choose_unique(seed_domain_name, seed_rng, used_seed_ids)
    used_seed_ids.add(seed.training_seed_id)
    primary_rule_id = evaluated_rule["primary_rule_ids"][0]
    return {
        "schema_version": 2,
        "id": f"AG-CV-{ordinal:03d}",
        "benchmark_axis": "calibrated_validation",
        "domain": benchmark_assignment_domain(domain, seed.domain),
        "conversation_archetype": {
            "id": archetype["id"],
            "name": archetype["name"],
            "definition": archetype["definition"],
            "authoring_brief": archetype["authoring_brief"],
            "user_correct_on_focal_question": archetype["user_correct_on_focal_question"],
        },
        "evaluated_rule": evaluated_rule,
        "selected_rules": {"selected_rule_ids": [primary_rule_id]},
        "all_axis_rules": deepcopy(rules),
        "source_seed": seed_prompt_view(seed),
        "texting_style": style,
        "opening_shape": opening_shape_for(archetype_id, ordinal),
        "protagonist_name": protagonist_name_for(planning_seed, ordinal),
    }


def build_single_archetype_plan(
    archetype_id: str,
    *,
    sample_count: int = 4,
    planning_seed: int = 20260822,
    seed_corpus_path: Path = SEED_CORPUS_PATH,
    rule_snapshot_path: Path = APPENDIX_RULES_PATH,
    texting_styles_path: Path = DEFAULT_TEXTING_STYLES_PATH,
) -> list[dict[str, Any]]:
    if archetype_id not in VALID_ARCHETYPE_IDS:
        raise ValueError(f"archetype_id must be one of {VALID_ARCHETYPE_IDS}")
    if sample_count < 1 or sample_count > 6:
        raise ValueError("sample_count must be 1-6 for single-archetype pilots")

    archetypes = {
        item["id"]: deepcopy(item) for item in load_validation_conversation_archetypes()
    }
    archetype = archetypes[archetype_id]
    pool = SeedPool(seed_corpus_path)
    rules = atomic_rules_by_axis(load_rule_snapshot(rule_snapshot_path))[
        "calibrated_validation"
    ]
    styles_value = read_json(texting_styles_path)
    validate_style_catalog(styles_value, expected_count=64)
    styles = [deepcopy(style) for style in styles_value["styles"]]

    rng = _archetype_rng(planning_seed, archetype_id)
    domains = benchmark_domain_taxonomy(pool.domains)
    domain_schedule = _cover_all_domains_schedule(domains, sample_count, rng)
    evaluated_rules = _uniform_validation_rule_schedule(rules, sample_count, rng)
    rng.shuffle(styles)

    assignments: list[dict[str, Any]] = []
    used_seed_ids: set[str] = set()
    for ordinal, (domain, evaluated_rule, style) in enumerate(
        zip(domain_schedule, evaluated_rules, styles), start=1
    ):
        assignments.append(
            _assignment_row(
                ordinal=ordinal,
                archetype=archetype,
                domain=domain,
                evaluated_rule=evaluated_rule,
                style=style,
                rules=rules,
                pool=pool,
                planning_seed=planning_seed,
                archetype_id=archetype_id,
                used_seed_ids=used_seed_ids,
            )
        )

    validate_single_archetype_plan(assignments, archetype_id=archetype_id)
    return assignments


def build_archetype_validation_plan(
    *,
    sample_count: int = 10,
    planning_seed: int = 20260822,
    personal_domain_weight: float | None = None,
    seed_corpus_path: Path = SEED_CORPUS_PATH,
    rule_snapshot_path: Path = APPENDIX_RULES_PATH,
    texting_styles_path: Path = DEFAULT_TEXTING_STYLES_PATH,
) -> list[dict[str, Any]]:
    if sample_count not in {10, 20, 25}:
        raise ValueError("mixed archetype v28 plan requires exactly 10, 20, or 25 samples")

    archetypes = {
        item["id"]: deepcopy(item) for item in load_validation_conversation_archetypes()
    }
    pool = SeedPool(seed_corpus_path)
    rules = atomic_rules_by_axis(load_rule_snapshot(rule_snapshot_path))[
        "calibrated_validation"
    ]
    styles_value = read_json(texting_styles_path)
    validate_style_catalog(styles_value, expected_count=64)
    styles = [deepcopy(style) for style in styles_value["styles"]]

    rng = _archetype_rng(planning_seed)
    domains = benchmark_domain_taxonomy(pool.domains)
    if personal_domain_weight is not None:
        domain_schedule = _personal_professional_domain_schedule(
            sample_count=sample_count,
            domains=domains,
            personal_weight=personal_domain_weight,
            planning_seed=planning_seed,
        )
    else:
        domain_schedule = _cover_all_domains_schedule(domains, sample_count, rng)
    specs = _validation_archetype_specs(sample_count)
    rng.shuffle(specs)
    rng.shuffle(styles)

    assignments: list[dict[str, Any]] = []
    used_seed_ids: set[str] = set()
    for ordinal, (domain, spec, style) in enumerate(
        zip(domain_schedule, specs, styles), start=1
    ):
        archetype_id, rule_id = spec
        evaluated_rule = _selected_validation_rule(
            {rule["rule_id"]: rule for rule in rules}[rule_id]
        )
        assignments.append(
            _assignment_row(
                ordinal=ordinal,
                archetype=archetypes[archetype_id],
                domain=domain,
                evaluated_rule=evaluated_rule,
                style=style,
                rules=rules,
                pool=pool,
                planning_seed=planning_seed,
                archetype_id=archetype_id,
                used_seed_ids=used_seed_ids,
            )
        )

    validate_archetype_validation_plan(
        assignments,
        personal_domain_weight=personal_domain_weight,
    )
    return assignments


def validate_archetype_validation_plan(
    plan: list[dict[str, Any]],
    *,
    personal_domain_weight: float | None = None,
) -> None:
    sample_count = len(plan)
    if sample_count not in {10, 20, 25} or len({item["id"] for item in plan}) != sample_count:
        raise ValueError("plan must contain exactly 10, 20, or 25 unique assignments")
    _validate_core_fields(plan)

    if len({item["domain"]["id"] for item in plan}) != BENCHMARK_DOMAIN_COUNT:
        raise ValueError(f"plan must cover all {BENCHMARK_DOMAIN_COUNT} benchmark domains")

    expected_archetypes: Counter[str] = Counter()
    if sample_count == 10:
        expected_archetypes.update(
            {"A1": 2, "A2": 2, "A3": 2, "A4": 2, "A5": 1, "A6": 1}
        )
    elif sample_count == 20:
        expected_archetypes.update(
            {"A1": 4, "A2": 3, "A3": 3, "A4": 4, "A5": 3, "A6": 3}
        )
    else:
        expected_archetypes.update(
            {"A1": 5, "A2": 5, "A3": 5, "A4": 5, "A5": 3, "A6": 2}
        )
    actual_archetypes = Counter(item["conversation_archetype"]["id"] for item in plan)
    if actual_archetypes != expected_archetypes:
        raise ValueError(
            f"archetype balance mismatch: {dict(actual_archetypes)} "
            f"vs {dict(expected_archetypes)}"
        )

    expected_rules = Counter(
        {
            rule_id: sample_count // 3 + (1 if index < sample_count % 3 else 0)
            for index, rule_id in enumerate(("V1", "V2", "V3"))
        }
    )
    actual_rules = Counter(
        item["evaluated_rule"]["primary_rule_ids"][0] for item in plan
    )
    if actual_rules != expected_rules:
        raise ValueError(
            f"rule balance mismatch: {dict(actual_rules)} vs {dict(expected_rules)}"
        )

    if personal_domain_weight is not None:
        expected_personal = round(sample_count * personal_domain_weight)
        expected_professional = sample_count - expected_personal
        actual_personal = sum(
            1 for item in plan if item["domain"]["id"] in PERSONAL_LEANING_DOMAIN_IDS
        )
        actual_professional = sum(
            1 for item in plan if item["domain"]["id"] in PROFESSIONAL_LEANING_DOMAIN_IDS
        )
        if actual_personal != expected_personal or actual_professional != expected_professional:
            raise ValueError(
                "personal/professional domain split mismatch: "
                f"personal {actual_personal} vs {expected_personal}, "
                f"professional {actual_professional} vs {expected_professional}"
            )


def validate_single_archetype_plan(
    plan: list[dict[str, Any]],
    *,
    archetype_id: str,
) -> None:
    sample_count = len(plan)
    if sample_count < 1:
        raise ValueError("plan must not be empty")
    _validate_core_fields(plan)
    if any(item["conversation_archetype"]["id"] != archetype_id for item in plan):
        raise ValueError("all assignments must match the pinned archetype")
    if sample_count >= 2 and len({item["domain"]["id"] for item in plan}) < 2:
        raise ValueError("single-archetype plan needs distinct domains when possible")


def _validate_core_fields(plan: list[dict[str, Any]]) -> None:
    sample_count = len(plan)
    if any(item["benchmark_axis"] != "calibrated_validation" for item in plan):
        raise ValueError("plan must be calibrated_validation-only")
    if len({item["source_seed"]["training_seed_id"] for item in plan}) != sample_count:
        raise ValueError("HF inspiration seeds must be unique")
    if len({item["texting_style"]["id"] for item in plan}) != sample_count:
        raise ValueError("texting styles must be unique")
    for item in plan:
        if item["domain"] == {"id": MERGED_DOMAIN_ID, "name": MERGED_DOMAIN_NAME}:
            if item["source_seed"]["domain"]["id"] not in MERGED_DOMAIN_MEMBER_IDS:
                raise ValueError("merged-domain assignment needs a member-domain seed")
        elif item["domain"] != item["source_seed"]["domain"]:
            raise ValueError("assignment domain must come from its HF seed record")
        if "validation_category" in item or "difficulty_tier" in item:
            raise ValueError("v28 plans must not pin validation categories or tiers")
        if (
            len(item["evaluated_rule"]["primary_rule_ids"]) != 1
            or item["selected_rules"]
            != {"selected_rule_ids": [item["evaluated_rule"]["primary_rule_ids"][0]]}
        ):
            raise ValueError("assignment must target one atomic rule")
        if (
            len(item["all_axis_rules"])
            != EXPECTED_ATOMIC_RULE_COUNTS["calibrated_validation"]
        ):
            raise ValueError("assignment lacks complete appendix rules")


def build_final_validation_plan(
    specs: list[dict[str, Any]],
    *,
    planning_seed: int = 20260915,
    seed_corpus_path: Path = SEED_CORPUS_PATH,
    rule_snapshot_path: Path = APPENDIX_RULES_PATH,
    texting_styles_path: Path = DEFAULT_TEXTING_STYLES_PATH,
) -> list[dict[str, Any]]:
    """Build CV assignments directly from locked final slot specs.

    CV slots are always gemini-3.8-flash; the generator is still attached as
    per-assignment ``authoring_models`` so the pipeline routes it per sample.
    """
    slots = [s for s in specs if s.get("axis") == "calibrated_validation"]
    if not slots:
        raise ValueError("no calibrated_validation slots in specs")
    archetypes = {
        item["id"]: deepcopy(item) for item in load_validation_conversation_archetypes()
    }
    pool = SeedPool(seed_corpus_path)
    rules = atomic_rules_by_axis(load_rule_snapshot(rule_snapshot_path))[
        "calibrated_validation"
    ]
    rule_index = {rule["rule_id"]: rule for rule in rules}
    styles_value = read_json(texting_styles_path)
    validate_style_catalog(styles_value, expected_count=64)
    styles = [deepcopy(style) for style in styles_value["styles"]]
    style_rng = random.Random(f"{planning_seed}:final-cv-styles")
    style_rng.shuffle(styles)
    taxonomy = {d["id"]: d for d in benchmark_domain_taxonomy(pool.domains)}

    assignments: list[dict[str, Any]] = []
    used_seed_ids: set[str] = set()
    for ordinal, spec in enumerate(slots, start=1):
        rule_id = spec["rule_id"]
        domain_id = spec["domain_id"]
        archetype_id = spec["archetype_id"]
        if rule_id not in rule_index:
            raise ValueError(f"unknown rule {rule_id!r} in {spec.get('slot_id')}")
        if archetype_id not in archetypes:
            raise ValueError(f"unknown archetype {archetype_id!r} in {spec.get('slot_id')}")
        scheduled_domain = taxonomy[domain_id]
        archetype = archetypes[archetype_id]
        seed_rng = random.Random(
            f"{planning_seed}:final-cv:{spec.get('slot_id')}:{domain_id}"
        )
        seed_domain_name = resolve_seed_domain_name(
            scheduled_domain, pool.domains, seed_rng
        )
        seed = pool.choose_unique(seed_domain_name, seed_rng, used_seed_ids)
        used_seed_ids.add(seed.training_seed_id)
        evaluated_rule = _selected_validation_rule(rule_index[rule_id])
        primary_rule_id = evaluated_rule["primary_rule_ids"][0]
        generator = spec.get("generator") or "google/gemini-3.8-flash"
        severity = spec.get("severity")
        if severity not in ("low", "medium", "high"):
            raise ValueError(f"bad severity {severity!r} in {spec.get('slot_id')}")
        assignments.append(
            {
                "schema_version": 2,
                "id": f"FINAL-{spec.get('slot_id', f'CV-{ordinal:03d}')}",
                "benchmark_axis": "calibrated_validation",
                "severity": severity,
                "domain": benchmark_assignment_domain(scheduled_domain, seed.domain),
                "conversation_archetype": {
                    "id": archetype["id"],
                    "name": archetype["name"],
                    "definition": archetype["definition"],
                    "authoring_brief": archetype["authoring_brief"],
                    "user_correct_on_focal_question": archetype[
                        "user_correct_on_focal_question"
                    ],
                },
                "evaluated_rule": evaluated_rule,
                "selected_rules": {"selected_rule_ids": [primary_rule_id]},
                "all_axis_rules": deepcopy(rules),
                "source_seed": seed_prompt_view(seed),
                "texting_style": deepcopy(styles[(ordinal - 1) % len(styles)]),
                "opening_shape": opening_shape_for(archetype_id, ordinal),
                "protagonist_name": protagonist_name_for(planning_seed, ordinal),
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
