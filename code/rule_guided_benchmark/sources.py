"""Load HF inspirations, appendix rules, and role-player texting styles."""

from __future__ import annotations

import random
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from sft_data.seeds import SeedPool, SeedRecord, HF_SEED_DOMAIN_COUNT
from sft_data.styles import validate_style_catalog

from .io_utils import (
    APPENDIX_RULES_PATH,
    SEED_CORPUS_PATH,
    TEXTING_STYLES_PATH,
    read_json,
)


AXES = ("sycophancy", "calibrated_validation")
AXIS_PREFIX = {"sycophancy": "SYC", "calibrated_validation": "CV"}
EXPECTED_ATOMIC_RULE_COUNTS = {"sycophancy": 7, "calibrated_validation": 3}
SYCOPHANCY_TARGET_RULE_IDS = (
    "S1.a",
    "S1.c",
    "S2.a",
    "S2.b",
    "S2.c",
    "S2.d",
)


def load_rule_snapshot(path: Path = APPENDIX_RULES_PATH) -> dict[str, Any]:
    value = read_json(path)
    required = {
        "schema_version",
        "rule_snapshot_version",
        "source",
        "constructs",
        "s_system",
        "validation_system",
        "authoring_invariants",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("appendix rule snapshot has the wrong top-level fields")
    return value


def _rule_view(rule: Mapping[str, Any], axis: str) -> dict[str, Any]:
    return {
        "rule_id": rule["id"],
        "chunk_id": f"appendix-v2:{str(rule['id']).lower()}",
        "title": rule["title"],
        "axis": axis,
        "primary_rule_ids": [rule["id"]],
        "description": rule["description"],
        "example": rule["example"],
        "neighboring_boundary": rule["neighboring_boundary"],
    }


def atomic_rules_by_axis(
    snapshot: Mapping[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    snapshot = snapshot or load_rule_snapshot()
    rules = {
        "sycophancy": [
            _rule_view(rule, "sycophancy")
            for rule in [
                *snapshot["s_system"]["s1_conditions"],
                *snapshot["s_system"]["s2_loci"],
            ]
        ],
        "calibrated_validation": [
            _rule_view(rule, "calibrated_validation")
            for rule in snapshot["validation_system"]["rules"]
        ],
    }
    counts = {axis: len(values) for axis, values in rules.items()}
    if counts != EXPECTED_ATOMIC_RULE_COUNTS:
        raise ValueError(f"unexpected appendix-rule counts: {counts}")
    return rules


def load_texting_styles(path: Path = TEXTING_STYLES_PATH) -> list[dict[str, str]]:
    value = read_json(path)
    validate_style_catalog(value, expected_count=64)
    return [deepcopy(style) for style in value["styles"]]


def seed_prompt_view(seed: SeedRecord) -> dict[str, Any]:
    return {
        "training_seed_id": seed.training_seed_id,
        "domain": deepcopy(seed.domain),
        "text": seed.redacted_text,
    }


def _selected_syc_rule(rule: Mapping[str, Any]) -> dict[str, Any]:
    if rule["rule_id"] not in SYCOPHANCY_TARGET_RULE_IDS:
        raise ValueError(f"{rule['rule_id']} is not a failure-seeking target")
    return deepcopy(rule)


def _selected_validation_rule(rule: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": rule["chunk_id"],
        "title": rule["title"],
        "axis": "calibrated_validation",
        "primary_rule_ids": [rule["rule_id"]],
        "rule": deepcopy(rule),
    }


def _schedule_domains(
    domains: list[dict[str, str]],
    count: int,
    rng: random.Random,
) -> list[dict[str, str]]:
    values: list[dict[str, str]] = []
    while len(values) < count:
        round_values = [deepcopy(item) for item in domains]
        rng.shuffle(round_values)
        values.extend(round_values)
    rng.shuffle(values)
    return values[:count]


def _axis_domains(
    domains: list[dict[str, str]],
    *,
    axis_index: int,
    extra_domains: list[dict[str, str]],
    rng: random.Random,
) -> list[dict[str, str]]:
    del axis_index, extra_domains
    return _schedule_domains(domains, 10, rng)


def _syc_rule_schedule(
    rules: list[dict[str, Any]], rng: random.Random
) -> list[dict[str, Any]]:
    index = {rule["rule_id"]: rule for rule in rules}
    rule_ids = [
        "S1.a",
        "S1.a",
        "S1.c",
        "S1.c",
        "S2.a",
        "S2.b",
        "S2.b",
        "S2.c",
        "S2.c",
        "S2.d",
    ]
    values = [_selected_syc_rule(index[rule_id]) for rule_id in rule_ids]
    rng.shuffle(values)
    return values


def _syc_rule_category_schedule(
    rules: list[dict[str, Any]],
    categories: tuple[dict[str, str], ...],
    *,
    sample_count: int,
    rng: random.Random,
) -> list[tuple[dict[str, Any], dict[str, str], str]]:
    """Jointly balance categories, targets, and personal/general contexts."""

    rule_index = {rule["rule_id"]: rule for rule in rules}
    category_index = {category["name"]: category for category in categories}
    personal = "personal_or_everyday"
    general = "professional_or_public"
    if sample_count == 10:
        specs = [
            ("Work & Plan Evaluation", "S2.b", personal),
            ("Work & Plan Evaluation", "S2.b", general),
            ("Advice & Recommendation", "S2.c", personal),
            ("Advice & Recommendation", "S1.a", general),
            ("Factual & Analytical", "S2.a", personal),
            ("Factual & Analytical", "S1.c", general),
            ("Moral & Ethical Assessment", "S1.c", personal),
            ("Moral & Ethical Assessment", "S2.d", general),
            ("Content Creation & Task Execution", "S2.c", personal),
            ("Content Creation & Task Execution", "S1.a", general),
        ]
    elif sample_count == 20:
        specs = [
            ("Work & Plan Evaluation", "S2.b", personal),
            ("Work & Plan Evaluation", "S2.b", general),
            ("Work & Plan Evaluation", "S2.b", personal),
            ("Work & Plan Evaluation", "S1.c", general),
            ("Advice & Recommendation", "S2.c", personal),
            ("Advice & Recommendation", "S2.c", general),
            ("Advice & Recommendation", "S1.a", personal),
            ("Advice & Recommendation", "S1.c", general),
            ("Factual & Analytical", "S2.a", personal),
            ("Factual & Analytical", "S2.a", general),
            ("Factual & Analytical", "S1.a", personal),
            ("Factual & Analytical", "S1.c", general),
            ("Moral & Ethical Assessment", "S2.b", personal),
            ("Moral & Ethical Assessment", "S2.c", general),
            ("Moral & Ethical Assessment", "S1.a", personal),
            ("Moral & Ethical Assessment", "S2.d", general),
            ("Content Creation & Task Execution", "S2.c", personal),
            ("Content Creation & Task Execution", "S1.a", general),
            ("Content Creation & Task Execution", "S1.c", personal),
            ("Content Creation & Task Execution", "S2.d", general),
        ]
    else:
        raise ValueError("sycophancy schedule requires exactly 10 or 20 samples")

    values = [
        (
            _selected_syc_rule(rule_index[rule_id]),
            deepcopy(category_index[category_name]),
            scenario_context,
        )
        for category_name, rule_id, scenario_context in specs
    ]
    rng.shuffle(values)
    return values


def _validation_rule_schedule(
    rules: list[dict[str, Any]], rng: random.Random
) -> list[dict[str, Any]]:
    index = {rule["rule_id"]: rule for rule in rules}
    ids = ["V1", "V2", "V3", "V1", "V2", "V3", "V1", "V2", "V3", "V1"]
    values = [_selected_validation_rule(index[rule_id]) for rule_id in ids]
    rng.shuffle(values)
    return values


def _validation_category_schedule(
    categories: tuple[dict[str, str], ...],
    rng: random.Random,
) -> list[dict[str, str]]:
    values = [deepcopy(category) for category in categories for _ in range(2)]
    rng.shuffle(values)
    return values


def _validation_category_tier_schedule(
    categories: Sequence[Mapping[str, Any]], rng: random.Random
) -> list[tuple[dict[str, Any], str]]:
    """Return one direct and one integrated sample per human-context category."""

    pairs = [
        (deepcopy(category), tier)
        for category in categories
        for tier in ("direct", "integrated")
    ]
    rng.shuffle(pairs)
    return pairs


def build_validation_plan(
    *,
    sample_count: int = 10,
    planning_seed: int = 20260813,
    seed_corpus_path: Path = SEED_CORPUS_PATH,
    rule_snapshot_path: Path = APPENDIX_RULES_PATH,
    texting_styles_path: Path = TEXTING_STYLES_PATH,
) -> list[dict[str, Any]]:
    """Build a calibrated-validation plan balanced across human contexts."""

    if sample_count != 10:
        raise ValueError("the calibrated-validation pilot requires exactly 10 samples")
    from .prompts import CALIBRATED_VALIDATION_HUMAN_CONTEXT_CATEGORIES

    pool = SeedPool(seed_corpus_path)
    rules = atomic_rules_by_axis(load_rule_snapshot(rule_snapshot_path))[
        "calibrated_validation"
    ]
    styles_value = read_json(texting_styles_path)
    validate_style_catalog(styles_value, expected_count=64)
    styles = [deepcopy(style) for style in styles_value["styles"]]

    rng = random.Random(planning_seed)
    domains = [deepcopy(item) for item in pool.domains]
    domain_schedule = _schedule_domains(domains, 10, rng)
    evaluated_rules = _validation_rule_schedule(rules, rng)
    category_tiers = _validation_category_tier_schedule(
        CALIBRATED_VALIDATION_HUMAN_CONTEXT_CATEGORIES, rng
    )
    rng.shuffle(styles)

    assignments: list[dict[str, Any]] = []
    used_seed_ids: set[str] = set()
    for ordinal, (domain, evaluated_rule, (category, difficulty_tier), style) in enumerate(
        zip(domain_schedule, evaluated_rules, category_tiers, styles), start=1
    ):
        seed_rng = random.Random(
            f"{planning_seed}:calibrated_validation:{ordinal}:{domain['id']}"
        )
        seed = pool.choose(domain["name"], seed_rng)
        while seed.training_seed_id in used_seed_ids:
            seed = pool.choose(domain["name"], seed_rng)
        used_seed_ids.add(seed.training_seed_id)
        primary_rule_id = evaluated_rule["primary_rule_ids"][0]
        assignments.append(
            {
                "schema_version": 2,
                "id": f"RG-CV-{ordinal:03d}",
                "benchmark_axis": "calibrated_validation",
                "domain": deepcopy(seed.domain),
                "validation_category": category,
                "difficulty_tier": difficulty_tier,
                "evaluated_rule": evaluated_rule,
                "selected_rules": {"selected_rule_ids": [primary_rule_id]},
                "all_axis_rules": deepcopy(rules),
                "source_seed": seed_prompt_view(seed),
                "texting_style": style,
            }
        )

    validate_validation_plan(assignments)
    return assignments


def _uniform_validation_rule_schedule(
    rules: list[dict[str, Any]], sample_count: int, rng: random.Random
) -> list[dict[str, Any]]:
    index = {rule["rule_id"]: rule for rule in rules}
    rule_ids = ("V1", "V2", "V3")
    base = sample_count // len(rule_ids)
    remainder = sample_count % len(rule_ids)
    ids = [
        rule_id
        for index_offset, rule_id in enumerate(rule_ids)
        for _ in range(base + (1 if index_offset < remainder else 0))
    ]
    values = [_selected_validation_rule(index[rule_id]) for rule_id in ids]
    rng.shuffle(values)
    return values


def _human_review_category_tier_schedule(
    categories: Sequence[Mapping[str, Any]],
    *,
    samples_per_category: int,
    direct_per_category: int,
    rng: random.Random,
) -> list[tuple[dict[str, Any], str]]:
    integrated_per_category = samples_per_category - direct_per_category
    pairs = [
        (deepcopy(category), tier)
        for category in categories
        for tier in ("direct",) * direct_per_category
        + ("integrated",) * integrated_per_category
    ]
    rng.shuffle(pairs)
    return pairs


def _choose_unused_seed(
    pool: SeedPool,
    domain_name: str,
    rng: random.Random,
    blocked: set[str],
    *,
    max_attempts: int = 256,
) -> SeedRecord:
    for _ in range(max_attempts):
        seed = pool.choose(domain_name, rng)
        if seed.training_seed_id not in blocked:
            return seed
    raise ValueError(f"could not find unused seed for domain {domain_name!r}")


def _uniform_domain_schedule(
    domains: list[dict[str, str]], count: int, rng: random.Random
) -> list[dict[str, str]]:
    base = count // len(domains)
    remainder = count % len(domains)
    values = [
        deepcopy(domain)
        for index, domain in enumerate(domains)
        for _ in range(base + (1 if index < remainder else 0))
    ]
    rng.shuffle(values)
    return values


def build_human_review_cv_plan(
    *,
    sample_count: int = 50,
    direct_fraction: float = 0.8,
    planning_seed: int = 20260812,
    excluded_seed_ids: set[str] | None = None,
    seed_corpus_path: Path = SEED_CORPUS_PATH,
    rule_snapshot_path: Path = APPENDIX_RULES_PATH,
    texting_styles_path: Path = TEXTING_STYLES_PATH,
) -> list[dict[str, Any]]:
    """Build a 50-sample calibrated-validation plan for human review."""

    if sample_count < HF_SEED_DOMAIN_COUNT:
        raise ValueError(
            f"human-review CV plan needs at least {HF_SEED_DOMAIN_COUNT} samples "
            "to cover every domain"
        )
    if not 0 < direct_fraction < 1:
        raise ValueError("direct_fraction must be between 0 and 1")
    from .prompts import CALIBRATED_VALIDATION_HUMAN_CONTEXT_CATEGORIES

    direct_count = int(round(sample_count * direct_fraction))
    if direct_count + (sample_count - direct_count) != sample_count:
        raise ValueError("direct_fraction must yield an integer direct count")
    integrated_count = sample_count - direct_count
    category_count = len(CALIBRATED_VALIDATION_HUMAN_CONTEXT_CATEGORIES)
    if sample_count % category_count:
        raise ValueError("sample_count must be divisible by validation category count")
    samples_per_category = sample_count // category_count
    direct_per_category = direct_count // category_count
    if direct_per_category * category_count != direct_count:
        raise ValueError("direct_fraction must divide evenly across validation categories")

    pool = SeedPool(seed_corpus_path)
    rules = atomic_rules_by_axis(load_rule_snapshot(rule_snapshot_path))[
        "calibrated_validation"
    ]
    styles_value = read_json(texting_styles_path)
    validate_style_catalog(styles_value, expected_count=64)
    styles = [deepcopy(style) for style in styles_value["styles"]]
    if sample_count > len(styles):
        raise ValueError("not enough texting styles for the requested sample count")

    rng = random.Random(planning_seed)
    domains = [deepcopy(item) for item in pool.domains]
    domain_schedule = _uniform_domain_schedule(domains, sample_count, rng)
    evaluated_rules = _uniform_validation_rule_schedule(rules, sample_count, rng)
    category_tiers = _human_review_category_tier_schedule(
        CALIBRATED_VALIDATION_HUMAN_CONTEXT_CATEGORIES,
        samples_per_category=samples_per_category,
        direct_per_category=direct_per_category,
        rng=rng,
    )
    rng.shuffle(styles)

    blocked = set(excluded_seed_ids or ())
    assignments: list[dict[str, Any]] = []
    used_seed_ids: set[str] = set()
    for ordinal, (domain, evaluated_rule, (category, difficulty_tier), style) in enumerate(
        zip(domain_schedule, evaluated_rules, category_tiers, styles), start=1
    ):
        seed_rng = random.Random(
            f"{planning_seed}:human_review_cv:{ordinal}:{domain['id']}"
        )
        seed = _choose_unused_seed(
            pool,
            domain["name"],
            seed_rng,
            blocked | used_seed_ids,
        )
        used_seed_ids.add(seed.training_seed_id)
        primary_rule_id = evaluated_rule["primary_rule_ids"][0]
        assignments.append(
            {
                "schema_version": 2,
                "id": f"RG-CV-{ordinal:03d}",
                "benchmark_axis": "calibrated_validation",
                "domain": deepcopy(seed.domain),
                "validation_category": category,
                "difficulty_tier": difficulty_tier,
                "evaluated_rule": evaluated_rule,
                "selected_rules": {"selected_rule_ids": [primary_rule_id]},
                "all_axis_rules": deepcopy(rules),
                "source_seed": seed_prompt_view(seed),
                "texting_style": style,
            }
        )

    validate_human_review_cv_plan(assignments)
    return assignments


def validate_human_review_cv_plan(plan: list[dict[str, Any]]) -> None:
    from .prompts import CALIBRATED_VALIDATION_HUMAN_CONTEXT_CATEGORIES

    sample_count = len(plan)
    if sample_count < 1 or len({item["id"] for item in plan}) != sample_count:
        raise ValueError("human-review CV plan must contain unique assignments")
    if any(item["benchmark_axis"] != "calibrated_validation" for item in plan):
        raise ValueError("human-review CV plan may not contain another axis")
    if len({item["source_seed"]["training_seed_id"] for item in plan}) != sample_count:
        raise ValueError("HF inspiration seeds must be unique")
    if len({item["texting_style"]["id"] for item in plan}) != sample_count:
        raise ValueError("role-player texting styles must be unique")

    domain_counts = Counter(item["domain"]["id"] for item in plan)
    if len(domain_counts) != HF_SEED_DOMAIN_COUNT:
        raise ValueError(
            f"human-review CV plan must cover all {HF_SEED_DOMAIN_COUNT} HF domains"
        )
    if max(domain_counts.values()) - min(domain_counts.values()) > 1:
        raise ValueError("human-review CV domain schedule must be nearly uniform")

    rule_counts = Counter(
        item["evaluated_rule"]["primary_rule_ids"][0] for item in plan
    )
    expected_rule_counts = Counter(
        {
            rule_id: sample_count // 3 + (1 if index < sample_count % 3 else 0)
            for index, rule_id in enumerate(("V1", "V2", "V3"))
        }
    )
    if rule_counts != expected_rule_counts:
        raise ValueError(
            f"human-review CV rule schedule must be uniform: "
            f"expected {dict(expected_rule_counts)}, got {dict(rule_counts)}"
        )

    category_count = len(CALIBRATED_VALIDATION_HUMAN_CONTEXT_CATEGORIES)
    if sample_count % category_count:
        raise ValueError("sample_count must divide validation category count")
    samples_per_category = sample_count // category_count
    expected_categories = Counter(
        {category["name"]: samples_per_category for category in CALIBRATED_VALIDATION_HUMAN_CONTEXT_CATEGORIES}
    )
    if Counter(item["validation_category"]["name"] for item in plan) != expected_categories:
        raise ValueError("human-review CV plan must be balanced by human context")

    direct_count = sum(1 for item in plan if item["difficulty_tier"] == "direct")
    integrated_count = sample_count - direct_count
    direct_per_category = direct_count // category_count
    if Counter(item["difficulty_tier"] for item in plan) != Counter(
        {"direct": direct_count, "integrated": integrated_count}
    ):
        raise ValueError("human-review CV plan has an invalid direct/integrated split")
    for category in CALIBRATED_VALIDATION_HUMAN_CONTEXT_CATEGORIES:
        name = category["name"]
        tiers = Counter(
            item["difficulty_tier"]
            for item in plan
            if item["validation_category"]["name"] == name
        )
        if tiers != Counter(
            {
                "direct": direct_per_category,
                "integrated": samples_per_category - direct_per_category,
            }
        ):
            raise ValueError(
                f"validation category {name} must preserve the direct/integrated ratio"
            )

    for item in plan:
        if "interaction_category" in item:
            raise ValueError("validation assignments must not receive a task category")
        if item["domain"] != item["source_seed"]["domain"]:
            raise ValueError("assignment domain must come from its HF seed record")
        if item["evaluated_rule"]["axis"] != "calibrated_validation":
            raise ValueError("calibrated-validation assignment contains a cross-axis rule")
        if item["selected_rules"] != {
            "selected_rule_ids": [item["evaluated_rule"]["primary_rule_ids"][0]]
        }:
            raise ValueError("calibrated-validation assignment must target one rule")
        if len(item["all_axis_rules"]) != EXPECTED_ATOMIC_RULE_COUNTS[
            "calibrated_validation"
        ]:
            raise ValueError("calibrated-validation assignment lacks complete rules")


def build_plan(
    *,
    sample_count: int = 20,
    planning_seed: int = 20260807,
    seed_corpus_path: Path = SEED_CORPUS_PATH,
    rule_snapshot_path: Path = APPENDIX_RULES_PATH,
    texting_styles_path: Path = TEXTING_STYLES_PATH,
) -> list[dict[str, Any]]:
    """Build a deterministic 10/10 plan using diverse HF-seed domains.

    The seed text and its domain inspire the scenario.  The texting style is a
    separate role-player input and is deliberately not part of scenario design.
    """
    if sample_count != 20:
        raise ValueError("the v2 replacement requires exactly 20 samples")
    from .prompts import CALIBRATED_VALIDATION_HUMAN_CONTEXT_CATEGORIES

    pool = SeedPool(seed_corpus_path)
    rules_by_axis = atomic_rules_by_axis(load_rule_snapshot(rule_snapshot_path))
    styles_value = read_json(texting_styles_path)
    validate_style_catalog(styles_value, expected_count=64)
    styles = [deepcopy(style) for style in styles_value["styles"]]

    rng = random.Random(planning_seed)
    domains = pool.domains
    extra_domains = [deepcopy(item) for item in domains]
    rng.shuffle(extra_domains)
    extra_domains = extra_domains[:4]
    rng.shuffle(styles)

    assignments: list[dict[str, Any]] = []
    used_seed_ids: set[str] = set()
    style_cursor = 0
    for axis_index, axis in enumerate(AXES):
        axis_rng = random.Random(f"{planning_seed}:{axis}")
        domain_schedule = _axis_domains(
            domains,
            axis_index=axis_index,
            extra_domains=extra_domains,
            rng=axis_rng,
        )
        if axis == "sycophancy":
            evaluated_rules = _syc_rule_schedule(rules_by_axis[axis], axis_rng)
        else:
            evaluated_rules = _validation_rule_schedule(
                rules_by_axis[axis], axis_rng
            )
            category_tiers = _validation_category_tier_schedule(
                CALIBRATED_VALIDATION_HUMAN_CONTEXT_CATEGORIES, axis_rng
            )

        for ordinal, (domain, evaluated_rule) in enumerate(
            zip(domain_schedule, evaluated_rules), start=1
        ):
            seed_rng = random.Random(
                f"{planning_seed}:{axis}:{ordinal}:{domain['id']}"
            )
            seed = pool.choose(domain["name"], seed_rng)
            while seed.training_seed_id in used_seed_ids:
                seed = pool.choose(domain["name"], seed_rng)
            used_seed_ids.add(seed.training_seed_id)
            primary_ids = evaluated_rule["primary_rule_ids"]
            selected_rules = {"selected_rule_ids": [primary_ids[0]]}
            cv_fields: dict[str, Any] = {}
            if axis == "calibrated_validation":
                category, difficulty_tier = category_tiers[ordinal - 1]
                cv_fields = {
                    "validation_category": category,
                    "difficulty_tier": difficulty_tier,
                }
            assignments.append(
                {
                    "schema_version": 2,
                    "id": f"RG-{AXIS_PREFIX[axis]}-{ordinal:03d}",
                    "benchmark_axis": axis,
                    "domain": deepcopy(seed.domain),
                    **(
                        {
                            "scenario_context": (
                                "personal_or_everyday"
                                if ordinal % 2
                                else "professional_or_public"
                            )
                        }
                        if axis == "sycophancy"
                        else cv_fields
                    ),
                    "evaluated_rule": evaluated_rule,
                    "selected_rules": selected_rules,
                    "all_axis_rules": deepcopy(rules_by_axis[axis]),
                    "source_seed": seed_prompt_view(seed),
                    "texting_style": deepcopy(styles[style_cursor]),
                }
            )
            style_cursor += 1

    syc = [item for item in assignments if item["benchmark_axis"] == "sycophancy"]
    validation = [
        item
        for item in assignments
        if item["benchmark_axis"] == "calibrated_validation"
    ]
    plan = [item for pair in zip(syc, validation) for item in pair]
    validate_plan(plan)
    return plan


def build_sycophancy_plan(
    *,
    sample_count: int = 10,
    planning_seed: int = 20260812,
    seed_corpus_path: Path = SEED_CORPUS_PATH,
    rule_snapshot_path: Path = APPENDIX_RULES_PATH,
    texting_styles_path: Path = TEXTING_STYLES_PATH,
) -> list[dict[str, Any]]:
    """Build a deterministic sycophancy plan balanced across five categories."""
    if sample_count not in {10, 20}:
        raise ValueError("the sycophancy pilot requires exactly 10 or 20 samples")

    # Imported lazily so source loading stays independent of prompt rendering.
    from .prompts import SYCOPHANCY_SCENARIO_CATEGORIES

    pool = SeedPool(seed_corpus_path)
    rules = atomic_rules_by_axis(load_rule_snapshot(rule_snapshot_path))["sycophancy"]
    styles_value = read_json(texting_styles_path)
    validate_style_catalog(styles_value, expected_count=64)
    styles = [deepcopy(style) for style in styles_value["styles"]]

    rng = random.Random(planning_seed)
    domains = [deepcopy(item) for item in pool.domains]
    rng.shuffle(domains)
    domain_schedule: list[dict[str, str]] = []
    while len(domain_schedule) < sample_count:
        domain_round = [deepcopy(item) for item in domains]
        rng.shuffle(domain_round)
        domain_schedule.extend(domain_round)
    domain_schedule = domain_schedule[:sample_count]
    rng.shuffle(domain_schedule)
    rule_category_schedule = _syc_rule_category_schedule(
        rules,
        SYCOPHANCY_SCENARIO_CATEGORIES,
        sample_count=sample_count,
        rng=rng,
    )
    rng.shuffle(styles)

    assignments: list[dict[str, Any]] = []
    used_seed_ids: set[str] = set()
    for ordinal, (
        domain,
        (evaluated_rule, category, scenario_context),
        style,
    ) in enumerate(
        zip(domain_schedule, rule_category_schedule, styles), start=1
    ):
        seed_rng = random.Random(
            f"{planning_seed}:sycophancy:{ordinal}:{domain['id']}"
        )
        seed = pool.choose(domain["name"], seed_rng)
        while seed.training_seed_id in used_seed_ids:
            seed = pool.choose(domain["name"], seed_rng)
        used_seed_ids.add(seed.training_seed_id)
        primary_rule_id = evaluated_rule["primary_rule_ids"][0]
        assignments.append(
            {
                "schema_version": 2,
                "id": f"RG-SYC-{ordinal:03d}",
                "benchmark_axis": "sycophancy",
                "domain": deepcopy(seed.domain),
                "interaction_category": category,
                "scenario_context": scenario_context,
                "evaluated_rule": evaluated_rule,
                "selected_rules": {"selected_rule_ids": [primary_rule_id]},
                "all_axis_rules": deepcopy(rules),
                "source_seed": seed_prompt_view(seed),
                "texting_style": style,
            }
        )

    validate_sycophancy_plan(assignments)
    return assignments


def validate_plan(plan: list[dict[str, Any]]) -> None:
    if len(plan) != 20 or len({item["id"] for item in plan}) != 20:
        raise ValueError("plan must contain exactly 20 unique assignments")
    if Counter(item["benchmark_axis"] for item in plan) != Counter(
        {"sycophancy": 10, "calibrated_validation": 10}
    ):
        raise ValueError("plan must split 10/10 across benchmark axes")
    if len({item["source_seed"]["training_seed_id"] for item in plan}) != 20:
        raise ValueError("HF inspiration seeds must be unique")
    if len({item["texting_style"]["id"] for item in plan}) != 20:
        raise ValueError("role-player texting styles must be unique")
    for axis in AXES:
        axis_items = [item for item in plan if item["benchmark_axis"] == axis]
        if len({item["domain"]["id"] for item in axis_items}) != HF_SEED_DOMAIN_COUNT:
            raise ValueError(
                f"{axis} must cover all {HF_SEED_DOMAIN_COUNT} HF-seed domains"
            )
        if any(item["domain"] != item["source_seed"]["domain"] for item in axis_items):
            raise ValueError("assignment domain must come from its HF seed record")
        if any(item["evaluated_rule"]["axis"] != axis for item in axis_items):
            raise ValueError(f"{axis} assignment contains a cross-axis rule")
        if any(
            len(item["evaluated_rule"]["primary_rule_ids"]) != 1
            or item["selected_rules"]
            != {
                "selected_rule_ids": [
                    item["evaluated_rule"]["primary_rule_ids"][0]
                ]
            }
            for item in axis_items
        ):
            raise ValueError(f"{axis} assignment must target one atomic rule")
        if any(
            len(item["all_axis_rules"]) != EXPECTED_ATOMIC_RULE_COUNTS[axis]
            for item in axis_items
        ):
            raise ValueError(f"{axis} assignment lacks complete appendix rules")
        if axis == "sycophancy" and Counter(
            item.get("scenario_context") for item in axis_items
        ) != Counter(
            {"personal_or_everyday": 5, "professional_or_public": 5}
        ):
            raise ValueError("sycophancy assignments must split context emphasis 5/5")
    validate_validation_plan(
        [item for item in plan if item["benchmark_axis"] == "calibrated_validation"]
    )


def validate_sycophancy_plan(plan: list[dict[str, Any]]) -> None:
    from .prompts import SYCOPHANCY_SCENARIO_CATEGORIES

    sample_count = len(plan)
    if sample_count not in {10, 20} or len({item["id"] for item in plan}) != sample_count:
        raise ValueError(
            "sycophancy plan must contain exactly 10 or 20 unique assignments"
        )
    if any(item["benchmark_axis"] != "sycophancy" for item in plan):
        raise ValueError("sycophancy plan may not contain another benchmark axis")
    if len({item["source_seed"]["training_seed_id"] for item in plan}) != sample_count:
        raise ValueError("HF inspiration seeds must be unique")
    if len({item["texting_style"]["id"] for item in plan}) != sample_count:
        raise ValueError("role-player texting styles must be unique")
    if len({item["domain"]["id"] for item in plan}) != HF_SEED_DOMAIN_COUNT:
        raise ValueError(
            f"sycophancy plan must cover all {HF_SEED_DOMAIN_COUNT} HF-seed domains"
        )
    expected_categories = Counter(
        {
            category["name"]: sample_count // len(SYCOPHANCY_SCENARIO_CATEGORIES)
            for category in SYCOPHANCY_SCENARIO_CATEGORIES
        }
    )
    actual_categories = Counter(
        item["interaction_category"]["name"] for item in plan
    )
    if actual_categories != expected_categories:
        raise ValueError("sycophancy plan must be equally balanced by category")
    expected_context_count = sample_count // 2
    if Counter(item.get("scenario_context") for item in plan) != Counter(
        {
            "personal_or_everyday": expected_context_count,
            "professional_or_public": expected_context_count,
        }
    ):
        raise ValueError("sycophancy plan must split personal and general contexts")
    work_items = [
        item
        for item in plan
        if item["interaction_category"]["name"] == "Work & Plan Evaluation"
    ]
    expected_work_s2b = 2 if sample_count == 10 else 3
    if sum(
        item["evaluated_rule"]["primary_rule_ids"] == ["S2.b"]
        for item in work_items
    ) != expected_work_s2b:
        raise ValueError("Work & Plan Evaluation must primarily target S2.b")
    for item in plan:
        if item["domain"] != item["source_seed"]["domain"]:
            raise ValueError("assignment domain must come from its HF seed record")
        if item["evaluated_rule"]["axis"] != "sycophancy":
            raise ValueError("sycophancy assignment contains a cross-axis rule")
        if (
            len(item["evaluated_rule"]["primary_rule_ids"]) != 1
            or item["selected_rules"]
            != {
                "selected_rule_ids": [
                    item["evaluated_rule"]["primary_rule_ids"][0]
                ]
            }
        ):
            raise ValueError("sycophancy assignment must target one atomic rule")
        if len(item["all_axis_rules"]) != EXPECTED_ATOMIC_RULE_COUNTS["sycophancy"]:
            raise ValueError("sycophancy assignment lacks complete appendix rules")


def validate_validation_plan(plan: list[dict[str, Any]]) -> None:
    from .prompts import CALIBRATED_VALIDATION_HUMAN_CONTEXT_CATEGORIES

    if len(plan) != 10 or len({item["id"] for item in plan}) != 10:
        raise ValueError(
            "calibrated-validation plan must contain exactly 10 unique assignments"
        )
    if any(item["benchmark_axis"] != "calibrated_validation" for item in plan):
        raise ValueError("calibrated-validation plan may not contain another axis")
    if len({item["source_seed"]["training_seed_id"] for item in plan}) != 10:
        raise ValueError("HF inspiration seeds must be unique")
    if len({item["texting_style"]["id"] for item in plan}) != 10:
        raise ValueError("role-player texting styles must be unique")
    if len({item["domain"]["id"] for item in plan}) != HF_SEED_DOMAIN_COUNT:
        raise ValueError(
            f"calibrated-validation plan must cover all {HF_SEED_DOMAIN_COUNT} HF domains"
        )
    if Counter(
        item["evaluated_rule"]["primary_rule_ids"][0] for item in plan
    ) != Counter({"V1": 4, "V2": 3, "V3": 3}):
        raise ValueError("validation rule schedule must be V1/V2/V3 = 4/3/3")
    expected_categories = Counter(
        {
            category["name"]: 2
            for category in CALIBRATED_VALIDATION_HUMAN_CONTEXT_CATEGORIES
        }
    )
    if Counter(item["validation_category"]["name"] for item in plan) != expected_categories:
        raise ValueError("validation plan must be equally balanced by human context")
    if Counter(item["difficulty_tier"] for item in plan) != Counter(
        {"direct": 5, "integrated": 5}
    ):
        raise ValueError("validation plan must split direct and integrated tiers 5/5")
    for category in CALIBRATED_VALIDATION_HUMAN_CONTEXT_CATEGORIES:
        name = category["name"]
        tiers = {
            item["difficulty_tier"]
            for item in plan
            if item["validation_category"]["name"] == name
        }
        if tiers != {"direct", "integrated"}:
            raise ValueError(
                f"validation category {name} must appear once per difficulty tier"
            )
    for item in plan:
        if "interaction_category" in item:
            raise ValueError("validation assignments must not receive a task category")
        if item["domain"] != item["source_seed"]["domain"]:
            raise ValueError("assignment domain must come from its HF seed record")
        if item["evaluated_rule"]["axis"] != "calibrated_validation":
            raise ValueError("calibrated-validation assignment contains a cross-axis rule")
        if item["selected_rules"] != {
            "selected_rule_ids": [item["evaluated_rule"]["primary_rule_ids"][0]]
        }:
            raise ValueError("calibrated-validation assignment must target one rule")
        if len(item["all_axis_rules"]) != EXPECTED_ATOMIC_RULE_COUNTS[
            "calibrated_validation"
        ]:
            raise ValueError("calibrated-validation assignment lacks complete rules")


__all__ = [
    "AXES",
    "EXPECTED_ATOMIC_RULE_COUNTS",
    "SYCOPHANCY_TARGET_RULE_IDS",
    "atomic_rules_by_axis",
    "build_human_review_cv_plan",
    "build_plan",
    "build_sycophancy_plan",
    "build_validation_plan",
    "load_rule_snapshot",
    "load_texting_styles",
    "seed_prompt_view",
    "validate_human_review_cv_plan",
    "validate_plan",
    "validate_sycophancy_plan",
    "validate_validation_plan",
]
