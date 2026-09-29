"""Plan builder for archetype-guided calibrated-validation samples."""

from __future__ import annotations

import random
from collections import Counter
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
    HF_SEED_DOMAIN_COUNT,
    SeedPool,
    _selected_validation_rule,
    atomic_rules_by_axis,
    load_rule_snapshot,
    seed_prompt_view,
    validate_style_catalog,
)

from .prompts_validation import load_validation_conversation_archetypes
from .severity import assign_severities, normalize_severity

VALIDATION_ARCHETYPE_IDS = tuple(
    item["id"] for item in load_validation_conversation_archetypes()
)

DEFAULT_TEXTING_STYLES_PATH = (
    PROJECT_ROOT
    / "sft_data"
    / "data"
    / "runs"
    / "sft-150-hybrid-20260812"
    / "texting_styles.json"
)


def _validation_archetype_specs(sample_count: int) -> list[tuple[str, str]]:
    """Return (archetype_id, rule_id) specs."""

    if sample_count == 10:
        rows = [
            ("A1", "V1"),
            ("A1", "V2"),
            ("A2", "V1"),
            ("A2", "V2"),
            ("A3", "V2"),
            ("A3", "V3"),
            ("A4", "V1"),
            ("A4", "V1"),
            ("A5", "V3"),
            ("A6", "V3"),
        ]
    elif sample_count == 20:
        rows = [
            ("A1", "V2"),
            ("A1", "V3"),
            ("A1", "V1"),
            ("A1", "V2"),
            ("A2", "V1"),
            ("A2", "V2"),
            ("A2", "V3"),
            ("A3", "V1"),
            ("A3", "V2"),
            ("A3", "V3"),
            ("A4", "V1"),
            ("A4", "V2"),
            ("A4", "V3"),
            ("A4", "V1"),
            ("A5", "V2"),
            ("A5", "V3"),
            ("A5", "V1"),
            ("A6", "V1"),
            ("A6", "V2"),
            ("A6", "V3"),
        ]
    elif sample_count == 25:
        rows = list(_validation_archetype_specs(20))
        rows.remove(("A6", "V3"))
        rows.extend(
            [
                ("A1", "V1"),
                ("A2", "V1"),
                ("A2", "V3"),
                ("A3", "V3"),
                ("A3", "V2"),
                ("A4", "V3"),
            ]
        )
    else:
        raise ValueError("validation archetype plan requires exactly 10, 20, or 25 samples")
    return rows


def build_archetype_validation_plan(
    *,
    sample_count: int = 10,
    planning_seed: int = 20260815,
    seed_corpus_path: Path = SEED_CORPUS_PATH,
    rule_snapshot_path: Path = APPENDIX_RULES_PATH,
    texting_styles_path: Path = DEFAULT_TEXTING_STYLES_PATH,
    severity: str = "mixed",
) -> list[dict[str, Any]]:
    severity_levels = assign_severities(sample_count, severity, planning_seed)
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

    specs = _validation_archetype_specs(sample_count)
    rng.shuffle(specs)
    rng.shuffle(styles)

    assignments: list[dict[str, Any]] = []
    used_seed_ids: set[str] = set()
    for ordinal, (domain, spec, style) in enumerate(
        zip(domain_schedule, specs, styles), start=1
    ):
        archetype_id, rule_id = spec
        seed_rng = random.Random(
            f"{planning_seed}:archetype-cv:{ordinal}:{domain['id']}"
        )
        seed = pool.choose_unique(domain["name"], seed_rng, used_seed_ids)
        used_seed_ids.add(seed.training_seed_id)
        evaluated_rule = _selected_validation_rule(rule_index[rule_id])
        primary_rule_id = evaluated_rule["primary_rule_ids"][0]
        archetype = archetypes[archetype_id]
        assignments.append(
            {
                "schema_version": 2,
                "id": f"AG-CV-{ordinal:03d}",
                "benchmark_axis": "calibrated_validation",
                "severity": severity_levels[ordinal - 1],
                "domain": deepcopy(seed.domain),
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
                "texting_style": style,
            }
        )

    validate_archetype_validation_plan(assignments, severity=severity)
    return assignments


def validate_archetype_validation_plan(
    plan: list[dict[str, Any]], severity: str = "mixed"
) -> None:
    sample_count = len(plan)
    if sample_count not in {10, 20} or len({item["id"] for item in plan}) != sample_count:
        raise ValueError("plan must contain exactly 10 or 20 unique assignments")
    if any(item["benchmark_axis"] != "calibrated_validation" for item in plan):
        raise ValueError("plan must be calibrated_validation-only")
    for item in plan:
        normalize_severity(item.get("severity", "medium"))
    if severity != "mixed" and any(item.get("severity") != severity for item in plan):
        raise ValueError(f"plan must be uniform severity={severity}")
    if len({item["source_seed"]["training_seed_id"] for item in plan}) != sample_count:
        raise ValueError("HF inspiration seeds must be unique")
    if len({item["texting_style"]["id"] for item in plan}) != sample_count:
        raise ValueError("texting styles must be unique")
    if len({item["domain"]["id"] for item in plan}) != HF_SEED_DOMAIN_COUNT:
        raise ValueError(f"plan must cover all {HF_SEED_DOMAIN_COUNT} HF-seed domains")

    expected_archetypes: Counter[str] = Counter()
    if sample_count == 10:
        expected_archetypes.update(
            {
                "A1": 2,
                "A2": 2,
                "A3": 2,
                "A4": 2,
                "A5": 1,
                "A6": 1,
            }
        )
    else:
        expected_archetypes.update(
            {
                "A1": 4,
                "A2": 3,
                "A3": 3,
                "A4": 4,
                "A5": 3,
                "A6": 3,
            }
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

    for item in plan:
        if item["domain"] != item["source_seed"]["domain"]:
            raise ValueError("assignment domain must come from its HF seed record")
        if item["evaluated_rule"]["axis"] != "calibrated_validation":
            raise ValueError("cross-axis rule in validation assignment")
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
