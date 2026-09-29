"""Strict contracts for appendix-guided scenarios, feedback, audits, and transcripts."""

from __future__ import annotations

import re
from typing import Any, Mapping


SCENARIO_SCHEMA_VERSION = 2
SCENARIO_FIELDS = {
    "schema_version",
    "id",
    "benchmark_axis",
    "domain",
    "evaluated_rule",
    "source_seed_id",
    "user_role",
    "scenario_plan",
}
SYCOPHANCY_SCENARIO_FIELDS = SCENARIO_FIELDS | {"is_draft"}
CONTENT_CREATION_CATEGORY = "Content Creation & Task Execution"
DRAFT_DISPLACEMENT_PATTERN = re.compile(
    r"\b("
    r"draft|wording|copyedit|paste|memo section|polish|rewrite|"
    r"help.*draft|revise.*text|tone polish|ready to (?:paste|send)"
    r")\b",
    re.I,
)
CALIBRATED_VALIDATION_SCENARIO_FIELDS = SCENARIO_FIELDS | {"difficulty_tier"}
FORBIDDEN_SCENARIO_FIELDS = {
    "attack_vector",
    "attack_vector_name",
    "attack_plan",
    "difficulty_mechanism",
    "validation_targets",
    "validation_target",
    "minimum_recognition",
    "may_validate",
    "must_validate",
    "must_not_validate",
    "ideal_response_guidance",
    "anticipated_failure",
    "roleplay_card",
    "turn_plan",
    "immediate_goal",
    "opening_context",
    "initial_context",
    "private_reaction",
    "facts_available_now",
    "texting_style",
}

def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def sycophancy_requires_is_draft() -> bool:
    from rule_guided_benchmark.prompts import PROMPT_VERSION

    return "draft-declaration" in PROMPT_VERSION


def _scenario_fields(assignment: Mapping[str, Any]) -> set[str]:
    if assignment.get("benchmark_axis") == "calibrated_validation":
        fields = set(SCENARIO_FIELDS)
        if "difficulty_tier" in assignment:
            fields.add("difficulty_tier")
        return fields
    if assignment.get("benchmark_axis") == "sycophancy":
        if sycophancy_requires_is_draft():
            return set(SYCOPHANCY_SCENARIO_FIELDS)
        return set(SCENARIO_FIELDS)
    return set(SCENARIO_FIELDS)


def expected_is_draft(assignment: Mapping[str, Any]) -> bool:
    if assignment.get("benchmark_axis") != "sycophancy":
        raise ValueError("is_draft applies only to sycophancy scenarios")
    category = assignment.get("interaction_category", {}).get("name")
    return category == CONTENT_CREATION_CATEGORY


def scenario_has_draft_displacement(
    scenario: Mapping[str, Any],
) -> bool:
    combined = f"{scenario.get('user_role', '')}\n{scenario.get('scenario_plan', '')}"
    return bool(DRAFT_DISPLACEMENT_PATTERN.search(combined))


def validate_scenario(
    scenario: Mapping[str, Any], assignment: Mapping[str, Any]
) -> None:
    expected_fields = _scenario_fields(assignment)
    if set(scenario) != expected_fields:
        raise ValueError(
            "scenario fields mismatch; "
            f"missing={sorted(expected_fields - set(scenario))}, "
            f"extra={sorted(set(scenario) - expected_fields)}"
        )
    forbidden = sorted(FORBIDDEN_SCENARIO_FIELDS & set(scenario))
    if forbidden:
        raise ValueError(f"scenario contains forbidden authoring fields: {forbidden}")
    if scenario["schema_version"] != SCENARIO_SCHEMA_VERSION:
        raise ValueError(
            f"scenario schema_version must be {SCENARIO_SCHEMA_VERSION}"
        )
    for field in ("id", "benchmark_axis", "domain", "evaluated_rule"):
        if scenario[field] != assignment[field]:
            raise ValueError(f"scenario {field} must match its pinned assignment")
    if scenario["source_seed_id"] != assignment["source_seed"]["training_seed_id"]:
        raise ValueError("scenario source_seed_id must match its pinned assignment")
    if assignment.get("benchmark_axis") == "calibrated_validation":
        if (
            "difficulty_tier" in assignment
            and scenario.get("difficulty_tier") != assignment["difficulty_tier"]
        ):
            raise ValueError("scenario difficulty_tier must match its pinned assignment")
    if not _text(scenario["user_role"]):
        raise ValueError("user_role must be concrete prose")
    if not _text(scenario["scenario_plan"]):
        raise ValueError("scenario_plan must be concrete prose")
    if assignment.get("benchmark_axis") == "sycophancy" and sycophancy_requires_is_draft():
        is_draft = scenario.get("is_draft")
        if not isinstance(is_draft, bool):
            raise ValueError("is_draft must be a boolean for sycophancy scenarios")
        if is_draft != expected_is_draft(assignment):
            category = assignment["interaction_category"]["name"]
            expected = "true" if expected_is_draft(assignment) else "false"
            raise ValueError(
                f"is_draft must be {expected} for category {category!r}"
            )
        if not is_draft and scenario_has_draft_displacement(scenario):
            raise ValueError(
                "is_draft is false but user_role or scenario_plan centers "
                "draft, copyedit, polish, or paste-for-review language"
            )


REFINER_FEEDBACK_FIELDS = {
    "iteration",
    "verdict",
    "strengths",
    "issues",
    "revision_brief",
    "must_preserve",
}


def validate_refiner_feedback(value: Mapping[str, Any], *, iteration: int) -> None:
    if set(value) != REFINER_FEEDBACK_FIELDS:
        raise ValueError("refiner feedback has the wrong fields")
    if value["iteration"] != iteration:
        raise ValueError("refiner feedback iteration mismatch")
    if value["verdict"] not in {"ready", "revise"}:
        raise ValueError("refiner verdict must be ready or revise")
    for field in ("strengths", "issues", "must_preserve"):
        if not isinstance(value[field], list):
            raise ValueError(f"refiner {field} must be a list")
    if not all(_text(item) for item in value["strengths"]):
        raise ValueError("refiner strengths must contain text")
    if not all(_text(item) for item in value["must_preserve"]):
        raise ValueError("refiner must_preserve must contain text")
    for issue in value["issues"]:
        if not isinstance(issue, dict) or set(issue) != {
            "dimension",
            "severity",
            "explanation",
            "specific_change",
        }:
            raise ValueError("refiner issue has the wrong fields")
        if issue["severity"] not in {"minor", "major"}:
            raise ValueError("refiner issue severity must be minor or major")
        if not all(_text(issue[field]) for field in issue):
            raise ValueError("refiner issue fields must contain text")
    if not _text(value["revision_brief"]):
        raise ValueError("refiner revision_brief must contain text")


AUDIT_FIELDS = {
    "iteration",
    "decision",
    "checks",
    "issues",
    "change_summary",
    "revised_user_role",
    "revised_scenario_plan",
}
AUDIT_CHECKS = {
    "ten_turn_structure",
    "human_realism",
    "texting_style_fidelity",
    "conversation_contingency",
    "plan_abstraction",
    "plan_independence",
    "facts_visible_before_use",
    "selected_opportunity_reached",
    "difficulty_realized",
    "axis_purity",
    "no_benchmark_language",
}


def validate_audit(value: Mapping[str, Any], *, iteration: int) -> None:
    if set(value) != AUDIT_FIELDS:
        raise ValueError("audit has the wrong fields")
    if value["iteration"] != iteration:
        raise ValueError("audit iteration mismatch")
    decision = value["decision"]
    if decision not in {"keep", "revise_and_reroll"}:
        raise ValueError("audit decision must be keep or revise_and_reroll")
    checks = value["checks"]
    if not isinstance(checks, dict) or set(checks) != AUDIT_CHECKS:
        raise ValueError("audit checks have the wrong fields")
    if any(result not in {"pass", "fail"} for result in checks.values()):
        raise ValueError("audit checks must be pass or fail")
    if not isinstance(value["issues"], list) or not all(
        _text(item) for item in value["issues"]
    ):
        raise ValueError("audit issues must be a text list")
    if not _text(value["change_summary"]):
        raise ValueError("audit change_summary must contain text")
    revised = (value["revised_user_role"], value["revised_scenario_plan"])
    if decision == "keep":
        failed = sorted(name for name, result in checks.items() if result == "fail")
        if failed:
            raise ValueError(
                "keep audits require every check to pass; "
                f"failed={failed}"
            )
        if value["issues"]:
            raise ValueError("keep audits require an empty issues list")
        if revised != (None, None):
            raise ValueError("keep audits may not revise the scenario")
    else:
        if all(result == "pass" for result in checks.values()):
            raise ValueError("reroll audits must identify at least one failed check")
        if not value["issues"]:
            raise ValueError("reroll audits require at least one specific issue")
        if not all(_text(item) for item in revised):
            raise ValueError("reroll audits must return both revised scenario fields")


def validate_transcript(transcript: Mapping[str, Any]) -> None:
    turns = transcript.get("turns")
    if not isinstance(turns, list) or len(turns) != 10:
        raise ValueError("transcript must contain ten alternating turns")
    for index, turn in enumerate(turns, 1):
        expected_role = "user" if index % 2 else "assistant"
        if (
            not isinstance(turn, dict)
            or turn.get("turn") != index
            or turn.get("role") != expected_role
            or not _text(turn.get("content"))
        ):
            raise ValueError(f"invalid transcript turn {index}")


__all__ = [
    "AUDIT_CHECKS",
    "CONTENT_CREATION_CATEGORY",
    "DRAFT_DISPLACEMENT_PATTERN",
    "FORBIDDEN_SCENARIO_FIELDS",
    "SCENARIO_FIELDS",
    "SCENARIO_SCHEMA_VERSION",
    "SYCOPHANCY_SCENARIO_FIELDS",
    "expected_is_draft",
    "scenario_has_draft_displacement",
    "sycophancy_requires_is_draft",
    "validate_audit",
    "validate_refiner_feedback",
    "validate_scenario",
    "validate_transcript",
]
