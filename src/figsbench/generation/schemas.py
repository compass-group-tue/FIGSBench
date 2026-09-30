"""Strict contracts for authored scenarios, refiner reviews, audits and transcripts."""

from __future__ import annotations

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
RATIONALE_FIELDS = ("rule_elicitation_rationale", "archetype_fit_rationale")
# Sycophancy scenarios also carry the archetype id; the three are checked together.
SYCOPHANCY_EXTRA_FIELDS = frozenset(
    {"rule_elicitation_rationale", "archetype_fit_rationale", "conversation_archetype_id"}
)
VALIDATION_EXTRA_FIELDS = frozenset(RATIONALE_FIELDS)

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

REFINER_FEEDBACK_FIELDS = {
    "iteration",
    "verdict",
    "strengths",
    "issues",
    "revision_brief",
    "must_preserve",
}
VALIDATION_FEEDBACK_EXTRA_FIELDS = frozenset(
    {"archetype_fit_assessment", "rule_elicitation_assessment"}
)

AUDIT_FIELDS = {
    "iteration",
    "decision",
    "checks",
    "issues",
    "change_summary",
    "revised_user_role",
    "revised_scenario_plan",
}
SYCOPHANCY_AUDIT_EXTRA_FIELDS = frozenset(
    {"revised_rule_elicitation_rationale", "revised_archetype_fit_rationale"}
)
VALIDATION_AUDIT_EXTRA_FIELDS = frozenset(
    {
        "archetype_fit_review",
        "rule_elicitation_review",
        "revised_rule_elicitation_rationale",
        "revised_archetype_fit_rationale",
    }
)
AUDIT_CHECKS = {
    "ten_turn_structure",
    "human_realism",
    "conversation_contingency",
    "plan_abstraction",
    "plan_independence",
    "facts_visible_before_use",
    "selected_opportunity_reached",
    "difficulty_realized",
    "axis_purity",
    "no_benchmark_language",
}


def text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def scenario_fields(assignment: Mapping[str, Any]) -> set[str]:
    if assignment.get("benchmark_axis") == "sycophancy":
        return set(SCENARIO_FIELDS) | SYCOPHANCY_EXTRA_FIELDS
    return set(SCENARIO_FIELDS) | VALIDATION_EXTRA_FIELDS


def validate_scenario(scenario: Mapping[str, Any], assignment: Mapping[str, Any]) -> None:
    """Field set, pinned assignment fields and non-empty authored prose."""
    expected_fields = scenario_fields(assignment)
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
    if not text(scenario["user_role"]):
        raise ValueError("user_role must be concrete prose")
    if not text(scenario["scenario_plan"]):
        raise ValueError("scenario_plan must be concrete prose")


def validate_scenario_strict(
    scenario: Mapping[str, Any], assignment: Mapping[str, Any]
) -> None:
    """validate_scenario plus non-empty rationale (and, for sycophancy, archetype id) fields."""
    validate_scenario(scenario, assignment)
    extra = (
        SYCOPHANCY_EXTRA_FIELDS
        if assignment.get("benchmark_axis") == "sycophancy"
        else VALIDATION_EXTRA_FIELDS
    )
    for field in extra:
        if not text(scenario.get(field)):
            raise ValueError(f"{field} must contain text")


def validate_refiner_feedback(
    value: Mapping[str, Any], *, iteration: int, extra_fields: frozenset[str] = frozenset()
) -> None:
    if set(value) != REFINER_FEEDBACK_FIELDS | extra_fields:
        raise ValueError("refiner feedback has the wrong fields")
    if value["iteration"] != iteration:
        raise ValueError("refiner feedback iteration mismatch")
    if value["verdict"] not in {"ready", "revise"}:
        raise ValueError("refiner verdict must be ready or revise")
    for field in ("strengths", "issues", "must_preserve"):
        if not isinstance(value[field], list):
            raise ValueError(f"refiner {field} must be a list")
    if not all(text(item) for item in value["strengths"]):
        raise ValueError("refiner strengths must contain text")
    if not all(text(item) for item in value["must_preserve"]):
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
        if not all(text(issue[field]) for field in issue):
            raise ValueError("refiner issue fields must contain text")
    if not text(value["revision_brief"]):
        raise ValueError("refiner revision_brief must contain text")
    for field in extra_fields:
        if not text(value.get(field)):
            raise ValueError(f"{field} must contain text")


def validate_audit(
    value: Mapping[str, Any], *, iteration: int, extra_fields: frozenset[str]
) -> None:
    if set(value) != AUDIT_FIELDS | extra_fields:
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
        text(item) for item in value["issues"]
    ):
        raise ValueError("audit issues must be a text list")
    if not text(value["change_summary"]):
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
        if not all(text(item) for item in revised):
            raise ValueError("reroll audits must return both revised scenario fields")


def validate_audit_rationales(value: Mapping[str, Any]) -> None:
    """Keep audits leave the rationales alone; reroll audits revise both."""
    rationale_revised = (
        value.get("revised_rule_elicitation_rationale"),
        value.get("revised_archetype_fit_rationale"),
    )
    if value["decision"] == "keep":
        if any(item is not None for item in rationale_revised):
            raise ValueError("keep audits may not revise rationale fields")
    else:
        if not all(text(item) for item in rationale_revised):
            raise ValueError(
                "reroll audits must return both revised rationale fields"
            )


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
            or not text(turn.get("content"))
        ):
            raise ValueError(f"invalid transcript turn {index}")
