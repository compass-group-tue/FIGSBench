"""Archetype-guided calibrated-validation authoring prompts (A1–A6 conversation species)."""

from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Any, Mapping, Sequence

from .severity import severity_guidance

from rule_guided_benchmark.prompts import (
    _audit_schema,
    _axis_motivation,
    _complete_rule_context,
    _feedback_schema,
    _format_refinement_history,
    _json,
    _scenario_schema,
    _selected_rule_context,
    _style_description,
    RULE_SNAPSHOT_VERSION,
)

VALIDATION_PROMPT_VERSION = "archetype-guided-validation-v4-full-history-20260815"

_PACKAGE_ROOT = Path(__file__).resolve().parent
_TEMPLATE_ROOT = _PACKAGE_ROOT / "prompt_templates"
_ARCHETYPE_PATH = _PACKAGE_ROOT / "config" / "validation_conversation_archetypes.json"
_RULE_SNAPSHOT_PATH = (
    _PACKAGE_ROOT.parent / "rule_guided_benchmark" / "config" / "appendix_rules_v2.json"
)

_TEMPLATE_NAMES = {
    "scenario_generator": "scenario_generator_validation_archetype_v1.txt",
    "scenario_refiner": "scenario_refiner_validation_archetype_v1.txt",
    "scenario_revision": "scenario_revision_validation_archetype_v1.txt",
    "transcript_auditor": "transcript_auditor_validation_archetype_v1.txt",
}


@lru_cache(maxsize=1)
def load_validation_conversation_archetypes() -> tuple[dict[str, Any], ...]:
    with _ARCHETYPE_PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return tuple(deepcopy(item) for item in payload["archetypes"])


@lru_cache(maxsize=1)
def _cached_rule_snapshot() -> dict[str, Any]:
    with _RULE_SNAPSHOT_PATH.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if value.get("rule_snapshot_version") != RULE_SNAPSHOT_VERSION:
        raise ValueError("appendix rule snapshot version mismatch")
    return value


@lru_cache(maxsize=None)
def _load_template(name: str) -> Template:
    path = _TEMPLATE_ROOT / name
    if not path.is_file():
        raise ValueError(f"unknown validation archetype prompt template: {name}")
    return Template(path.read_text(encoding="utf-8"))


def _render(template_key: str, **values: Any) -> str:
    name = _TEMPLATE_NAMES[template_key]
    rendered = {
        key: value if isinstance(value, str) else _json(value)
        for key, value in values.items()
    }
    return _load_template(name).substitute(rendered).strip()


def select_validation_conversation_archetype(
    assignment: Mapping[str, Any],
) -> dict[str, Any]:
    if assignment.get("benchmark_axis") != "calibrated_validation":
        raise ValueError("validation conversation archetypes are CV-only")
    pinned = assignment.get("conversation_archetype")
    if not isinstance(pinned, Mapping):
        raise ValueError("assignment lacks conversation_archetype")
    archetype_id = str(pinned.get("id", ""))
    index = {item["id"]: item for item in load_validation_conversation_archetypes()}
    if archetype_id not in index:
        raise ValueError(f"unknown conversation_archetype id: {archetype_id}")
    canonical = index[archetype_id]
    return {
        "id": canonical["id"],
        "name": canonical["name"],
        "definition": canonical["definition"],
        "authoring_brief": canonical["authoring_brief"],
        "user_correct_on_focal_question": canonical["user_correct_on_focal_question"],
    }


def build_validation_generator_prompt(assignment: Mapping[str, Any]) -> str:
    snapshot = _cached_rule_snapshot()
    return _render(
        "scenario_generator",
        severity_guidance=severity_guidance(assignment),
        validation_definition=snapshot["constructs"]["calibrated_validation"],
        selected_rule_context=_selected_rule_context(assignment),
        complete_rule_context=_complete_rule_context(),
        conversation_archetype=select_validation_conversation_archetype(assignment),
        domain=assignment["domain"],
        seed=assignment["source_seed"],
        scenario_schema=_scenario_schema(assignment),
    )


def build_validation_refiner_prompt(
    assignment: Mapping[str, Any],
    scenario: Mapping[str, Any],
    iteration: int,
    refinement_history: Sequence[Mapping[str, Any]] | None = None,
    *,
    max_refinement_iterations: int = 5,
) -> str:
    snapshot = _cached_rule_snapshot()
    return _render(
        "scenario_refiner",
        severity_guidance=severity_guidance(assignment),
        iteration=iteration,
        max_refinement_iterations=max_refinement_iterations,
        validation_definition=snapshot["constructs"]["calibrated_validation"],
        selected_rule_context=_selected_rule_context(assignment),
        complete_rule_context=_complete_rule_context(),
        conversation_archetype=select_validation_conversation_archetype(assignment),
        domain_and_seed={
            "domain": assignment["domain"],
            "source_seed": assignment["source_seed"],
        },
        scenario=scenario,
        refinement_history=_format_refinement_history(refinement_history),
        feedback_schema=_feedback_schema(iteration),
    )


def build_validation_scenario_revision_prompt(
    assignment: Mapping[str, Any],
    scenario: Mapping[str, Any],
    feedback: Mapping[str, Any],
    iteration: int,
    refinement_history: Sequence[Mapping[str, Any]] | None = None,
    *,
    max_refinement_iterations: int = 5,
) -> str:
    return _render(
        "scenario_revision",
        severity_guidance=severity_guidance(assignment),
        iteration=iteration,
        max_refinement_iterations=max_refinement_iterations,
        axis_motivation=_axis_motivation(assignment["benchmark_axis"]),
        selected_rule_context=_selected_rule_context(assignment),
        complete_rule_context=_complete_rule_context(),
        conversation_archetype=select_validation_conversation_archetype(assignment),
        domain=assignment["domain"],
        seed=assignment["source_seed"],
        scenario=scenario,
        refinement_history=_format_refinement_history(refinement_history),
        feedback=feedback,
        scenario_schema=_scenario_schema(assignment),
    )


def build_validation_transcript_auditor_prompt(
    assignment: Mapping[str, Any],
    scenario: Mapping[str, Any],
    transcript: Mapping[str, Any],
    iteration: int,
    audit_history: Sequence[Mapping[str, Any]] | None = None,
    previous_audit: Mapping[str, Any] | None = None,
    terminal_verification: bool = False,
) -> str:
    from rule_guided_benchmark.prompts import _format_audit_history

    resolved_history = list(audit_history or ())
    if previous_audit is not None and not resolved_history:
        resolved_history = [previous_audit]
    return _render(
        "transcript_auditor",
        severity_guidance=severity_guidance(assignment),
        iteration=iteration,
        audit_mode=(
            "TERMINAL VERIFICATION OF THE ACTUAL FINAL TRANSCRIPT"
            if terminal_verification
            else "REPAIR PASS; A REVISE DECISION TRIGGERS A FRESH ROLLOUT"
        ),
        terminal_policy=(
            "This pass verifies the stored transcript only. Do not request reroll "
            "unless a check fails."
            if terminal_verification
            else "Return substantive scenario revisions when checks fail."
        ),
        axis_motivation=_axis_motivation(assignment["benchmark_axis"]),
        selected_rule_context=_selected_rule_context(assignment),
        complete_rule_context=_complete_rule_context(),
        conversation_archetype=select_validation_conversation_archetype(assignment),
        domain=assignment["domain"],
        texting_style=_style_description(assignment["texting_style"]),
        scenario=scenario,
        transcript=transcript,
        previous_audit=_format_audit_history(resolved_history),
        audit_schema=_audit_schema(iteration),
    )


def build_generator_prompt(assignment: Mapping[str, Any]) -> str:
    if assignment["benchmark_axis"] != "calibrated_validation":
        raise ValueError("validation prompts support calibrated_validation only")
    return build_validation_generator_prompt(assignment)


def build_refiner_prompt(
    assignment: Mapping[str, Any],
    scenario: Mapping[str, Any],
    iteration: int,
    refinement_history: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    if assignment["benchmark_axis"] != "calibrated_validation":
        raise ValueError("validation prompts support calibrated_validation only")
    return build_validation_refiner_prompt(
        assignment, scenario, iteration, refinement_history
    )
