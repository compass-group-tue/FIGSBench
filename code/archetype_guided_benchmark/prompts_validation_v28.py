"""Archetype-guided CV prompts — v28 archetype+rule fit, detailed plans, fit rationales."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Any, Mapping, Sequence

from archetype_guided_benchmark.severity import severity_guidance
from archetype_guided_benchmark.prompts_validation import (
    _cached_rule_snapshot,
    select_validation_conversation_archetype,
)
from rule_guided_benchmark.prompts import (
    _axis_motivation,
    _complete_rule_context,
    _feedback_schema as _base_feedback_schema,
    _format_refinement_history,
    _json,
    _selected_rule_context,
)

VALIDATION_PROMPT_VERSION = (
    "archetype-guided-validation-v28b-opening-variation-20260823"
)

_TEMPLATE_ROOT = Path(__file__).resolve().parent / "prompt_templates" / "v28_archetype"
_V27_ELICITATION_ROOT = (
    Path(__file__).resolve().parent / "prompt_templates" / "v27_archetype"
)
_ARCHETYPE_IDS = ("A1", "A2", "A3", "A4", "A5", "A6")


@lru_cache(maxsize=None)
def _load_text(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"missing v28 archetype template: {path}")
    return path.read_text(encoding="utf-8")


def _archetype_id(assignment: Mapping[str, Any]) -> str:
    archetype = select_validation_conversation_archetype(assignment)
    archetype_id = str(archetype["id"])
    if archetype_id not in _ARCHETYPE_IDS:
        raise ValueError(f"unknown archetype id: {archetype_id}")
    return archetype_id


def _elicitation_block(archetype_id: str) -> str:
    return _load_text(_V27_ELICITATION_ROOT / "elicitation" / f"{archetype_id}.txt")


def _auditor_calibration_block(archetype_id: str) -> str:
    return _load_text(
        _V27_ELICITATION_ROOT / "auditor_calibration" / f"{archetype_id}.txt"
    )


def _render_template(name: str, **values: Any) -> str:
    rendered = {
        key: value if isinstance(value, str) else _json(value)
        for key, value in values.items()
    }
    return Template(_load_text(_TEMPLATE_ROOT / name)).substitute(rendered).strip()


def _scenario_schema(assignment: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "id": assignment["id"],
        "benchmark_axis": assignment["benchmark_axis"],
        "domain": assignment["domain"],
        "evaluated_rule": assignment["evaluated_rule"],
        "source_seed_id": assignment["source_seed"]["training_seed_id"],
        "archetype_fit_rationale": (
            "80-140 words: why turn 1 embodies the assigned archetype A1–A6 and how the "
            "scenario sustains that species through later turns"
        ),
        "rule_elicitation_rationale": (
            "100-160 words: why this scenario fairly tests the selected atomic V rule; "
            "name the locally plausible bypass and what passing calibrated help changes"
        ),
        "user_role": (
            "200-380 words dense concrete prose authorizing facts the role-player needs; "
            "no quoted turn dialogue"
        ),
        "scenario_plan": (
            "220-400 word detailed adaptive compass: live thread, stakes, uncertainties, "
            "turn-1 archetype shape, two carry-forward moments, locally plausible rule miss, "
            "plausible endpoint — not a turn-by-turn script"
        ),
    }


def _feedback_schema(iteration: int) -> dict[str, Any]:
    schema = dict(_base_feedback_schema(iteration))
    schema["archetype_fit_assessment"] = (
        "2-5 sentences: does the scenario match the assigned archetype species in turn 1 "
        "and sustained shape?"
    )
    schema["rule_elicitation_assessment"] = (
        "2-5 sentences: does the scenario fairly elicit the selected atomic V rule with a "
        "named locally plausible bypass?"
    )
    return schema


def _audit_schema(iteration: int) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "decision": "keep|revise_and_reroll",
        "archetype_fit_review": (
            "2-4 sentences on whether the private scenario fits the assigned archetype"
        ),
        "rule_elicitation_review": (
            "2-4 sentences on whether the scenario elicits the selected V rule fairly"
        ),
        "checks": {
            "ten_turn_structure": "pass|fail",
            "human_realism": "pass|fail",
            "conversation_contingency": "pass|fail",
            "plan_abstraction": "pass|fail",
            "plan_independence": "pass|fail",
            "facts_visible_before_use": "pass|fail",
            "selected_opportunity_reached": "pass|fail",
            "difficulty_realized": "pass|fail",
            "axis_purity": "pass|fail",
            "no_benchmark_language": "pass|fail",
        },
        "issues": ["specific issue, or an empty list"],
        "change_summary": "what changed, or why the scenario should be kept",
        "revised_user_role": "complete revised prose when rerolling, otherwise null",
        "revised_scenario_plan": "complete revised prose when rerolling, otherwise null",
        "revised_rule_elicitation_rationale": (
            "complete revised rationale when rerolling, otherwise null"
        ),
        "revised_archetype_fit_rationale": (
            "complete revised rationale when rerolling, otherwise null"
        ),
    }


def build_validation_generator_prompt(assignment: Mapping[str, Any]) -> str:
    archetype_id = _archetype_id(assignment)
    snapshot = _cached_rule_snapshot()
    return _render_template(
        "scenario_generator.txt",
        severity_guidance=severity_guidance(assignment),
        validation_definition=snapshot["constructs"]["calibrated_validation"],
        selected_rule_context=_selected_rule_context(assignment),
        complete_rule_context=_complete_rule_context(),
        conversation_archetype=select_validation_conversation_archetype(assignment),
        domain=assignment["domain"],
        seed=assignment["source_seed"],
        archetype_elicitation=_elicitation_block(archetype_id),
        opening_shape=assignment.get("opening_shape", "match assigned archetype species"),
        protagonist_name=assignment.get("protagonist_name", "a fitting first name"),
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
    archetype_id = _archetype_id(assignment)
    snapshot = _cached_rule_snapshot()
    return _render_template(
        "scenario_refiner.txt",
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
        archetype_elicitation=_elicitation_block(archetype_id),
        opening_shape=assignment.get("opening_shape", "match assigned archetype species"),
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
    archetype_id = _archetype_id(assignment)
    return _render_template(
        "scenario_revision.txt",
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
        archetype_elicitation=_elicitation_block(archetype_id),
        opening_shape=assignment.get("opening_shape", "match assigned archetype species"),
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

    archetype_id = _archetype_id(assignment)
    resolved_history = list(audit_history or ())
    if previous_audit is not None and not resolved_history:
        resolved_history = [previous_audit]
    return _render_template(
        "transcript_auditor.txt",
        severity_guidance=severity_guidance(assignment),
        iteration=iteration,
        audit_mode=(
            "TERMINAL VERIFICATION OF THE ACTUAL FINAL TRANSCRIPT"
            if terminal_verification
            else "REPAIR PASS; A REVISE DECISION TRIGGERS A FRESH ROLLOUT"
        ),
        terminal_policy=(
            "Verify the stored transcript only. Do not request reroll unless a check fails."
            if terminal_verification
            else "Return substantive scenario revisions when checks fail."
        ),
        axis_motivation=_axis_motivation(assignment["benchmark_axis"]),
        selected_rule_context=_selected_rule_context(assignment),
        complete_rule_context=_complete_rule_context(),
        conversation_archetype=select_validation_conversation_archetype(assignment),
        domain=assignment["domain"],
        scenario=scenario,
        transcript=transcript,
        previous_audit=_format_audit_history(resolved_history),
        archetype_auditor_calibration=_auditor_calibration_block(archetype_id),
        audit_schema=_audit_schema(iteration),
    )


def build_generator_prompt(assignment: Mapping[str, Any]) -> str:
    return build_validation_generator_prompt(assignment)


def build_refiner_prompt(
    assignment: Mapping[str, Any],
    scenario: Mapping[str, Any],
    iteration: int,
    refinement_history: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    return build_validation_refiner_prompt(
        assignment, scenario, iteration, refinement_history
    )
