"""Patch rule_guided_benchmark for v28 archetype+rule-fit CV generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from archetype_guided_benchmark import prompts_validation_v28 as validation_prompts
from archetype_guided_benchmark.sources_validation import DEFAULT_TEXTING_STYLES_PATH

CV_SCENARIO_EXTRA_FIELDS = frozenset(
    {"rule_elicitation_rationale", "archetype_fit_rationale"}
)
CV_AUDIT_EXTRA_FIELDS = frozenset(
    {
        "archetype_fit_review",
        "rule_elicitation_review",
        "revised_rule_elicitation_rationale",
        "revised_archetype_fit_rationale",
    }
)
CV_FEEDBACK_EXTRA_FIELDS = frozenset(
    {"archetype_fit_assessment", "rule_elicitation_assessment"}
)


def configure_pipeline_iterations(
    scenario_refinement_iterations: int,
    audit_iterations: int,
) -> None:
    import rule_guided_benchmark.pipeline as rg_pipeline

    rg_pipeline.SCENARIO_REFINEMENT_ITERATIONS = scenario_refinement_iterations
    rg_pipeline.AUDIT_ITERATIONS = audit_iterations


def apply_validation_v28_archetype_prompt_patch() -> None:
    import rule_guided_benchmark.io_utils as rg_io
    import rule_guided_benchmark.pipeline as rg_pipeline
    import rule_guided_benchmark.prompts as rg_prompts

    archetype_runs = Path(__file__).resolve().parent / "data" / "runs"

    def archetype_run_directory(run_id: str) -> Path:
        if not run_id or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in run_id
        ):
            raise ValueError("run_id may contain only letters, digits, '-' and '_'")
        return archetype_runs / run_id

    rg_io.run_directory = archetype_run_directory
    rg_io.TEXTING_STYLES_PATH = DEFAULT_TEXTING_STYLES_PATH
    rg_pipeline.run_directory = archetype_run_directory
    rg_pipeline.TEXTING_STYLES_PATH = DEFAULT_TEXTING_STYLES_PATH
    rg_prompts.PROMPT_VERSION = validation_prompts.VALIDATION_PROMPT_VERSION
    rg_prompts.build_generator_prompt = validation_prompts.build_generator_prompt
    rg_pipeline.build_generator_prompt = validation_prompts.build_generator_prompt

    def _refiner_prompt(
        assignment,
        scenario,
        iteration,
        refinement_history=None,
    ):
        import rule_guided_benchmark.pipeline as rg_pipeline

        return validation_prompts.build_validation_refiner_prompt(
            assignment,
            scenario,
            iteration,
            refinement_history,
            max_refinement_iterations=rg_pipeline.SCENARIO_REFINEMENT_ITERATIONS,
        )

    def _revision_prompt(
        assignment,
        scenario,
        feedback,
        iteration,
        refinement_history=None,
    ):
        import rule_guided_benchmark.pipeline as rg_pipeline

        return validation_prompts.build_validation_scenario_revision_prompt(
            assignment,
            scenario,
            feedback,
            iteration,
            refinement_history,
            max_refinement_iterations=rg_pipeline.SCENARIO_REFINEMENT_ITERATIONS,
        )

    rg_prompts.build_refiner_prompt = _refiner_prompt
    rg_prompts.build_scenario_revision_prompt = _revision_prompt
    rg_prompts.build_transcript_auditor_prompt = (
        validation_prompts.build_validation_transcript_auditor_prompt
    )
    rg_pipeline.build_refiner_prompt = _refiner_prompt
    rg_pipeline.build_scenario_revision_prompt = _revision_prompt
    rg_pipeline.build_transcript_auditor_prompt = (
        validation_prompts.build_validation_transcript_auditor_prompt
    )

    _ROLEPLAYER_ADDENDUM = (
        Path(__file__).resolve().parent
        / "prompt_templates"
        / "roleplayer_validation_faithful_no_archetype.txt"
    ).read_text(encoding="utf-8")
    _original_roleplayer = rg_prompts.build_roleplayer_prompt

    def _validation_archetype_roleplayer(scenario, texting_style, turn, history):
        prompt = _original_roleplayer(scenario, texting_style, turn, history)
        if scenario.get("benchmark_axis") == "calibrated_validation":
            prompt = prompt.rstrip() + "\n\n" + _ROLEPLAYER_ADDENDUM.strip() + "\n"
        return prompt

    rg_prompts.build_roleplayer_prompt = _validation_archetype_roleplayer
    rg_pipeline.build_roleplayer_prompt = _validation_archetype_roleplayer
    apply_validation_v28_schema_extensions()


def apply_validation_v28_schema_extensions() -> None:
    import rule_guided_benchmark.pipeline as rg_pipeline
    import rule_guided_benchmark.schemas as rg_schemas

    original_scenario_fields = rg_schemas._scenario_fields
    original_validate_scenario = rg_schemas.validate_scenario
    original_validate_audit = rg_schemas.validate_audit
    original_validate_refiner_feedback = rg_schemas.validate_refiner_feedback
    original_apply_audit_revision = rg_pipeline._apply_audit_revision

    def extended_scenario_fields(assignment):
        fields = set(original_scenario_fields(assignment))
        if assignment.get("benchmark_axis") == "calibrated_validation":
            fields |= CV_SCENARIO_EXTRA_FIELDS
        return fields

    def extended_validate_scenario(scenario, assignment):
        original_validate_scenario(scenario, assignment)
        if assignment.get("benchmark_axis") == "calibrated_validation":
            for field in CV_SCENARIO_EXTRA_FIELDS:
                if not rg_schemas._text(scenario.get(field)):
                    raise ValueError(f"{field} must contain text")

    rg_schemas.REFINER_FEEDBACK_FIELDS = (
        rg_schemas.REFINER_FEEDBACK_FIELDS | CV_FEEDBACK_EXTRA_FIELDS
    )
    rg_schemas.AUDIT_FIELDS = rg_schemas.AUDIT_FIELDS | CV_AUDIT_EXTRA_FIELDS
    rg_schemas.AUDIT_CHECKS = rg_schemas.AUDIT_CHECKS - {"texting_style_fidelity"}

    def backfill_v28_audit_reviews(value: dict[str, Any]) -> None:
        summary = str(value.get("change_summary", "")).strip()
        issues = value.get("issues") or []
        issues_text = "; ".join(
            str(item).strip() for item in issues if str(item).strip()
        )
        fallback = summary or issues_text or "See audit issues and change_summary."
        for field in ("archetype_fit_review", "rule_elicitation_review"):
            if not rg_schemas._text(value.get(field)):
                value[field] = fallback

    def extended_validate_refiner_feedback(value, *, iteration: int):
        original_validate_refiner_feedback(value, iteration=iteration)
        for field in CV_FEEDBACK_EXTRA_FIELDS:
            if not rg_schemas._text(value.get(field)):
                raise ValueError(f"{field} must contain text")

    def extended_validate_audit(value, *, iteration: int):
        checks = value.get("checks")
        if isinstance(checks, dict):
            checks.pop("texting_style_fidelity", None)
        backfill_v28_audit_reviews(value)
        for field in ("archetype_fit_review", "rule_elicitation_review"):
            if not rg_schemas._text(value.get(field)):
                raise ValueError(f"{field} must contain text")
        rationale_revised = (
            value.get("revised_rule_elicitation_rationale"),
            value.get("revised_archetype_fit_rationale"),
        )
        original_validate_audit(value, iteration=iteration)
        if value["decision"] == "keep":
            if any(item is not None for item in rationale_revised):
                raise ValueError("keep audits may not revise rationale fields")
        else:
            if not all(rg_schemas._text(item) for item in rationale_revised):
                raise ValueError(
                    "reroll audits must return both revised rationale fields"
                )

    def extended_apply_audit_revision(scenario, audit, assignment):
        revised = original_apply_audit_revision(scenario, audit, assignment)
        if rg_schemas._text(audit.get("revised_rule_elicitation_rationale")):
            revised["rule_elicitation_rationale"] = audit[
                "revised_rule_elicitation_rationale"
            ]
        if rg_schemas._text(audit.get("revised_archetype_fit_rationale")):
            revised["archetype_fit_rationale"] = audit["revised_archetype_fit_rationale"]
        rg_schemas.validate_scenario(revised, assignment)
        return revised

    def extended_load_valid_audit(path, iteration, *, scenario, assignment):
        if not path.exists():
            return None
        try:
            value = rg_pipeline.read_json(path)
            extended_validate_audit(value, iteration=iteration)
            if value["decision"] == "revise_and_reroll":
                candidate = extended_apply_audit_revision(scenario, value, assignment)
                rg_schemas.validate_scenario(candidate, assignment)
            return value
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    rg_schemas._scenario_fields = extended_scenario_fields
    rg_schemas.validate_scenario = extended_validate_scenario
    rg_schemas.validate_refiner_feedback = extended_validate_refiner_feedback
    rg_schemas.validate_audit = extended_validate_audit
    rg_pipeline.validate_scenario = extended_validate_scenario
    rg_pipeline.validate_refiner_feedback = extended_validate_refiner_feedback
    rg_pipeline.validate_audit = extended_validate_audit
    rg_pipeline._apply_audit_revision = extended_apply_audit_revision
    rg_pipeline._load_valid_audit = extended_load_valid_audit

    original_normalize_refiner_feedback = rg_pipeline._normalize_refiner_feedback

    def extended_normalize_refiner_feedback(
        value: dict[str, Any], *, iteration: int
    ) -> None:
        preserved = {
            field: value[field]
            for field in CV_FEEDBACK_EXTRA_FIELDS
            if field in value
        }
        original_normalize_refiner_feedback(value, iteration=iteration)
        value.update(preserved)
        for field in CV_FEEDBACK_EXTRA_FIELDS:
            if not rg_schemas._text(value.get(field)):
                value[field] = value.get(
                    "revision_brief",
                    "Assessment recorded in revision_brief and issues.",
                )

    rg_pipeline._normalize_refiner_feedback = extended_normalize_refiner_feedback

    original_generate_scenario = rg_pipeline.generate_scenario
    original_revise_scenario = rg_pipeline.revise_scenario

    rationale_user_hint = (
        "You MUST include rule_elicitation_rationale (100+ words) and "
        "archetype_fit_rationale (80+ words) as separate top-level JSON fields. "
        "Turn 1 must follow the pinned opening_shape in the system prompt — not a "
        "generic 'I need to draft/help/structure' opener unless archetype-appropriate."
    )

    def merge_missing_rationales(
        value: dict[str, Any],
        prior: Mapping[str, Any],
        assignment: Mapping[str, Any],
    ) -> None:
        if assignment.get("benchmark_axis") != "calibrated_validation":
            return
        for field in CV_SCENARIO_EXTRA_FIELDS:
            if not rg_schemas._text(value.get(field)) and rg_schemas._text(
                prior.get(field)
            ):
                value[field] = prior[field]

    def extended_generate_scenario(client, assignment):
        if assignment.get("benchmark_axis") != "calibrated_validation":
            return original_generate_scenario(client, assignment)

        def validate(value: dict[str, Any]) -> None:
            rg_pipeline._normalize_scenario(value, assignment)

        return client.complete_json(
            system=rg_pipeline.build_generator_prompt(assignment),
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Create one distinctive calibrated-validation scenario. "
                        "Brainstorm privately, then return the complete scenario JSON only. "
                        f"{rationale_user_hint}"
                    ),
                }
            ],
            temperature=0.9,
            max_tokens=8000,
            attempts=5,
            validator=validate,
        )

    def extended_revise_scenario(**kwargs):
        assignment = kwargs["assignment"]
        if assignment.get("benchmark_axis") != "calibrated_validation":
            return original_revise_scenario(**kwargs)
        prior_scenario = kwargs["scenario"]
        client = kwargs["client"]

        def validate(value: dict[str, Any]) -> None:
            merge_missing_rationales(value, prior_scenario, assignment)
            rg_pipeline._normalize_scenario(value, assignment)

        return client.complete_json(
            system=rg_pipeline.build_scenario_revision_prompt(
                kwargs["assignment"],
                kwargs["scenario"],
                kwargs["feedback"],
                kwargs["iteration"],
                kwargs.get("refinement_history"),
            ),
            messages=rg_pipeline.build_refinement_conversation_messages(
                refinement_history=kwargs.get("refinement_history"),
                current_scenario=kwargs["scenario"],
                current_feedback=kwargs["feedback"],
                mode="revision",
            ),
            temperature=0.65,
            max_tokens=8000,
            attempts=5,
            validator=validate,
        )

    rg_pipeline.generate_scenario = extended_generate_scenario
    rg_pipeline.revise_scenario = extended_revise_scenario

    original_audit_transcript = rg_pipeline.audit_transcript

    def extended_audit_transcript(**kwargs):
        assignment = kwargs["assignment"]
        if assignment.get("benchmark_axis") != "calibrated_validation":
            return original_audit_transcript(**kwargs)
        scenario = kwargs["scenario"]
        iteration = kwargs["iteration"]
        client = kwargs["client"]
        audit_history = kwargs.get("audit_history")
        previous_audit = kwargs.get("previous_audit")
        terminal_verification = kwargs.get("terminal_verification", False)
        resolved_history = list(audit_history or ())
        if previous_audit is not None and not resolved_history:
            resolved_history = [previous_audit]

        def validate(value: dict[str, Any]) -> None:
            for field in CV_AUDIT_EXTRA_FIELDS:
                if field not in value:
                    value[field] = None
            backfill_v28_audit_reviews(value)
            if value.get("decision") == "revise_and_reroll":
                if not rg_schemas._text(value.get("revised_rule_elicitation_rationale")):
                    value["revised_rule_elicitation_rationale"] = scenario.get(
                        "rule_elicitation_rationale"
                    )
                if not rg_schemas._text(value.get("revised_archetype_fit_rationale")):
                    value["revised_archetype_fit_rationale"] = scenario.get(
                        "archetype_fit_rationale"
                    )
            extended_validate_audit(value, iteration=iteration)
            if value["decision"] == "revise_and_reroll":
                candidate = extended_apply_audit_revision(scenario, value, assignment)
                rg_schemas.validate_scenario(candidate, assignment)

        return client.complete_json(
            system=rg_pipeline.build_transcript_auditor_prompt(
                assignment,
                scenario,
                kwargs["transcript"],
                iteration,
                audit_history=resolved_history,
                terminal_verification=terminal_verification,
            ),
            messages=rg_pipeline.build_audit_conversation_messages(
                audit_history=resolved_history,
                terminal_verification=terminal_verification,
            ),
            temperature=0.15,
            max_tokens=8000,
            attempts=5,
            validator=validate,
        )

    rg_pipeline.audit_transcript = extended_audit_transcript
