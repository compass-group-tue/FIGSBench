"""Resumable appendix-guided generation, rollout, and transcript auditing."""

from __future__ import annotations

import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from benchmark.pipeline.client import OpenRouterClient, build_client

from .io_utils import (
    APPENDIX_RULES_PATH,
    ASSISTANT_PROMPT_PATH,
    SEED_CORPUS_PATH,
    TEXTING_STYLES_PATH,
    file_hash,
    read_json,
    run_directory,
    stable_hash,
    utc_now,
    write_json,
)
from .prompts import (
    CALIBRATED_VALIDATION_PROMPT_VERSION,
    PROMPT_VERSION,
    build_generator_prompt,
    build_audit_conversation_messages,
    build_refiner_prompt,
    build_roleplayer_prompt,
    build_scenario_revision_prompt,
    build_refinement_conversation_messages,
    build_transcript_auditor_prompt,
    refinement_history_record,
)
from .schemas import (
    SCENARIO_SCHEMA_VERSION,
    validate_audit,
    validate_refiner_feedback,
    validate_scenario,
    validate_transcript,
)
from .sources import (
    build_plan,
    build_sycophancy_plan,
    build_validation_plan,
    validate_plan,
    validate_sycophancy_plan,
    validate_validation_plan,
)


DEFAULT_GENERATOR_MODEL = "moonshotai/kimi-k2.6"
DEFAULT_REFINER_MODEL = "moonshotai/kimi-k2.6"
DEFAULT_AUDITOR_MODEL = "moonshotai/kimi-k2.6"
DEFAULT_ASSISTANT_MODEL = "moonshotai/kimi-k2.6"
DEFAULT_USER_MODEL = "deepseek/deepseek-v4-pro"
RUN_SCHEMA_VERSION = 2
SCENARIO_REFINEMENT_ITERATIONS = 5
AUDIT_ITERATIONS = 5
TERMINAL_AUDIT_ITERATIONS = 1


def prompt_version_for_run(benchmark_axis: str) -> str:
    from . import prompts

    if benchmark_axis == "sycophancy":
        return prompts.PROMPT_VERSION
    if benchmark_axis == "calibrated_validation":
        return prompts.CALIBRATED_VALIDATION_PROMPT_VERSION
    if benchmark_axis == "both":
        return (
            f"{prompts.PROMPT_VERSION}+{prompts.CALIBRATED_VALIDATION_PROMPT_VERSION}"
        )
    raise ValueError(f"unsupported benchmark axis: {benchmark_axis}")


def _normalize_scenario(
    value: dict[str, Any], assignment: Mapping[str, Any]
) -> None:
    """Pin immutable assignment data while leaving only two authored fields."""
    value["schema_version"] = SCENARIO_SCHEMA_VERSION
    value["id"] = deepcopy(assignment["id"])
    value["benchmark_axis"] = deepcopy(assignment["benchmark_axis"])
    value["domain"] = deepcopy(assignment["domain"])
    value["evaluated_rule"] = deepcopy(assignment["evaluated_rule"])
    value["source_seed_id"] = assignment["source_seed"]["training_seed_id"]
    if assignment.get("benchmark_axis") == "calibrated_validation":
        if "difficulty_tier" in assignment:
            value["difficulty_tier"] = deepcopy(assignment["difficulty_tier"])
    validate_scenario(value, assignment)


def generate_scenario(
    client: OpenRouterClient, assignment: Mapping[str, Any]
) -> dict[str, Any]:
    def validate(value: dict[str, Any]) -> None:
        _normalize_scenario(value, assignment)

    return client.complete_json(
        system=build_generator_prompt(assignment),
        messages=[
            {
                "role": "user",
                "content": (
                    "Create one distinctive scenario. Brainstorm privately, then "
                    "return the complete scenario JSON only."
                ),
            }
        ],
        temperature=0.9,
        max_tokens=6500,
        attempts=3,
        validator=validate,
    )


def review_scenario(
    *,
    client: OpenRouterClient,
    assignment: Mapping[str, Any],
    scenario: Mapping[str, Any],
    iteration: int,
    refinement_history: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    def validate(value: dict[str, Any]) -> None:
        _normalize_refiner_feedback(value, iteration=iteration)
        validate_refiner_feedback(value, iteration=iteration)

    return client.complete_json(
        system=build_refiner_prompt(
            assignment, scenario, iteration, refinement_history
        ),
        messages=build_refinement_conversation_messages(
            refinement_history=refinement_history,
            current_scenario=scenario,
            mode="refiner",
        ),
        temperature=0.25,
        max_tokens=4500,
        attempts=3,
        validator=validate,
    )


def _normalize_refiner_feedback(value: dict[str, Any], *, iteration: int) -> None:
    """Normalize common harmless JSON variations without changing the review."""
    value["iteration"] = iteration
    verdict = str(value.get("verdict", "revise")).strip().lower()
    value["verdict"] = "ready" if verdict in {"ready", "pass", "accept"} else "revise"

    strengths = value.get("strengths", [])
    if isinstance(strengths, str):
        strengths = [strengths]
    if not isinstance(strengths, list):
        strengths = []
    value["strengths"] = [str(item).strip() for item in strengths if str(item).strip()]

    raw_issues = value.get("issues", [])
    if isinstance(raw_issues, dict):
        raw_issues = [raw_issues]
    if not isinstance(raw_issues, list):
        raw_issues = []
    normalized_issues: list[dict[str, str]] = []
    for issue in raw_issues:
        if isinstance(issue, str) and issue.strip():
            normalized_issues.append(
                {
                    "dimension": "scenario_quality",
                    "severity": "major",
                    "explanation": issue.strip(),
                    "specific_change": issue.strip(),
                }
            )
            continue
        if not isinstance(issue, Mapping):
            continue
        explanation = str(
            issue.get("explanation") or issue.get("reason") or issue.get("issue") or ""
        ).strip()
        change = str(
            issue.get("specific_change")
            or issue.get("recommendation")
            or issue.get("fix")
            or explanation
        ).strip()
        if not explanation or not change:
            continue
        severity = str(issue.get("severity", "major")).strip().lower()
        normalized_issues.append(
            {
                "dimension": str(issue.get("dimension", "scenario_quality")).strip()
                or "scenario_quality",
                "severity": "minor" if severity == "minor" else "major",
                "explanation": explanation,
                "specific_change": change,
            }
        )
    value["issues"] = normalized_issues

    must_preserve = value.get("must_preserve", [])
    if isinstance(must_preserve, str):
        must_preserve = [must_preserve]
    if not isinstance(must_preserve, list):
        must_preserve = []
    value["must_preserve"] = [
        str(item).strip() for item in must_preserve if str(item).strip()
    ]

    brief = str(
        value.get("revision_brief")
        or value.get("highest_priority_change")
        or value.get("single_primary_change")
        or (
            normalized_issues[0]["specific_change"]
            if normalized_issues
            else "Preserve the scenario's strengths and make no substantive change."
        )
    ).strip()
    value["revision_brief"] = brief
    for key in list(value):
        if key not in {
            "iteration",
            "verdict",
            "strengths",
            "issues",
            "revision_brief",
            "must_preserve",
        }:
            del value[key]


def revise_scenario(
    *,
    client: OpenRouterClient,
    assignment: Mapping[str, Any],
    scenario: Mapping[str, Any],
    feedback: Mapping[str, Any],
    iteration: int,
    refinement_history: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    def validate(value: dict[str, Any]) -> None:
        _normalize_scenario(value, assignment)

    return client.complete_json(
        system=build_scenario_revision_prompt(
            assignment,
            scenario,
            feedback,
            iteration,
            refinement_history,
        ),
        messages=build_refinement_conversation_messages(
            refinement_history=refinement_history,
            current_scenario=scenario,
            current_feedback=feedback,
            mode="revision",
        ),
        temperature=0.65,
        max_tokens=6500,
        attempts=3,
        validator=validate,
    )


def _clean_user_message(value: str) -> str:
    message = value.strip()
    if message.startswith("```") and message.endswith("```"):
        message = "\n".join(message.splitlines()[1:-1]).strip()
    try:
        parsed = json.loads(message)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, str) and parsed.strip():
        message = parsed.strip()
    for prefix in ("User:", "USER:", "user:"):
        if message.startswith(prefix):
            message = message[len(prefix) :].strip()
            break
    if not message:
        raise ValueError("role-playing model returned an empty user message")
    return message


def run_conversation(
    *,
    assistant_client: OpenRouterClient,
    user_client: OpenRouterClient,
    scenario: Mapping[str, Any],
    texting_style: Mapping[str, Any],
    assistant_system_prompt: str,
    purpose: str,
) -> dict[str, Any]:
    history: list[dict[str, str]] = []
    for user_turn in range(1, 6):
        user_message = _clean_user_message(
            user_client.complete(
                system=build_roleplayer_prompt(
                    scenario, texting_style, user_turn, history
                ),
                messages=[
                    {
                        "role": "user",
                        "content": "Write the next in-character user message only.",
                    }
                ],
                temperature=0.85,
                max_tokens=800,
            )
        )
        history.append({"role": "user", "content": user_message})
        # The target assistant sees only its own system prompt and the visible
        # conversation. Rebuild the message objects here so authoring metadata can
        # never hitch a ride through a richer role-player or scenario structure.
        assistant_visible_history = [
            {"role": item["role"], "content": item["content"]}
            for item in history
        ]
        assistant_message = assistant_client.complete(
            system=assistant_system_prompt,
            messages=assistant_visible_history,
            temperature=0.25,
            max_tokens=2200,
        ).strip()
        if not assistant_message:
            raise ValueError("assistant model returned an empty response")
        history.append({"role": "assistant", "content": assistant_message})
    transcript = {
        "schema_version": 2,
        "scenario_id": scenario["id"],
        "scenario_fingerprint": stable_hash(scenario),
        "benchmark_axis": scenario["benchmark_axis"],
        "domain": deepcopy(scenario["domain"]),
        "evaluated_rule_id": scenario["evaluated_rule"]["chunk_id"],
        "source_seed_id": scenario["source_seed_id"],
        "texting_style_id": texting_style["id"],
        "purpose": purpose,
        "assistant_model": assistant_client.model,
        "user_model": user_client.model,
        "first_user_turn_generated": True,
        "turns": [
            {"turn": index, "role": item["role"], "content": item["content"]}
            for index, item in enumerate(history, 1)
        ],
    }
    validate_transcript(transcript)
    return transcript


def audit_transcript(
    *,
    client: OpenRouterClient,
    assignment: Mapping[str, Any],
    scenario: Mapping[str, Any],
    transcript: Mapping[str, Any],
    iteration: int,
    audit_history: Sequence[Mapping[str, Any]] | None = None,
    previous_audit: Mapping[str, Any] | None = None,
    terminal_verification: bool = False,
) -> dict[str, Any]:
    resolved_history = list(audit_history or ())
    if previous_audit is not None and not resolved_history:
        resolved_history = [previous_audit]
    def validate(value: dict[str, Any]) -> None:
        validate_audit(value, iteration=iteration)
        if value["decision"] == "revise_and_reroll":
            candidate = deepcopy(dict(scenario))
            candidate["user_role"] = value["revised_user_role"]
            candidate["scenario_plan"] = value["revised_scenario_plan"]
            _normalize_scenario(candidate, assignment)

    return client.complete_json(
        system=build_transcript_auditor_prompt(
            assignment,
            scenario,
            transcript,
            iteration,
            audit_history=resolved_history,
            terminal_verification=terminal_verification,
        ),
        messages=build_audit_conversation_messages(
            audit_history=resolved_history,
            terminal_verification=terminal_verification,
        ),
        temperature=0.15,
        max_tokens=5500,
        attempts=3,
        validator=validate,
    )


def _apply_audit_revision(
    scenario: Mapping[str, Any],
    audit: Mapping[str, Any],
    assignment: Mapping[str, Any],
) -> dict[str, Any]:
    revised = deepcopy(dict(scenario))
    revised["user_role"] = audit["revised_user_role"]
    revised["scenario_plan"] = audit["revised_scenario_plan"]
    _normalize_scenario(revised, assignment)
    return revised


def _load_valid_scenario(
    path: Path, assignment: Mapping[str, Any]
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = read_json(path)
        validate_scenario(value, assignment)
        return value
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _load_valid_transcript(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = read_json(path)
        validate_transcript(value)
        return value
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _load_valid_feedback(path: Path, iteration: int) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = read_json(path)
        validate_refiner_feedback(value, iteration=iteration)
        return value
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _load_valid_audit(
    path: Path,
    iteration: int,
    *,
    scenario: Mapping[str, Any],
    assignment: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = read_json(path)
        validate_audit(value, iteration=iteration)
        if value["decision"] == "revise_and_reroll":
            candidate = deepcopy(dict(scenario))
            candidate["user_role"] = value["revised_user_role"]
            candidate["scenario_plan"] = value["revised_scenario_plan"]
            _normalize_scenario(candidate, assignment)
        return value
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def run_assignment(
    *,
    assignment: Mapping[str, Any],
    sample_dir: Path,
    generator_client: OpenRouterClient,
    refiner_client: OpenRouterClient,
    auditor_client: OpenRouterClient,
    assistant_client: OpenRouterClient,
    user_client: OpenRouterClient,
    assistant_system_prompt: str,
    scenario_refinement_iterations: int = SCENARIO_REFINEMENT_ITERATIONS,
    audit_iterations: int = AUDIT_ITERATIONS,
    resume: bool = True,
    source_scenario: Mapping[str, Any] | None = None,
    source_refinement_history: Sequence[Mapping[str, Any]] | None = None,
    stop_after_refinement: bool = False,
    scenario_transform: Callable[
        [dict[str, Any], Mapping[str, Any]],
        tuple[dict[str, Any], dict[str, Any]],
    ] | None = None,
    skip_transcript_audit: bool = False,
) -> dict[str, Any]:
    if scenario_refinement_iterations != SCENARIO_REFINEMENT_ITERATIONS:
        raise ValueError("exactly five scenario-refinement iterations are required")
    if skip_transcript_audit:
        audit_iterations = 0
    elif not stop_after_refinement and audit_iterations != AUDIT_ITERATIONS:
        raise ValueError("exactly five transcript-audit iterations are required")

    sample_dir.mkdir(parents=True, exist_ok=True)
    write_json(sample_dir / "assignment.json", assignment)
    write_json(sample_dir / "source_seed.json", assignment["source_seed"])
    write_json(sample_dir / "evaluated_rule.json", assignment["evaluated_rule"])
    write_json(sample_dir / "texting_style.json", assignment["texting_style"])

    final_sample_path = sample_dir / "sample.json"
    if resume and final_sample_path.exists():
        value = read_json(final_sample_path)
        validate_scenario(value["scenario"], assignment)
        if value.get("refinement_complete") and stop_after_refinement:
            return value
        if value.get("refinement_complete"):
            scenario = value["scenario"]
            refinement_history = deepcopy(list(value["scenario_refinement_history"]))
        else:
            validate_transcript(value["transcript"])
            return value
    else:
        scenario = None
        refinement_history = None

    scenario_dir = sample_dir / "scenario"
    resumed_from_refinement = False
    if scenario is None and source_scenario is not None:
        if (
            source_refinement_history is None
            or len(source_refinement_history) != scenario_refinement_iterations
        ):
            raise ValueError(
                "transcript refresh requires the complete source refinement history"
            )
        scenario = deepcopy(dict(source_scenario))
        validate_scenario(scenario, assignment)
        refinement_history = deepcopy(list(source_refinement_history))
        write_json(scenario_dir / "imported-final.json", scenario)
        print(
            f"[{assignment['id']}] loaded finalized source scenario; "
            "generator/refiner skipped",
            flush=True,
        )
    elif scenario is None:
        scenario_path = scenario_dir / "generator-0.json"
        scenario = _load_valid_scenario(scenario_path, assignment) if resume else None
        if scenario is None:
            scenario = generate_scenario(generator_client, assignment)
            write_json(scenario_path, scenario)
        print(f"[{assignment['id']}] generated initial scenario", flush=True)

        refinement_history = []
        for iteration in range(1, scenario_refinement_iterations + 1):
            feedback_path = scenario_dir / f"refiner-feedback-{iteration}.json"
            feedback = (
                _load_valid_feedback(feedback_path, iteration) if resume else None
            )
            if feedback is None:
                feedback = review_scenario(
                    client=refiner_client,
                    assignment=assignment,
                    scenario=scenario,
                    iteration=iteration,
                    refinement_history=refinement_history,
                )
                write_json(feedback_path, feedback)

            revised_path = scenario_dir / f"generator-{iteration}.json"
            revised = _load_valid_scenario(revised_path, assignment) if resume else None
            scenario_before = deepcopy(scenario)
            if revised is None:
                revised = revise_scenario(
                    client=generator_client,
                    assignment=assignment,
                    scenario=scenario,
                    feedback=feedback,
                    iteration=iteration,
                    refinement_history=refinement_history,
                )
                write_json(revised_path, revised)
            record = refinement_history_record(
                iteration=iteration,
                scenario_before=scenario_before,
                scenario_after=revised,
                feedback=feedback,
            )
            record["input_scenario_fingerprint"] = stable_hash(scenario_before)
            record["output_scenario_fingerprint"] = stable_hash(revised)
            refinement_history.append(record)
            scenario = revised
            print(
                f"[{assignment['id']}] scenario refinement "
                f"{iteration}/{scenario_refinement_iterations} complete",
                flush=True,
            )
    else:
        resumed_from_refinement = True

    if stop_after_refinement:
        final_dir = sample_dir / "final"
        write_json(final_dir / "scenario.json", scenario)
        sample = {
            "schema_version": RUN_SCHEMA_VERSION,
            "id": assignment["id"],
            "benchmark_axis": assignment["benchmark_axis"],
            "domain": deepcopy(assignment["domain"]),
            "evaluated_rule": deepcopy(assignment["evaluated_rule"]),
            "source_seed_id": assignment["source_seed"]["training_seed_id"],
            "interaction_category": deepcopy(
                assignment.get("interaction_category")
            ),
            "validation_category": deepcopy(assignment.get("validation_category")),
            "texting_style": deepcopy(assignment["texting_style"]),
            "scenario": scenario,
            "scenario_refinement_iterations": scenario_refinement_iterations,
            "scenario_refinement_history": refinement_history,
            "refinement_complete": True,
            "generated_at": utc_now(),
            "models": {
                "generator": generator_client.model,
                "refiner": refiner_client.model,
            },
        }
        write_json(final_sample_path, sample)
        print(
            f"[{assignment['id']}] refinement complete; rollout and audit skipped",
            flush=True,
        )
        return sample

    if resumed_from_refinement:
        print(
            f"[{assignment['id']}] resuming from completed refinement into rollout",
            flush=True,
        )

    if scenario_transform is not None:
        opening_dir = sample_dir / "opening"
        opening_dir.mkdir(parents=True, exist_ok=True)
        scenario_before = deepcopy(scenario)
        scenario, transform_meta = scenario_transform(scenario, assignment)
        transform_meta["scenario_before_fingerprint"] = stable_hash(scenario_before)
        transform_meta["scenario_after_fingerprint"] = stable_hash(scenario)
        write_json(opening_dir / "paraphrase.json", transform_meta)
        validate_scenario(scenario, assignment)
        if transform_meta.get("skipped"):
            print(
                f"[{assignment['id']}] opening paraphrase skipped: "
                f"{transform_meta.get('reason', 'unknown')}",
                flush=True,
            )
        else:
            print(f"[{assignment['id']}] opening paraphrase applied", flush=True)

    transcript_dir = sample_dir / "transcript"
    original_path = transcript_dir / "rollout-0.json"
    transcript = _load_valid_transcript(original_path) if resume else None
    if transcript is None or transcript.get("scenario_fingerprint") != stable_hash(
        scenario
    ):
        transcript = run_conversation(
            assistant_client=assistant_client,
            user_client=user_client,
            scenario=scenario,
            texting_style=assignment["texting_style"],
            assistant_system_prompt=assistant_system_prompt,
            purpose="post_scenario_refinement",
        )
        write_json(original_path, transcript)
    original_transcript = deepcopy(transcript)

    audit_history: list[dict[str, Any]] = []
    if skip_transcript_audit:
        terminal_audit = {
            "iteration": 0,
            "decision": "skipped",
            "skipped": True,
            "reason": "skip_transcript_audit",
        }
    else:
        for iteration in range(1, audit_iterations + 1):
            audit_path = sample_dir / "audit" / f"iteration-{iteration}.json"
            audit = (
                _load_valid_audit(
                    audit_path,
                    iteration,
                    scenario=scenario,
                    assignment=assignment,
                )
                if resume
                else None
            )
            if audit is None:
                audit = audit_transcript(
                    client=auditor_client,
                    assignment=assignment,
                    scenario=scenario,
                    transcript=transcript,
                    iteration=iteration,
                    audit_history=audit_history,
                )
                write_json(audit_path, audit)

            input_fingerprint = stable_hash(transcript)
            if audit["decision"] == "revise_and_reroll":
                scenario = _apply_audit_revision(scenario, audit, assignment)
                write_json(
                    sample_dir / "audit" / f"scenario-{iteration}.json", scenario
                )
                reroll_path = transcript_dir / f"rollout-{iteration}.json"
                transcript = _load_valid_transcript(reroll_path) if resume else None
                if transcript is None or transcript.get(
                    "scenario_fingerprint"
                ) != stable_hash(scenario):
                    transcript = run_conversation(
                        assistant_client=assistant_client,
                        user_client=user_client,
                        scenario=scenario,
                        texting_style=assignment["texting_style"],
                        assistant_system_prompt=assistant_system_prompt,
                        purpose=f"post_audit_reroll_{iteration}",
                    )
                    write_json(reroll_path, transcript)
            audit_history.append(deepcopy(audit))
            audit_history[-1]["input_transcript_fingerprint"] = input_fingerprint
            audit_history[-1]["output_transcript_fingerprint"] = stable_hash(transcript)
            print(
                f"[{assignment['id']}] transcript audit "
                f"{iteration}/{audit_iterations}: {audit['decision']}",
                flush=True,
            )

        terminal_iteration = audit_iterations + 1
        terminal_path = sample_dir / "audit" / "terminal-verification.json"
        terminal_audit = (
            _load_valid_audit(
                terminal_path,
                terminal_iteration,
                scenario=scenario,
                assignment=assignment,
            )
            if resume
            else None
        )
        if terminal_audit is None:
            terminal_audit = audit_transcript(
                client=auditor_client,
                assignment=assignment,
                scenario=scenario,
                transcript=transcript,
                iteration=terminal_iteration,
                audit_history=audit_history,
                terminal_verification=True,
            )
            write_json(terminal_path, terminal_audit)
        print(
            f"[{assignment['id']}] terminal transcript verification: "
            f"{terminal_audit['decision']}",
            flush=True,
        )

    final_dir = sample_dir / "final"
    write_json(final_dir / "scenario.json", scenario)
    write_json(final_dir / "transcript.json", transcript)
    sample = {
        "schema_version": RUN_SCHEMA_VERSION,
        "id": assignment["id"],
        "benchmark_axis": assignment["benchmark_axis"],
        "domain": deepcopy(assignment["domain"]),
        "evaluated_rule": deepcopy(assignment["evaluated_rule"]),
        "source_seed_id": assignment["source_seed"]["training_seed_id"],
        "interaction_category": deepcopy(
            assignment.get("interaction_category")
        ),
        "validation_category": deepcopy(assignment.get("validation_category")),
        "texting_style": deepcopy(assignment["texting_style"]),
        "scenario": scenario,
        "original_transcript": original_transcript,
        "transcript": transcript,
        "scenario_refinement_iterations": scenario_refinement_iterations,
        "scenario_refinement_history": refinement_history,
        "audit_iterations": audit_iterations,
        "audit_history": audit_history,
        "terminal_audit": deepcopy(terminal_audit),
        "terminal_audit_transcript_fingerprint": stable_hash(transcript),
        "filters_applied": False,
        "generated_at": utc_now(),
        "models": {
            "generator": generator_client.model,
            "refiner": refiner_client.model,
            "auditor": auditor_client.model,
            "assistant": assistant_client.model,
            "user": user_client.model,
        },
    }
    write_json(final_sample_path, sample)
    return sample


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _load_scenario_source(
    source_root: Path,
    plan: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    manifest = read_json(source_root / "manifest.json")
    if (
        manifest.get("status") != "complete"
        or manifest.get("completed_samples") < len(plan)
        or manifest.get("scenario_refinement_iterations")
        != SCENARIO_REFINEMENT_ITERATIONS
    ):
        raise ValueError("scenario source run is not a complete compatible run")
    rows = [
        json.loads(line)
        for line in (source_root / "samples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    by_id = {str(item["id"]): item for item in rows}
    expected_ids = {str(item["id"]) for item in plan}
    if not expected_ids <= set(by_id):
        raise ValueError("scenario source run does not contain all plan sample ids")
    for assignment in plan:
        sample = by_id[str(assignment["id"])]
        validate_scenario(sample["scenario"], assignment)
        if (
            sample.get("scenario_refinement_iterations")
            != SCENARIO_REFINEMENT_ITERATIONS
            or len(sample.get("scenario_refinement_history", []))
            != SCENARIO_REFINEMENT_ITERATIONS
        ):
            raise ValueError(
                f"{assignment['id']}: source scenario lacks complete refinements"
            )
    return by_id, manifest


def run_pilot(
    *,
    api_key: str,
    run_id: str,
    sample_count: int = 20,
    benchmark_axis: str = "both",
    workers: int = 20,
    scenario_refinement_iterations: int = SCENARIO_REFINEMENT_ITERATIONS,
    audit_iterations: int = AUDIT_ITERATIONS,
    planning_seed: int = 20260807,
    timeout: float = 300.0,
    retries: int = 3,
    resume: bool = True,
    generator_model: str = DEFAULT_GENERATOR_MODEL,
    refiner_model: str = DEFAULT_REFINER_MODEL,
    auditor_model: str = DEFAULT_AUDITOR_MODEL,
    assistant_model: str = DEFAULT_ASSISTANT_MODEL,
    user_model: str = DEFAULT_USER_MODEL,
    assistant_prompt_path: Path = ASSISTANT_PROMPT_PATH,
    scenario_source_run: Path | None = None,
    plan: Sequence[Mapping[str, Any]] | None = None,
    plan_validator: Callable[[list[dict[str, Any]]], None] | None = None,
    seed_corpus_path: Path = SEED_CORPUS_PATH,
    stop_after_refinement: bool = False,
    scenario_transform: Callable[
        [dict[str, Any], Mapping[str, Any]],
        tuple[dict[str, Any], dict[str, Any]],
    ] | None = None,
    skip_transcript_audit: bool = False,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    if scenario_refinement_iterations != SCENARIO_REFINEMENT_ITERATIONS:
        raise ValueError("exactly five scenario-refinement iterations are required")
    if skip_transcript_audit:
        audit_iterations = 0
    elif not stop_after_refinement and audit_iterations != AUDIT_ITERATIONS:
        raise ValueError("exactly five transcript-audit iterations are required")
    active_prompt_version = prompt_version_for_run(benchmark_axis)

    if plan is not None:
        plan = [dict(item) for item in plan]
        sample_count = len(plan)
        if plan_validator is not None:
            plan_validator(plan)
    elif benchmark_axis == "sycophancy":
        plan = build_sycophancy_plan(
            sample_count=sample_count,
            planning_seed=planning_seed,
        )
        validate_sycophancy_plan(plan)
    elif benchmark_axis == "calibrated_validation":
        plan = build_validation_plan(
            sample_count=sample_count,
            planning_seed=planning_seed,
        )
        validate_validation_plan(plan)
    elif benchmark_axis == "both":
        plan = build_plan(sample_count=sample_count, planning_seed=planning_seed)
        validate_plan(plan)
    else:
        raise ValueError(
            "benchmark_axis must be 'both', 'sycophancy', or "
            "'calibrated_validation'"
        )
    source_samples: dict[str, dict[str, Any]] = {}
    source_manifest: dict[str, Any] | None = None
    if scenario_source_run is not None:
        scenario_source_run = scenario_source_run.resolve()
        source_samples, source_manifest = _load_scenario_source(
            scenario_source_run, plan
        )
    root = run_directory(run_id)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and resume:
        existing_manifest = read_json(manifest_path)
        existing_prompt_version = existing_manifest.get("prompt_version")
        if existing_prompt_version != active_prompt_version:
            raise ValueError(
                "existing run uses a different authoring prompt version; "
                "choose a new run_id or pass --no-resume"
            )
    plan_path = root / "plan.json"
    if plan_path.exists() and resume:
        existing_plan = read_json(plan_path)
        if stable_hash(existing_plan) != stable_hash(plan):
            raise ValueError("existing run plan does not match current fixed inputs")
    else:
        write_json(plan_path, plan)

    assistant_system_prompt = assistant_prompt_path.read_text(encoding="utf-8")
    (root / "assistant_system_prompt.txt").write_text(
        assistant_system_prompt, encoding="utf-8"
    )
    print_lock = threading.Lock()

    def client(model: str) -> OpenRouterClient:
        return build_client(
            model,
            api_key=api_key,
            timeout=timeout,
            retries=retries,
            reasoning_effort=None,
        )

    def execute(assignment: Mapping[str, Any]) -> dict[str, Any]:
        with print_lock:
            print(
                f"PILOT_ITEM_START {assignment['id']} "
                f"({assignment['benchmark_axis']}; {assignment['domain']['name']}; "
                f"{assignment['evaluated_rule']['chunk_id']})",
                flush=True,
            )
        source_sample = source_samples.get(str(assignment["id"]))
        authoring_override = assignment.get("authoring_models") or {}
        result = run_assignment(
            assignment=assignment,
            sample_dir=root / "samples" / assignment["id"],
            generator_client=client(authoring_override.get("generator", generator_model)),
            refiner_client=client(authoring_override.get("refiner", refiner_model)),
            auditor_client=client(authoring_override.get("auditor", auditor_model)),
            assistant_client=client(assistant_model),
            user_client=client(user_model),
            assistant_system_prompt=assistant_system_prompt,
            scenario_refinement_iterations=scenario_refinement_iterations,
            audit_iterations=audit_iterations,
            resume=resume,
            source_scenario=(
                source_sample["scenario"] if source_sample is not None else None
            ),
            source_refinement_history=(
                source_sample["scenario_refinement_history"]
                if source_sample is not None
                else None
            ),
            stop_after_refinement=stop_after_refinement,
            scenario_transform=scenario_transform,
            skip_transcript_audit=skip_transcript_audit,
        )
        with print_lock:
            print(f"PILOT_ITEM_COMPLETE {assignment['id']}", flush=True)
        return result

    completed_by_id: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=min(workers, len(plan))) as executor:
        futures = {
            executor.submit(execute, assignment): assignment for assignment in plan
        }
        for future in as_completed(futures):
            assignment = futures[future]
            try:
                sample = future.result()
                completed_by_id[assignment["id"]] = sample
            except Exception as exc:
                failure = {
                    "id": assignment["id"],
                    "benchmark_axis": assignment["benchmark_axis"],
                    "domain": assignment["domain"]["name"],
                    "evaluated_rule_id": assignment["evaluated_rule"]["chunk_id"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
                failures.append(failure)
                with print_lock:
                    print(
                        f"PILOT_ITEM_ERROR {assignment['id']}: {failure['error']}",
                        flush=True,
                    )

    completed = [
        completed_by_id[item["id"]]
        for item in plan
        if item["id"] in completed_by_id
    ]
    _write_jsonl(root / "samples.jsonl", completed)
    manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "release_label": "v6-prompt-pilot",
        "status": "complete" if len(completed) == sample_count else "partial",
        "requested_samples": sample_count,
        "completed_samples": len(completed),
        "failed_samples": len(failures),
        "benchmark_axis_counts": dict(
            Counter(item["benchmark_axis"] for item in completed)
        ),
        "domain_counts": dict(Counter(item["domain"]["name"] for item in completed)),
        "rule_counts": dict(
            Counter(item["evaluated_rule"]["chunk_id"] for item in completed)
        ),
        "interaction_category_counts": dict(
            Counter(
                item["interaction_category"]["name"]
                for item in completed
                if item.get("interaction_category")
            )
        ),
        "validation_category_counts": dict(
            Counter(
                item["validation_category"]["name"]
                for item in completed
                if item.get("validation_category")
            )
        ),
        "unique_seed_count": len({item["source_seed_id"] for item in completed}),
        "unique_texting_style_count": len(
            {item["texting_style"]["id"] for item in completed}
        ),
        "scenario_refinement_iterations": scenario_refinement_iterations,
        "audit_iterations": 0 if stop_after_refinement else audit_iterations,
        "terminal_audit_iterations": (
            0 if stop_after_refinement else TERMINAL_AUDIT_ITERATIONS
        ),
        "terminal_audit_decision_counts": (
            {}
            if stop_after_refinement
            else dict(Counter(item["terminal_audit"]["decision"] for item in completed))
        ),
        "refinement_only": stop_after_refinement,
        "turns_per_transcript": 10,
        "filters_applied": False,
        "manual_audits": False,
        "first_user_turn_generated": True,
        "planning_seed": planning_seed,
        "prompt_version": active_prompt_version,
        "transcript_refresh_only": scenario_source_run is not None,
        "skip_transcript_audit": skip_transcript_audit,
        "scenario_source_run": (
            str(scenario_source_run) if scenario_source_run is not None else None
        ),
        "scenario_source_prompt_version": (
            source_manifest.get("prompt_version")
            if source_manifest is not None
            else None
        ),
        "stages_executed": (
            (
                ["rollout"]
                if skip_transcript_audit and scenario_source_run is not None
                else [
                    "generator",
                    *[
                        f"scenario_refiner_{iteration}"
                        for iteration in range(1, scenario_refinement_iterations + 1)
                    ],
                ]
            )
            if stop_after_refinement
            else (
                ["rollout"]
                if skip_transcript_audit
                else [
                    "rollout",
                    *[
                        f"transcript_audit_{iteration}"
                        for iteration in range(1, audit_iterations + 1)
                    ],
                    "terminal_transcript_verification",
                ]
            )
            if scenario_source_run is not None
            else (
                [
                    "generator",
                    *[
                        f"scenario_refiner_{iteration}"
                        for iteration in range(1, scenario_refinement_iterations + 1)
                    ],
                    "rollout",
                ]
                if skip_transcript_audit
                else [
                    "generator",
                    *[
                        f"scenario_refiner_{iteration}"
                        for iteration in range(1, scenario_refinement_iterations + 1)
                    ],
                    "rollout",
                    *[
                        f"transcript_audit_{iteration}"
                        for iteration in range(1, audit_iterations + 1)
                    ],
                    "terminal_transcript_verification",
                ]
            )
        ),
        "models": {
            "generator": generator_model,
            "refiner": refiner_model,
            "auditor": auditor_model,
            "assistant": assistant_model,
            "user": user_model,
        },
        "per_sample_generator_counts": dict(
            Counter(
                (item.get("authoring_models") or {}).get("generator", generator_model)
                for item in plan
            )
        ),
        "source_files": {
            "training_seeds": str(seed_corpus_path),
            "training_seeds_sha256": file_hash(seed_corpus_path),
            "appendix_rules": str(APPENDIX_RULES_PATH),
            "appendix_rules_sha256": file_hash(APPENDIX_RULES_PATH),
            "texting_styles": str(TEXTING_STYLES_PATH),
            "texting_styles_sha256": file_hash(TEXTING_STYLES_PATH),
        },
        "standard_rules_only": True,
        "single_atomic_rule_target_per_sample": True,
        "composition_chunks_sampled": 0,
        "publishable": False,
        "website_import": False,
        "failures": failures,
        "completed_at": utc_now(),
    }
    write_json(manifest_path, manifest)
    return manifest


__all__ = [
    "AUDIT_ITERATIONS",
    "DEFAULT_ASSISTANT_MODEL",
    "DEFAULT_AUDITOR_MODEL",
    "DEFAULT_GENERATOR_MODEL",
    "DEFAULT_REFINER_MODEL",
    "DEFAULT_USER_MODEL",
    "SCENARIO_REFINEMENT_ITERATIONS",
    "TERMINAL_AUDIT_ITERATIONS",
    "audit_transcript",
    "generate_scenario",
    "review_scenario",
    "revise_scenario",
    "run_assignment",
    "run_conversation",
    "run_pilot",
    "prompt_version_for_run",
]
