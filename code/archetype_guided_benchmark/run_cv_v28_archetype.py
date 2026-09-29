"""Run v28 per-archetype diverse CV pilots (Kimi authoring)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from benchmark.pipeline.client import load_dotenv

from archetype_guided_benchmark.bootstrap_validation_v28 import (
    apply_validation_v28_archetype_prompt_patch,
    configure_pipeline_iterations,
)
from archetype_guided_benchmark.judge_v9 import DEFAULT_JUDGE_MODEL
from rule_guided_judging.hr_v11_judge import (
    API_JUDGE_MODEL,
    DEFAULT_JUDGE_RUN_SUFFIX,
    run_locked_judge,
)
from archetype_guided_benchmark.prompts_validation_v28 import VALIDATION_PROMPT_VERSION
from archetype_guided_benchmark.sources_validation_v28 import (
    VALID_ARCHETYPE_IDS,
    archetype_planning_seed,
    build_single_archetype_plan,
    validate_single_archetype_plan,
)
from rule_guided_benchmark import io_utils as rg_io
from rule_guided_benchmark.io_utils import ASSISTANT_PROMPT_PATH, PROJECT_ROOT

COLD_ASSISTANT_PROMPT_PATH = (
    PROJECT_ROOT
    / "validation_pressure_experiment"
    / "config"
    / "cold_assistant_system_prompt.txt"
)
from rule_guided_benchmark.pipeline import run_pilot

KIMI = "moonshotai/kimi-k2.6"
LUNA = "openai/gpt-5.6-luna"
DEEPSEEK_USER = "deepseek/deepseek-v4-pro"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archetype",
        required=True,
        choices=VALID_ARCHETYPE_IDS,
        help="Pin entire run to one conversation archetype A1–A6.",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--scenario-refinement-iterations", type=int, default=5)
    parser.add_argument("--audit-iterations", type=int, default=5)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--planning-seed", type=int, default=20260822)
    parser.add_argument("--generator-model", default=KIMI)
    parser.add_argument("--refiner-model", default=KIMI)
    parser.add_argument("--auditor-model", default=KIMI)
    parser.add_argument("--assistant-model", default=LUNA)
    parser.add_argument("--user-model", default="local-deepseek-v4-flash")
    parser.add_argument(
        "--assistant-system-prompt-file",
        type=Path,
        default=COLD_ASSISTANT_PROMPT_PATH,
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--run-judge", action="store_true",
                        help="After generation, run locked hr-v11 judges (syc-score-v1 + syc-rules-v1, cv-score-v1 + cv-rules-v1).")
    parser.add_argument(
        "--api",
        action="store_true",
        help="Judge via OpenRouter API (z-ai/glm-5.3-flash) instead of the local deployment.",
    )
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-workers", type=int, default=10)
    return parser.parse_args(argv)


def default_run_id(archetype_id: str, planning_seed: int) -> str:
    return f"archetype-cv-v28-{archetype_id.lower()}-kimi-{planning_seed}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv(args.env_file)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY is required", file=sys.stderr)
        return 1

    apply_validation_v28_archetype_prompt_patch()
    configure_pipeline_iterations(
        args.scenario_refinement_iterations,
        args.audit_iterations,
    )

    effective_seed = archetype_planning_seed(args.planning_seed, args.archetype)
    plan = build_single_archetype_plan(
        args.archetype,
        sample_count=args.samples,
        planning_seed=effective_seed,
    )

    def validate_plan(items: list[dict[str, Any]]) -> None:
        validate_single_archetype_plan(items, archetype_id=args.archetype)

    validate_plan(plan)

    run_id = args.run_id or default_run_id(args.archetype, effective_seed)
    manifest = run_pilot(
        api_key=api_key,
        run_id=run_id,
        benchmark_axis="calibrated_validation",
        workers=min(args.workers, len(plan)),
        scenario_refinement_iterations=args.scenario_refinement_iterations,
        audit_iterations=args.audit_iterations,
        planning_seed=effective_seed,
        timeout=args.timeout,
        retries=args.retries,
        resume=not args.no_resume,
        generator_model=args.generator_model,
        refiner_model=args.refiner_model,
        auditor_model=args.auditor_model,
        assistant_model=args.assistant_model,
        user_model=args.user_model,
        assistant_prompt_path=args.assistant_system_prompt_file,
        plan=plan,
        plan_validator=validate_plan,
    )
    manifest["archetype_prompt_version"] = VALIDATION_PROMPT_VERSION
    manifest["pinned_archetype"] = args.archetype
    manifest["base_planning_seed"] = args.planning_seed
    manifest["authoring_models"] = {
        "generator": args.generator_model,
        "refiner": args.refiner_model,
        "auditor": args.auditor_model,
    }
    run_dir = rg_io.run_directory(run_id)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary: dict[str, Any] = {
        "run_id": run_id,
        "archetype": args.archetype,
        "planning_seed": effective_seed,
        "prompt_version": VALIDATION_PROMPT_VERSION,
        "generation_status": manifest.get("status"),
        "completed_samples": manifest.get("completed_samples"),
        "failed_samples": manifest.get("failed_samples"),
    }

    if args.run_judge:
        judge_run_id = f"{run_id}-{DEFAULT_JUDGE_RUN_SUFFIX}"
        judge_manifest = run_locked_judge(
            api_key=api_key,
            source_run=run_dir,
            judge_run_id=judge_run_id,
            judge_model=(API_JUDGE_MODEL if args.api else args.judge_model),
            workers=args.judge_workers,
        )
        summary["judge_run_id"] = judge_run_id
        summary["judge_status"] = judge_manifest.get("status")
        summary["judge_counts"] = judge_manifest.get("counts")

    print(json.dumps(summary, indent=2))
    if summary.get("generation_status") != "complete":
        return 1
    if args.run_judge and summary.get("judge_status") != "complete":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
