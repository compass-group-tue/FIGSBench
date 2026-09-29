"""Launch the locked 500-slot final benchmark generation.

Reads archetype_guided_benchmark/data/final_500_specs.json (seed 20260915):
- sycophancy slots: per-slot generator, randomly alternating inkling vs
  google/gemini-3.8-flash (196 vs 194, balanced per rule).
- calibrated_validation slots: google/gemini-3.8-flash only.

Each axis runs as its own run_pilot (syc uses the v5 species-lock templates,
CV uses v28) because the two prompt patches target the same pipeline globals.
Per-sample generator routing is via assignment["authoring_models"].

Examples:
  python -m archetype_guided_benchmark.run_final --dry-run
  python -m archetype_guided_benchmark.run_final --axis syc --limit 4 --workers 4
  python -m archetype_guided_benchmark.run_final --axis both --workers 10 --run-judge
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark.pipeline.client import load_dotenv

from archetype_guided_benchmark.bootstrap import (
    apply_archetype_prompt_patch,
    apply_archetype_schema_extensions,
    configure_pipeline_iterations,
)
from archetype_guided_benchmark.bootstrap_validation_v28 import (
    apply_validation_v28_archetype_prompt_patch,
)
from archetype_guided_benchmark import prompts as archetype_prompts
from archetype_guided_benchmark.prompts import configure_species_lock_variant
from archetype_guided_benchmark.prompts_validation_v28 import (
    VALIDATION_PROMPT_VERSION,
)
from archetype_guided_benchmark.sources import build_final_sycophancy_plan
from archetype_guided_benchmark.sources_validation_v28 import (
    build_final_validation_plan,
)
from archetype_guided_benchmark.final_specs import FINAL_PLANNING_SEED
from rule_guided_benchmark import io_utils as rg_io
from rule_guided_benchmark.io_utils import ASSISTANT_PROMPT_PATH, PROJECT_ROOT
from rule_guided_benchmark.io_utils import SEED_CORPUS_PATH
from archetype_guided_benchmark.run_cv_v28_archetype import (
    COLD_ASSISTANT_PROMPT_PATH,
)
from archetype_guided_benchmark.judge_v9 import DEFAULT_JUDGE_MODEL
from rule_guided_judging.hr_v11_judge import (
    API_JUDGE_MODEL,
    DEFAULT_JUDGE_RUN_SUFFIX,
    run_locked_judge,
)
from rule_guided_benchmark.pipeline import run_pilot

SPECS_PATH = Path(__file__).resolve().parent / "data" / "final_500_specs.json"

LUNA = "openai/gpt-5.6-luna"
GEMINI = "google/gemini-3.8-flash"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", choices=("syc", "cv", "both"), default="both")
    parser.add_argument("--run-id-syc", default="final-500-syc-v5-species-lock")
    parser.add_argument("--run-id-cv", default="final-500-cv-v28")
    parser.add_argument("--specs-file", type=Path, default=SPECS_PATH)
    parser.add_argument(
        "--seed-corpus",
        type=Path,
        default=SEED_CORPUS_PATH,
        help=(
            "JSONL inspiration corpus. Supply a corpus you are licensed to use; "
            "the historical corpus is not redistributed. Start with "
            "source_corpus/data/example-seeds-v1/seeds.jsonl (see its README "
            "for the schema)."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="directory for authoring run folders (default: package historical path)",
    )
    parser.add_argument("--planning-seed", type=int, default=FINAL_PLANNING_SEED)
    parser.add_argument("--scenario-refinement-iterations", type=int, default=5)
    parser.add_argument("--audit-iterations", type=int, default=5)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0,
                        help="Smoke test: only first N slots per axis (0 = all).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build plans and print margins without calling any API.")
    parser.add_argument("--refiner-model", default=None,
                        help="Run-wide refiner override (default: follow slot generator).")
    parser.add_argument("--auditor-model", default=None,
                        help="Run-wide auditor override (default: follow slot generator).")
    parser.add_argument("--generator-model", default=None,
                        help="Run-wide generator override (default: follow slot generator).")
    parser.add_argument(
        "--assistant-model", default=LUNA,
        help="Assistant rollout model for the authoring-time reference rollouts.",
    )
    parser.add_argument("--user-model", default="local-deepseek-v4-flash")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--run-judge", action="store_true",
                        help="After generation, run locked hr-v11 judges.")
    parser.add_argument("--api", action="store_true",
                        help="Judge via OpenRouter API instead of local deployment.")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-workers", type=int, default=10)
    return parser.parse_args(argv)


def apply_slot_model_overrides(
    plan: list[dict[str, Any]],
    *,
    generator: str | None,
    refiner: str | None,
    auditor: str | None,
) -> None:
    for assignment in plan:
        authoring = assignment.setdefault("authoring_models", {})
        if generator:
            authoring["generator"] = generator
        if refiner:
            authoring["refiner"] = refiner
        if auditor:
            authoring["auditor"] = auditor


def plan_margins(plan: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(plan),
        "generators": dict(Counter(
            (a.get("authoring_models") or {}).get("generator") for a in plan)),
        "rules": dict(Counter(a["evaluated_rule"]["chunk_id"] for a in plan)),
        "severity": dict(Counter(a.get("severity", "-") for a in plan)),
    }


def run_axis(
    *,
    api_key: str,
    axis: str,
    plan: list[dict[str, Any]],
    run_id: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    run_dir = rg_io.run_directory(run_id)
    manifest = run_pilot(
        api_key=api_key,
        run_id=run_id,
        benchmark_axis=("sycophancy" if axis == "syc" else "calibrated_validation"),
        workers=min(args.workers, len(plan)),
        scenario_refinement_iterations=args.scenario_refinement_iterations,
        audit_iterations=args.audit_iterations,
        planning_seed=args.planning_seed,
        timeout=args.timeout,
        retries=args.retries,
        resume=not args.no_resume,
        generator_model=GEMINI,
        refiner_model=args.refiner_model or GEMINI,
        auditor_model=args.auditor_model or GEMINI,
        assistant_model=args.assistant_model,
        user_model=args.user_model,
        assistant_prompt_path=(
            ASSISTANT_PROMPT_PATH if axis == "syc" else COLD_ASSISTANT_PROMPT_PATH
        ),
        plan=plan,
        plan_validator=None,
        seed_corpus_path=args.seed_corpus,
    )
    manifest["final_specs_file"] = str(args.specs_file)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    summary: dict[str, Any] = {
        "run_id": run_id,
        "generation_status": manifest.get("status"),
        "completed_samples": manifest.get("completed_samples"),
        "failed_samples": manifest.get("failed_samples"),
        "per_sample_generator_counts": manifest.get("per_sample_generator_counts"),
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
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output_root is not None:
        rg_io.DATA_ROOT = args.output_root.resolve()
    specs_data = json.loads(args.specs_file.read_text())
    specs = specs_data["specs"]
    if args.limit:
        syc = [s for s in specs if s["axis"] == "sycophancy"][: args.limit]
        cv = [s for s in specs if s["axis"] == "calibrated_validation"][: args.limit]
    else:
        syc = [s for s in specs if s["axis"] == "sycophancy"]
        cv = [s for s in specs if s["axis"] == "calibrated_validation"]

    # NOTE: the syc and CV prompt patches rebind the same
    # rule_guided_benchmark.pipeline globals, so the CV patch must never be
    # active while syc runs. Build and run syc first, then patch and run CV.
    plans: dict[str, list[dict[str, Any]]] = {}
    try:
        if args.axis in ("syc", "both"):
            configure_species_lock_variant()
            apply_archetype_prompt_patch()
            apply_archetype_schema_extensions()
            plans["syc"] = build_final_sycophancy_plan(
                syc,
                planning_seed=args.planning_seed,
                seed_corpus_path=args.seed_corpus,
            )
            apply_slot_model_overrides(
                plans["syc"], generator=args.generator_model, refiner=args.refiner_model, auditor=args.auditor_model
            )
        if args.axis == "cv" or args.dry_run:
            # Dry runs build both plans for display only (no runs follow, so the
            # CV patch cannot contaminate anything). Real both-axis runs build the
            # CV plan after the syc run completes (see below).
            if args.axis in ("cv", "both"):
                apply_validation_v28_archetype_prompt_patch()
                plans["cv"] = build_final_validation_plan(
                    cv,
                    planning_seed=args.planning_seed,
                    seed_corpus_path=args.seed_corpus,
                )
                apply_slot_model_overrides(
                    plans["cv"], generator=args.generator_model, refiner=args.refiner_model, auditor=args.auditor_model
                )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for axis, plan in plans.items():
        print(f"PLAN {axis}: " + json.dumps(plan_margins(plan), indent=1), flush=True)

    if args.dry_run:
        versions = {"syc_prompt": archetype_prompts.PROMPT_VERSION,
                    "cv_prompt": VALIDATION_PROMPT_VERSION}
        print("PROMPTS " + json.dumps(versions), flush=True)
        return 0

    load_dotenv(args.env_file)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY is required", file=sys.stderr)
        return 1
    configure_pipeline_iterations(
        args.scenario_refinement_iterations,
        args.audit_iterations,
    )

    summaries: dict[str, Any] = {}
    if "syc" in plans:
        configure_species_lock_variant()
        apply_archetype_prompt_patch()
        apply_archetype_schema_extensions()
        summaries["syc"] = run_axis(
            api_key=api_key, axis="syc", plan=plans["syc"],
            run_id=args.run_id_syc, args=args)
    if args.axis in ("cv", "both"):
        apply_validation_v28_archetype_prompt_patch()
        if "cv" not in plans:
            plans["cv"] = build_final_validation_plan(
                cv,
                planning_seed=args.planning_seed,
                seed_corpus_path=args.seed_corpus,
            )
            apply_slot_model_overrides(
                plans["cv"], generator=args.generator_model, refiner=args.refiner_model, auditor=args.auditor_model
            )
            print("PLAN cv: " + json.dumps(plan_margins(plans["cv"]), indent=1), flush=True)
        summaries["cv"] = run_axis(
            api_key=api_key, axis="cv", plan=plans["cv"],
            run_id=args.run_id_cv, args=args)
    print(json.dumps(summaries, indent=2))
    ok = all(s.get("generation_status") == "complete" for s in summaries.values())
    if args.run_judge:
        ok = ok and all(s.get("judge_status") == "complete" for s in summaries.values())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
