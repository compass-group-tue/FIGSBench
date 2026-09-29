"""Render the exact system prompts each generation stage sent for one final sample.

Added for the public release (not part of the original pipeline). It applies the
same prompt patches as `archetype_guided_benchmark.run_final`, rebuilds the
sample's plan assignment from `final_500_specs.json`, and calls the patched
pipeline prompt builders with the sample's own stored scenario, first
refinement feedback, and transcript. No API calls are made.

    cd code
    python scripts/render_generation_prompts.py FINAL-SYC-001 --out ../prompts/generation_rendered
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
RELEASE = CODE.parent
GEMINI = "google/gemini-3.8-flash"
# attempt a / b were planned with these seeds (see data/provenance/generation_runs/*/manifest.json)
ATTEMPT_SEEDS = {"a": 20260915, "b": 20260916}


def load_sample(sample_id: str) -> dict:
    for line in open(RELEASE / "data/provenance/benchmark_500_unexpanded.jsonl"):
        row = json.loads(line)
        if row["id"] == sample_id:
            return row
    raise SystemExit(f"unknown sample id {sample_id}")


def build_assignment(sample: dict, axis: str) -> dict:
    specs = json.load(open(CODE / "archetype_guided_benchmark/data/final_500_specs.json"))["specs"]
    seed = ATTEMPT_SEEDS[sample["gate_winner_attempt"]]
    if axis == "sycophancy":
        from archetype_guided_benchmark.bootstrap import (
            apply_archetype_prompt_patch, apply_archetype_schema_extensions)
        from archetype_guided_benchmark.prompts import configure_species_lock_variant
        from archetype_guided_benchmark.run_final import apply_slot_model_overrides
        from archetype_guided_benchmark.sources import build_final_sycophancy_plan
        configure_species_lock_variant()
        apply_archetype_prompt_patch()
        apply_archetype_schema_extensions()
        plan = build_final_sycophancy_plan(
            [s for s in specs if s["axis"] == "sycophancy"], planning_seed=seed)
        apply_slot_model_overrides(plan, generator=GEMINI, refiner=GEMINI, auditor=GEMINI)
    else:
        from archetype_guided_benchmark.bootstrap_validation_v28 import (
            apply_validation_v28_archetype_prompt_patch)
        from archetype_guided_benchmark.sources_validation_v28 import build_final_validation_plan
        apply_validation_v28_archetype_prompt_patch()
        plan = build_final_validation_plan(
            [s for s in specs if s["axis"] == "calibrated_validation"], planning_seed=seed)
    slot = sample["id"].replace("FINAL-", "")
    return next(a for a in plan if a["final_slot_id"] == slot)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sample_id")
    ap.add_argument("--out", type=Path, default=RELEASE / "prompts/generation_rendered")
    args = ap.parse_args()

    sample = load_sample(args.sample_id)
    axis = sample["benchmark_axis"]
    assignment = build_assignment(sample, axis)

    import rule_guided_benchmark.pipeline as rg_pipeline

    scenario = sample["scenario"]
    feedback = sample["scenario_refinement_history"][0]
    transcript = sample["transcript"]
    history = [t for t in transcript["turns"] if t["turn"] < 3]
    prompts = {
        "1_scenario_generator.txt": rg_pipeline.build_generator_prompt(assignment),
        "2_scenario_refiner.txt": rg_pipeline.build_refiner_prompt(assignment, scenario, 1, None),
        "3_scenario_revision.txt": rg_pipeline.build_scenario_revision_prompt(
            assignment, scenario, feedback, 1, None),
        "4_user_roleplayer_turn3.txt": rg_pipeline.build_roleplayer_prompt(
            scenario, sample["texting_style"], 3, history),
        "5_transcript_auditor.txt": rg_pipeline.build_transcript_auditor_prompt(
            assignment, scenario, transcript, 1, audit_history=[], terminal_verification=False),
    }
    out = args.out / args.sample_id
    out.mkdir(parents=True, exist_ok=True)
    for name, text in prompts.items():
        (out / name).write_text(text)
    (out / "assignment.json").write_text(json.dumps(assignment, indent=2, ensure_ascii=False))
    print(f"wrote {len(prompts)} prompts for {args.sample_id} ({axis}) -> {out}")


if __name__ == "__main__":
    main()
