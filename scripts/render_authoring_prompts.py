"""Write the exact authoring prompts one benchmark scenario was built with.

Uses the scenario's recorded assignment (data/provenance/generation_runs/) and its stored
scenario, first refinement review and reference transcript. No model is called.

    python scripts/render_authoring_prompts.py FINAL-SYC-001
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from figsbench.generation import prompts
from figsbench.generation.pipeline import AXES

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sample_id")
    ap.add_argument("--out", type=Path, default=REPO / "examples" / "rendered_prompts" / "authoring")
    args = ap.parse_args()

    sample = next((json.loads(l) for l in open(DATA / "provenance/benchmark_500_unexpanded.jsonl")
                   if json.loads(l)["id"] == args.sample_id), None)
    if sample is None:
        raise SystemExit(f"unknown sample id {args.sample_id}")
    axis = AXES[sample["benchmark_axis"]]
    plan = json.load(open(DATA / "provenance/generation_runs" / f"attempt-{sample['gate_winner_attempt']}"
                          / axis.name / "plan.json"))
    assignment = next(a for a in plan if a["id"] == args.sample_id)
    shown = deepcopy(assignment)
    shown["source_seed"].pop("text_sha256", None)  # provenance only; never part of a prompt

    scenario = sample["scenario"]
    feedback = sample["scenario_refinement_history"][0]
    transcript = sample["transcript"]
    history = [t for t in transcript["turns"] if t["turn"] < 3]
    rendered = {
        "1_scenario_generator.txt": axis.generator_prompt(shown),
        "2_scenario_refiner.txt": axis.refiner_prompt(shown, scenario, 1, None),
        "3_scenario_revision.txt": axis.revision_prompt(shown, scenario, feedback, 1, None),
        "4_user_roleplayer_turn3.txt": axis.roleplayer_prompt(
            scenario, sample["texting_style"], 3, history),
        "5_transcript_auditor.txt": axis.auditor_prompt(shown, scenario, transcript, 1, [], False),
    }
    out = args.out / args.sample_id
    out.mkdir(parents=True, exist_ok=True)
    for name, text in rendered.items():
        (out / name).write_text(text)
    (out / "assignment.json").write_text(json.dumps(assignment, indent=2, ensure_ascii=False))
    print(f"wrote {len(rendered)} prompts for {args.sample_id} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
