"""Author FIGS scenarios: python -m figsbench.generation --seed-corpus SEEDS.jsonl [options]

Builds one authoring assignment per slot in the specs file (default: the 500 benchmark
slots), then runs the authoring pipeline for each axis. Every stage's output is kept under
--output-dir/<axis>/, and interrupted runs resume. The benchmark itself was made this way,
twice per slot, and the harder attempt of each pair was kept (see the README).

Examples:
  python -m figsbench.generation --seed-corpus src/figsbench/data/example_seeds/seeds.jsonl --dry-run
  python -m figsbench.generation --seed-corpus seeds.jsonl --axis sycophancy --limit 2 \\
      --output-dir runs/my-authoring
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from figsbench.client import load_dotenv
from figsbench.generation.pipeline import run_authoring
from figsbench.generation.planning import build_sycophancy_plan, build_validation_plan
from figsbench.generation.specs import FINAL_PLANNING_SEED, SPECS_PATH

GEMINI = "google/gemini-3.8-flash"
LUNA = "openai/gpt-5.6-luna"
AXES = {"sycophancy": build_sycophancy_plan, "calibrated_validation": build_validation_plan}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed-corpus", type=Path, required=True,
                        help="JSONL inspiration corpus (the paper's corpus is not "
                             "redistributed; see src/figsbench/data/example_seeds/)")
    parser.add_argument("--axis", choices=("sycophancy", "calibrated_validation", "both"),
                        default="both")
    parser.add_argument("--specs-file", type=Path, default=SPECS_PATH,
                        help="slot specs (make new ones with scripts/make_specs.py)")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/authoring"))
    parser.add_argument("--planning-seed", type=int, default=FINAL_PLANNING_SEED,
                        help="the paper used 20260915 and 20260916 for its two attempts")
    parser.add_argument("--generator-model", default=GEMINI)
    parser.add_argument("--refiner-model", default=GEMINI)
    parser.add_argument("--auditor-model", default=GEMINI)
    parser.add_argument("--assistant-model", default=LUNA,
                        help="reference assistant for the authoring rollouts")
    parser.add_argument("--user-model", default="local-deepseek-v4-flash")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="only the first N slots per axis")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="build the plans and print their margins; no model calls")
    return parser.parse_args(argv)


def plan_margins(plan: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(plan),
        "generators": dict(Counter(
            (a.get("authoring_models") or {}).get("generator") for a in plan)),
        "rules": dict(Counter(a["evaluated_rule"]["chunk_id"] for a in plan)),
        "severity": dict(Counter(a.get("severity", "-") for a in plan)),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    specs = json.loads(args.specs_file.read_text(encoding="utf-8"))["specs"]
    axes = list(AXES) if args.axis == "both" else [args.axis]
    plans: dict[str, list[dict[str, Any]]] = {}
    for axis in axes:
        slots = [s for s in specs if s["axis"] == axis]
        if args.limit:
            slots = slots[: args.limit]
        try:
            plan = AXES[axis](slots, planning_seed=args.planning_seed,
                              seed_corpus_path=args.seed_corpus)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        for assignment in plan:
            assignment["authoring_models"] = {"generator": args.generator_model,
                                              "refiner": args.refiner_model,
                                              "auditor": args.auditor_model}
        plans[axis] = plan
        print(f"PLAN {axis}: " + json.dumps(plan_margins(plan), indent=1), flush=True)
    if args.dry_run:
        return 0

    load_dotenv(Path.cwd() / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY is required", file=sys.stderr)
        return 1
    summaries = {}
    for axis, plan in plans.items():
        run_dir = args.output_dir / axis
        manifest = run_authoring(
            api_key=api_key, axis_name=axis, plan=plan, run_dir=run_dir,
            planning_seed=args.planning_seed, seed_corpus_path=args.seed_corpus,
            generator_model=args.generator_model, refiner_model=args.refiner_model,
            auditor_model=args.auditor_model, assistant_model=args.assistant_model,
            user_model=args.user_model, workers=args.workers, timeout=args.timeout,
            retries=args.retries, resume=not args.no_resume,
        )
        manifest["final_specs_file"] = str(args.specs_file)
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        summaries[axis] = {k: manifest.get(k) for k in
                           ("status", "completed_samples", "failed_samples")}
    print(json.dumps(summaries, indent=2))
    return 0 if all(s["status"] == "complete" for s in summaries.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
