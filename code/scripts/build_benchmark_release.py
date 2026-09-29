#!/usr/bin/env python3
"""Build the deterministic harder-attempt benchmark release without API calls.

Example (separate sycophancy and CV authoring runs)::

    python code/scripts/build_benchmark_release.py \
      --attempt-a-run runs/round-a-syc --attempt-a-run runs/round-a-cv \
      --attempt-b-run runs/round-b-syc --attempt-b-run runs/round-b-cv \
      --attempt-a-judge-run judges/round-a \
      --attempt-b-judge-run judges/round-b \
      --gate-model deepseek/deepseek-v4.1-flash --output-dir build/release
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from benchmark.release_builder import ReleaseBuildError, build_release  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attempt-a-run",
        action="append",
        type=Path,
        required=True,
        help="attempt-a run directory (repeat for separate syc/CV runs)",
    )
    parser.add_argument(
        "--attempt-b-run",
        action="append",
        type=Path,
        required=True,
        help="attempt-b run directory (repeat for separate syc/CV runs)",
    )
    parser.add_argument(
        "--attempt-a-judge-run",
        action="append",
        type=Path,
        required=True,
        help="attempt-a judge run or results directory (repeatable)",
    )
    parser.add_argument(
        "--attempt-b-judge-run",
        action="append",
        type=Path,
        required=True,
        help="attempt-b judge run or results directory (repeatable)",
    )
    parser.add_argument("--gate-model", required=True, help="model used for gate rollouts")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tie-winner",
        choices=("a", "b"),
        default="a",
        help="deterministic default for equal scores (default: a)",
    )
    parser.add_argument(
        "--stored-picks",
        type=Path,
        help="existing complete gate_picks.json; preserves historical tie choices",
    )
    parser.add_argument(
        "--expanded-jsonl",
        type=Path,
        help="already-expanded one-to-one JSONL to validate and publish",
    )
    parser.add_argument("--benchmark-name", help="manifest label (default: final-N)")
    parser.add_argument("--force", action="store_true", help="replace existing outputs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        artifacts = build_release(
            attempt_a_runs=args.attempt_a_run,
            attempt_b_runs=args.attempt_b_run,
            attempt_a_judge_runs=args.attempt_a_judge_run,
            attempt_b_judge_runs=args.attempt_b_judge_run,
            output_dir=args.output_dir,
            gate_model=args.gate_model,
            tie_winner=args.tie_winner,
            stored_picks_path=args.stored_picks,
            expanded_jsonl=args.expanded_jsonl,
            benchmark_name=args.benchmark_name,
            force=args.force,
        )
    except (ReleaseBuildError, FileNotFoundError, FileExistsError, json.JSONDecodeError) as exc:
        print(f"release build failed: {exc}", file=sys.stderr)
        return 2
    print(f"gate picks: {artifacts.gate_picks}")
    print(f"gate manifest: {artifacts.gate_manifest}")
    print(f"unexpanded benchmark: {artifacts.unexpanded_jsonl}")
    if artifacts.benchmark_jsonl is not None:
        print(f"expanded benchmark: {artifacts.benchmark_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
