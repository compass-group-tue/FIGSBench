"""Generate a fresh deterministic slot-spec grid for new benchmark samples.

Wraps archetype_guided_benchmark.final_specs.build_final_specs: the same
rule x archetype x domain x severity construction as the locked
final_500_specs.json, re-rolled with a different planning seed. The output
feeds run_final directly via --specs-file.

Examples:
  python scripts/make_specs.py --seed 42 --output /tmp/my_specs.json
  python -m archetype_guided_benchmark.run_final --axis syc --limit 2 \\
      --seed-corpus source_corpus/data/example-seeds-v1/seeds.jsonl \\
      --specs-file /tmp/my_specs.json --run-id-syc my-syc --workers 2

With --seed 20260915 the output reproduces data/final_500_specs.json exactly.
New grids are NOT the canonical benchmark: evaluate against
data/benchmark_500.jsonl for comparable scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from archetype_guided_benchmark.final_specs import (  # noqa: E402
    FINAL_PLANNING_SEED,
    build_final_specs,
    spec_margins,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=FINAL_PLANNING_SEED,
        help="planning seed; 20260915 reproduces the locked 500-slot file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="destination JSON file ({planning_seed, specs})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    specs = build_final_specs(seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        # No trailing newline: matches final_specs.py so --seed 20260915
        # byte-reproduces data/final_500_specs.json.
        json.dumps({"planning_seed": args.seed, "specs": specs}, indent=1),
        encoding="utf-8",
    )
    print(json.dumps(spec_margins(specs), indent=1))
    print(f"wrote {args.output} ({len(specs)} slots, seed {args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
