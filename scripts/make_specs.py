"""Write a slot-spec grid for new scenarios: python scripts/make_specs.py --seed 42 --output specs.json

Same rule x archetype x domain x severity construction as the 500 benchmark slots.
--seed 20260915 reproduces src/figsbench/data/final_500_specs.json exactly. Pass the output to
python -m figsbench.generation --specs-file specs.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from figsbench.generation.specs import FINAL_PLANNING_SEED, build_final_specs, spec_margins


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=FINAL_PLANNING_SEED)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    specs = build_final_specs(seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # no trailing newline, so --seed 20260915 is byte-identical to the shipped file
    args.output.write_text(json.dumps({"planning_seed": args.seed, "specs": specs}, indent=1))
    print(json.dumps(spec_margins(specs), indent=1))
    print(f"wrote {len(specs)} slot specs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
