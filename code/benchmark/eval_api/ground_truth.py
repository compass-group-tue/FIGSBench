"""Ground-truth lock for the final benchmark.

The canonical scenario set is data/benchmark_500.jsonl at the release root
(500 scenarios: 390 sycophancy + 110 CV). Every evaluation through benchmark.eval_api must run on
these scenarios and nothing else. Anything else -- X4 pre-gate scenarios,
unknown ids, or scenario text that does not match the
locked file -- raises immediately instead of silently producing off-benchmark
numbers.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GROUND_TRUTH_PATH = PROJECT_ROOT.parent / "data" / "benchmark_500.jsonl"

# Per-run fields that legitimately differ from the locked scenario record:
# the rollout conversation and the models that produced it. Everything else
# (scenario plan, user role, texting style, rule, domain, audits, ...) must
# match the locked file exactly.
_RUN_ONLY_KEYS = frozenset({"transcript", "models"})


def _strip_x4(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("-X4", "")
    if isinstance(value, dict):
        return {k: _strip_x4(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_strip_x4(v) for v in value]
    return value


def _scenario_fingerprint(record: dict[str, Any]) -> str:
    scenario = {
        k: _strip_x4(v) for k, v in record.items() if k not in _RUN_ONLY_KEYS
    }
    return json.dumps(scenario, sort_keys=True)


@lru_cache(maxsize=1)
def load_ground_truth() -> dict[str, dict[str, Any]]:
    if not GROUND_TRUTH_PATH.is_file():
        raise FileNotFoundError(
            f"ground-truth scenario file missing: {GROUND_TRUTH_PATH}"
        )
    records: dict[str, dict[str, Any]] = {}
    with GROUND_TRUTH_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[record["id"]] = record
    if len(records) != 500:
        raise ValueError(
            f"ground-truth file has {len(records)} scenarios, expected 500: "
            f"{GROUND_TRUTH_PATH}"
        )
    return records


def assert_final_samples(samples: list[dict[str, Any]]) -> None:
    """Raise unless every sample is a locked final-500 scenario."""
    ground_truth = load_ground_truth()
    stale = [s.get("id") for s in samples if "-X4" in str(s.get("id", ""))]
    if stale:
        raise ValueError(
            f"{len(stale)} sample(s) carry stale pre-gate -X4 ids "
            f"(e.g. {stale[:5]}). Re-roll them on {GROUND_TRUTH_PATH.name}."
        )
    unknown = [s.get("id") for s in samples if s.get("id") not in ground_truth]
    if unknown:
        raise ValueError(
            f"{len(unknown)} sample id(s) are not in the locked final 500 "
            f"(e.g. {unknown[:5]}). Only {GROUND_TRUTH_PATH} may be evaluated."
        )
    mismatched = [
        s.get("id")
        for s in samples
        if _scenario_fingerprint(s)
        != _scenario_fingerprint(ground_truth[s["id"]])
    ]
    if mismatched:
        raise ValueError(
            f"{len(mismatched)} sample(s) match final-500 ids but their "
            f"scenario content differs from the locked file "
            f"(e.g. {mismatched[:5]}). The scenarios were edited or came "
            f"from the pre-gate X4 set; refusing to evaluate."
        )
