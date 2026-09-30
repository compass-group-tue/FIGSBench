"""Ground-truth lock for the final benchmark.

The canonical scenario set is data/benchmark_500.jsonl at the release root
(500 scenarios: 390 sycophancy + 110 CV). Every evaluation must run on
these scenarios and nothing else. Unknown ids or scenario text that does not match the locked
file raise immediately instead of silently producing off-benchmark numbers.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
# The canonical file ships in the repository (data/) and on Hugging Face; set
# FIGS_BENCHMARK_FILE to use a copy elsewhere.
GROUND_TRUTH_PATH = Path(
    os.environ.get("FIGS_BENCHMARK_FILE") or REPO_ROOT / "data" / "benchmark_500.jsonl"
)

# Per-run fields that legitimately differ from the locked scenario record: the rollout
# conversation and the models that produced it. Everything else must match exactly.
_RUN_ONLY_KEYS = frozenset({"transcript", "models"})


def _scenario_fingerprint(record: dict[str, Any]) -> str:
    return json.dumps({k: v for k, v in record.items() if k not in _RUN_ONLY_KEYS},
                      sort_keys=True)


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
            f"(e.g. {mismatched[:5]}). The scenarios were edited; refusing to evaluate."
        )
