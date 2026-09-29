"""data/judge_ratings_1000.jsonl: 1,000 conversations rated by the paper judge."""
from __future__ import annotations

import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULES = {
    "sycophancy": {"S1.a", "S1.b", "S1.c", "S2.a", "S2.b", "S2.c", "S2.d"},
    "calibrated_validation": {"V1", "V2", "V3"},
}


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in open(ROOT / "data" / name, encoding="utf-8")]


def test_every_scenario_twice_with_two_models() -> None:
    rows = load("judge_ratings_1000.jsonl")
    canon = {r["id"]: r for r in load("benchmark_500.jsonl")}
    assert len(rows) == 1000 and len({r["example_id"] for r in rows}) == 1000
    models = collections.defaultdict(set)
    for r in rows:
        assert r["benchmark_axis"] == canon[r["sample_id"]]["benchmark_axis"]
        models[r["sample_id"]].add(r["model_under_test"])
    assert set(models) == set(canon)
    assert all(len(m) == 2 for m in models.values())


def test_judgments_are_well_formed() -> None:
    for r in load("judge_ratings_1000.jsonl"):
        assert len(r["conversation"]) == 10
        for axis, j in r["judgment"].items():
            assert j["score"] in (1, 2, 3, 4)
            assert set(j["broken_rules"]) <= RULES[axis]
            assert (j["score"] == 1) == (not j["broken_rules"])
