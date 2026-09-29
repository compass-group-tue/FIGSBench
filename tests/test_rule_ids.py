"""Sycophancy rules are S1.a-S2.d (renamed from R1.a-R2.d for the public release)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from archetype_guided_benchmark.final_specs import _legacy_seed_label  # noqa: E402
from rule_guided_benchmark import prompts as rg_prompts  # noqa: E402

SYC_RULES = ["S1.a", "S1.b", "S1.c", "S2.a", "S2.b", "S2.c", "S2.d"]
OLD_ID = re.compile(r"(?<![A-Za-z0-9])R[12]\.[a-d](?![A-Za-z0-9])")


def test_rules_config_uses_s_ids() -> None:
    text = (ROOT / "code/rule_guided_benchmark/config/appendix_rules_v2.json").read_text()
    ids = re.findall(r'"id": "([SV][0-9](?:\.[a-d])?)"', text)
    assert [i for i in ids if i.startswith("S")] == SYC_RULES
    assert not OLD_ID.search(text)


def test_canonical_samples_use_s_ids() -> None:
    rows = [json.loads(line) for line in open(ROOT / "data/benchmark_500.jsonl")]
    syc = [r["evaluated_rule"]["rule_id"] for r in rows if r["benchmark_axis"] == "sycophancy"]
    assert len(syc) == 390 and set(syc) == set(SYC_RULES)


def test_authoring_selects_the_s_rule() -> None:
    context = rg_prompts._selected_rule_context(
        {"benchmark_axis": "sycophancy", "evaluated_rule": {"rule_id": "S2.b"}}
    )
    assert "S2.b" in json.dumps(context)


def test_planning_seeds_keep_pre_rename_labels() -> None:
    assert _legacy_seed_label("S1.a") == "R1.a"
    assert _legacy_seed_label("S2.d") == "R2.d"
    assert _legacy_seed_label("V2") == "V2"
