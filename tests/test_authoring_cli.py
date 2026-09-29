from __future__ import annotations

import json
import random
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from archetype_guided_benchmark import run_final
from sft_data.seeds import SeedPool

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
EXAMPLE_CORPUS = (
    CODE_DIR / "source_corpus" / "data" / "example-seeds-v1" / "seeds.jsonl"
)
LOCKED_SPECS = CODE_DIR / "archetype_guided_benchmark" / "data" / "final_500_specs.json"
EXPECTED_SEED_DOMAIN_IDS = frozenset(
    {
        "health_medicine",
        "work_careers_organizations",
        "finance_law",
        "science_technology",
        "intimate_relationships",
        "family_friends_social_life",
        "beliefs_culture_society",
    }
)


def test_authoring_cli_accepts_public_paths(tmp_path: Path) -> None:
    args = run_final.parse_args(
        [
            "--seed-corpus",
            str(tmp_path / "seeds.jsonl"),
            "--output-root",
            str(tmp_path / "runs"),
            "--dry-run",
        ]
    )
    assert args.seed_corpus == tmp_path / "seeds.jsonl"
    assert args.output_root == tmp_path / "runs"


def test_make_specs_reproduces_locked_file(tmp_path: Path) -> None:
    out = tmp_path / "specs.json"
    subprocess.run(
        [sys.executable, "scripts/make_specs.py", "--seed", "20260915",
         "--output", str(out)],
        cwd=CODE_DIR,
        check=True,
        capture_output=True,
    )
    assert out.read_bytes() == LOCKED_SPECS.read_bytes()


def test_make_specs_fresh_seed_is_valid_and_novel(tmp_path: Path) -> None:
    out = tmp_path / "specs.json"
    subprocess.run(
        [sys.executable, "scripts/make_specs.py", "--seed", "7",
         "--output", str(out)],
        cwd=CODE_DIR,
        check=True,
        capture_output=True,
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["planning_seed"] == 7
    assert len(data["specs"]) == 500
    # Extra slots sample random cells, so fresh grids vary slightly around the
    # locked 390/110 split; they must stay usable for both axes.
    syc_count = sum(1 for s in data["specs"] if s["axis"] == "sycophancy")
    cv_count = sum(1 for s in data["specs"] if s["axis"] == "calibrated_validation")
    assert syc_count + cv_count == 500
    assert 370 <= syc_count <= 410
    assert 90 <= cv_count <= 130
    assert data["specs"] != json.loads(LOCKED_SPECS.read_text())["specs"]


def test_example_seed_corpus_loads_with_warning_and_reuses_safely() -> None:
    with pytest.warns(UserWarning, match="non-canonical"):
        pool = SeedPool(EXAMPLE_CORPUS)
    assert {d["id"] for d in pool.domains} == EXPECTED_SEED_DOMAIN_IDS
    used = set(pool.by_id)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        seed = pool.choose_unique(
            "Health & Medicine", random.Random(0), used
        )
    assert seed.training_seed_id in used
    assert [w for w in records if "reusing seeds" in str(w.message)]
