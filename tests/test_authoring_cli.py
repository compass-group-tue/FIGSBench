from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from figsbench.generation import __main__ as cli
from figsbench.generation.planning import EXAMPLE_SEED_CORPUS, SeedPool
from figsbench.generation.specs import SPECS_PATH

REPO = Path(__file__).resolve().parents[1]
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


def test_cli_defaults_are_the_canonical_models(tmp_path: Path) -> None:
    args = cli.parse_args(["--seed-corpus", str(tmp_path / "s.jsonl"), "--dry-run"])
    assert (args.generator_model, args.refiner_model, args.auditor_model) == (
        "google/gemini-3.8-flash",) * 3
    assert args.assistant_model == "openai/gpt-5.6-luna"
    assert args.user_model == "local-deepseek-v4-flash"
    assert args.planning_seed == 20260915


def test_dry_run_plans_every_slot_with_the_example_corpus(capsys) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert cli.main(["--seed-corpus", str(EXAMPLE_SEED_CORPUS), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert '"n": 390' in out and '"n": 110' in out


def test_missing_corpus_is_a_clear_error(tmp_path: Path, capsys) -> None:
    assert cli.main(["--seed-corpus", str(tmp_path / "missing.jsonl"), "--dry-run"]) == 1
    assert "not redistributed" in capsys.readouterr().err


def test_example_corpus_covers_every_seed_domain() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pool = SeedPool(EXAMPLE_SEED_CORPUS)
    assert {d["id"] for d in pool.domains} == EXPECTED_SEED_DOMAIN_IDS


def test_make_specs_reproduces_the_shipped_slots(tmp_path: Path) -> None:
    out = tmp_path / "specs.json"
    subprocess.run([sys.executable, "scripts/make_specs.py", "--seed", "20260915",
                    "--output", str(out)], cwd=REPO, check=True, capture_output=True)
    assert out.read_bytes() == SPECS_PATH.read_bytes()


def test_make_specs_fresh_seed_is_valid_and_different(tmp_path: Path) -> None:
    out = tmp_path / "specs.json"
    subprocess.run([sys.executable, "scripts/make_specs.py", "--seed", "42",
                    "--output", str(out)], cwd=REPO, check=True, capture_output=True)
    specs = json.loads(out.read_text())["specs"]
    locked = json.loads(SPECS_PATH.read_text())["specs"]
    assert len(specs) == 500 and specs != locked
    assert sum(s["axis"] == "sycophancy" for s in specs) == 390
