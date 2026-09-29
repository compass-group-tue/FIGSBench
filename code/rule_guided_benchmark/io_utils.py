"""Paths, hashing, and atomic JSON persistence for rule-guided runs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "rule_guided_benchmark"
DATA_ROOT = PACKAGE_ROOT / "data" / "runs"
SEED_CORPUS_PATH = (
    PROJECT_ROOT
    / "source_corpus"
    / "data"
    / "training_seeds"
    / "broad-training-24k-v2-20260802"
    / "training_seeds.jsonl"
)
CHUNK_CATALOG_PATH = PROJECT_ROOT / "model_spec_chunks" / "chunks.jsonl"
APPENDIX_RULES_PATH = PACKAGE_ROOT / "config" / "appendix_rules_v2.json"
TEXTING_STYLES_PATH = (
    PROJECT_ROOT
    / "sft_data"
    / "data"
    / "runs"
    / "chunks-150-sanitized-v25"
    / "texting_styles.json"
)
ASSISTANT_PROMPT_PATH = (
    PROJECT_ROOT / "benchmark" / "config" / "assistant_system_prompt.txt"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_directory(run_id: str) -> Path:
    if not run_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in run_id
    ):
        raise ValueError("run_id may contain only letters, digits, '-' and '_'")
    return DATA_ROOT / run_id
