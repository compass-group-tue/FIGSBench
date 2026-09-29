"""Persistence helpers and paths for resumable SFT-data runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from source_corpus.io_utils import RunLock, rewrite_jsonl, utc_now, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "sft_data"
DATA_ROOT = PACKAGE_ROOT / "data" / "runs"
DEFAULT_SEED_CORPUS = (
    PROJECT_ROOT
    / "source_corpus"
    / "data"
    / "training_seeds"
    / "broad-training-24k-v2-20260802"
    / "training_seeds.jsonl"
)
def run_dir(run_id: str) -> Path:
    if not run_id or any(
        char
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for char in run_id
    ):
        raise ValueError("run_id may contain only letters, digits, '-' and '_'")
    return DATA_ROOT / run_id


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield value


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def corpus_digest(records: Iterable[dict[str, Any]]) -> str:
    compact = [
        {
            "id": item["id"],
            "attempt": item["attempt"],
            "messages": item["transcript"]["messages"],
        }
        for item in sorted(records, key=lambda row: row["id"])
    ]
    return sha256_json(compact)


__all__ = [
    "DATA_ROOT",
    "DEFAULT_SEED_CORPUS",
    "PROJECT_ROOT",
    "RunLock",
    "corpus_digest",
    "iter_jsonl",
    "read_json",
    "rewrite_jsonl",
    "run_dir",
    "sha256_json",
    "sha256_file",
    "sha256_text",
    "utc_now",
    "write_json",
]
