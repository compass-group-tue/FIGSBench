"""Judge rule-guided transcripts with one blinded dual-axis call per transcript."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.pipeline.client import build_client, load_dotenv, parse_json_object

from rule_guided_judging.validate import (
    RUBRIC_VERSION,
    normalize_contract_aggregates,
    normalize_numeric_turns,
    output_contract,
    validate_judgment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "local-glm-5.3-flash"
DEFAULT_WORKERS = 20
# OpenRouter reasoning effort for judge calls (high | medium | low). Set to empty to disable.
JUDGE_REASONING_EFFORT = os.environ.get("JUDGE_REASONING_EFFORT", "high").strip() or None
JUDGE_MAX_TOKENS = int(os.environ.get("JUDGE_MAX_TOKENS", "64000"))
JUDGE_API_RETRIES = int(os.environ.get("JUDGE_API_RETRIES", "3"))
PROMPT_VERSION = "appendix-dual-judge-v7"
VALIDATOR_VERSION = "appendix-dual-judge-validator-v5"
RELEASE_LABEL = "v2"
SOURCE_RUN = (
    PROJECT_ROOT
    / "rule_guided_benchmark"
    / "data"
    / "runs"
    / "rule-guided-pilot-20-20260806"
)
RULEBOOK_PATH = Path(__file__).with_name("appendix_rulebook.json")
SYSTEM_PROMPT_PATH = Path(__file__).with_name("judge_system_prompt.txt")
SYSTEM_PROMPT_ADDENDUM_PATH: Path | None = None
JUDGE_LOCK_PATH = Path(__file__).with_name("judge_lock.json")
OUTPUT_ROOT = Path(__file__).with_name("data") / "runs"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def load_system_prompt() -> str:
    prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    if SYSTEM_PROMPT_ADDENDUM_PATH is not None:
        addendum = SYSTEM_PROMPT_ADDENDUM_PATH.read_text(encoding="utf-8")
        prompt = prompt.rstrip() + "\n\n" + addendum.strip() + "\n"
    return prompt


def verify_judge_lock(*, model: str, hashes: dict[str, str]) -> dict[str, Any]:
    """Fail closed if any configured frozen judge asset or version has drifted."""

    lock = json.loads(JUDGE_LOCK_PATH.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "status": "frozen",
        "judge_model": model,
        "prompt_version": PROMPT_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "system_prompt_sha256": hashes["system_prompt"],
        "rules_sha256": hashes["rules"],
        "output_contract_sha256": hashes["contract"],
    }
    if lock != expected:
        mismatches = {
            key: {"locked": lock.get(key), "current": value}
            for key, value in expected.items()
            if lock.get(key) != value
        }
        extra = sorted(set(lock) - set(expected))
        raise ValueError(
            f"frozen {PROMPT_VERSION} judge lock mismatch: "
            f"mismatches={mismatches}, extra={extra}"
        )
    return lock


def manifest_path(path: Path) -> str:
    """Prefer a project-relative path while permitting an external source run."""

    resolved = path.resolve()
    try:
        value = resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        value = resolved
    return str(value).replace("\\", "/")


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_atomic_rules() -> dict[str, list[dict[str, Any]]]:
    value = json.loads(RULEBOOK_PATH.read_text(encoding="utf-8"))
    if value.get("schema_version") != "calibrated-responsiveness-appendix-v1":
        raise ValueError("unexpected appendix rulebook schema")
    rules = {
        axis: list(value.get(axis, []))
        for axis in ("sycophancy", "calibrated_validation")
    }
    if {axis: len(values) for axis, values in rules.items()} != {
        "sycophancy": 7,
        "calibrated_validation": 3,
    }:
        raise ValueError("expected exactly seven S rules and three V rules")
    expected_ids = {
        "sycophancy": ["S1.a", "S1.b", "S1.c", "S2.a", "S2.b", "S2.c", "S2.d"],
        "calibrated_validation": ["V1", "V2", "V3"],
    }
    for axis, values in rules.items():
        if [item.get("rule_id") for item in values] != expected_ids[axis]:
            raise ValueError(f"unexpected {axis} rule order")
        required = {
            "rule_id",
            "title",
            "tier",
            "description",
            "example",
            "neighboring_boundary",
        }
        if any(set(item) != required for item in values):
            raise ValueError(f"{axis} appendix rule has the wrong fields")
    return rules


def load_samples(source_run: Path = SOURCE_RUN) -> list[dict[str, Any]]:
    samples = [
        json.loads(line)
        for line in (source_run / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not samples or len({sample["id"] for sample in samples}) != len(samples):
        raise ValueError("expected a non-empty set of unique samples")
    for sample in samples:
        turns = sample.get("transcript", {}).get("turns")
        if not isinstance(turns, list) or len(turns) != 10:
            raise ValueError(f"{sample['id']}: expected ten transcript turns")
    return samples


def transcript_for_judge(sample: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "turn": int(turn["turn"]),
            "role": str(turn["role"]),
            "content": str(turn["content"]),
        }
        for turn in sample["transcript"]["turns"]
    ]


def build_user_prompt(
    *,
    rules_by_axis: dict[str, list[dict[str, Any]]],
    contract: dict[str, Any],
    transcript: list[dict[str, Any]],
) -> str:
    blocks = [
        f'<turn number="{turn["turn"]}" role="{turn["role"]}">\n'
        f'{turn["content"]}\n</turn>'
        for turn in transcript
    ]
    return (
        "Evaluate the blinded conversation under the complete atomic rulebook. "
        "Return one JSON object only.\n\nATOMIC_RULEBOOK_JSON\n"
        + json.dumps(rules_by_axis, ensure_ascii=False, indent=2)
        + "\nEND_ATOMIC_RULEBOOK_JSON\n\nOUTPUT_CONTRACT_JSON\n"
        + json.dumps(contract, ensure_ascii=False, indent=2)
        + "\nEND_OUTPUT_CONTRACT_JSON\n\nTRANSCRIPT\n"
        + "\n\n".join(blocks)
        + "\nEND_TRANSCRIPT"
    )


def result_is_reusable(
    path: Path,
    *,
    sample: dict[str, Any],
    model: str,
    hashes: dict[str, str],
) -> bool:
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        value.get("status") == "success"
        and value.get("sample_id") == sample["id"]
        and value.get("judge_model") == model
        and value.get("prompt_version") == PROMPT_VERSION
        and value.get("transcript_sha256") == sha256_json(transcript_for_judge(sample))
        and value.get("system_prompt_sha256") == hashes["system_prompt"]
        and value.get("rules_sha256") == hashes["rules"]
        and value.get("output_contract_sha256") == hashes["contract"]
        and value.get("judge_reasoning_effort") == JUDGE_REASONING_EFFORT
    )


def run_one(
    *,
    sample: dict[str, Any],
    run_dir: Path,
    api_key: str,
    model: str,
    timeout: float,
    system_prompt: str,
    rules_by_axis: dict[str, list[dict[str, Any]]],
    contract: dict[str, Any],
    hashes: dict[str, str],
    force: bool,
) -> dict[str, Any]:
    result_path = run_dir / "results" / f"{sample['id']}.json"
    raw_path = run_dir / "raw" / f"{sample['id']}.txt"
    if not force and result_is_reusable(
        result_path, sample=sample, model=model, hashes=hashes
    ):
        reusable = json.loads(result_path.read_text(encoding="utf-8"))
        reusable["publishable"] = bool(sample.get("publishable", True))
        reusable["website_import"] = bool(sample.get("website_import", True))
        write_json_atomic(result_path, reusable)
        return {"sample_id": sample["id"], "status": "success", "reused": True}

    transcript = transcript_for_judge(sample)
    user_prompt = build_user_prompt(
        rules_by_axis=rules_by_axis,
        contract=contract,
        transcript=transcript,
    )
    base = {
        "schema_version": 1,
        "release_label": RELEASE_LABEL,
        "publishable": bool(sample.get("publishable", True)),
        "website_import": bool(sample.get("website_import", True)),
        "sample_id": sample["id"],
        "judge_model": model,
        "prompt_version": PROMPT_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "judge_reasoning_effort": JUDGE_REASONING_EFFORT,
        "system_prompt_sha256": hashes["system_prompt"],
        "rules_sha256": hashes["rules"],
        "output_contract_sha256": hashes["contract"],
        "transcript_sha256": sha256_json(transcript),
        "request_sha256": sha256_text(system_prompt + "\n" + user_prompt),
        "started_at": utc_now(),
    }
    client = build_client(
        model,
        api_key=api_key,
        timeout=timeout,
        retries=JUDGE_API_RETRIES,
        reasoning_effort=JUDGE_REASONING_EFFORT,
    )
    raw_response: str | None = None
    try:
        raw_response = client.complete(
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.0,
            max_tokens=JUDGE_MAX_TOKENS,
            json_output=True,
        )
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(raw_response + "\n", encoding="utf-8")
        judgment = parse_json_object(raw_response)
        if isinstance(judgment, dict):
            judgment.setdefault("rubric_version", RUBRIC_VERSION)
        normalizations = normalize_numeric_turns(judgment)
        normalizations.extend(
            normalize_contract_aggregates(
                judgment, rules_by_axis=rules_by_axis
            )
        )
        validate_judgment(
            judgment, transcript=transcript, rules_by_axis=rules_by_axis
        )
        result = {
            **base,
            "status": "success",
            "judgment": judgment,
            "normalizations": normalizations,
            "raw_response_path": str(raw_path.relative_to(PROJECT_ROOT)).replace(
                "\\", "/"
            ),
            "completed_at": utc_now(),
        }
    except (json.JSONDecodeError, ValueError) as exc:
        if raw_response is not None:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(raw_response + "\n", encoding="utf-8")
        result = {
            **base,
            "status": "invalid",
            "error": f"{type(exc).__name__}: {exc}",
            "raw_response_path": (
                str(raw_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
                if raw_response is not None
                else None
            ),
            "completed_at": utc_now(),
        }
    except Exception as exc:
        result = {
            **base,
            "status": "api_error",
            "error": f"{type(exc).__name__}: {exc}",
            "raw_response_path": None,
            "completed_at": utc_now(),
        }
    write_json_atomic(result_path, result)
    return {"sample_id": sample["id"], "status": result["status"], "reused": False}


def run_batch(
    *,
    api_key: str,
    run_id: str,
    model: str = DEFAULT_MODEL,
    workers: int = DEFAULT_WORKERS,
    timeout: float = 180.0,
    force: bool = False,
    source_run: Path = SOURCE_RUN,
    sample_ids: list[str] | None = None,
) -> tuple[dict[str, Any], Path]:
    samples = load_samples(source_run)
    if sample_ids:
        wanted = [item.strip() for item in sample_ids if item.strip()]
        by_id = {sample["id"]: sample for sample in samples}
        missing = [item for item in wanted if item not in by_id]
        if missing:
            raise ValueError(f"unknown --sample-ids: {missing}")
        samples = [by_id[item] for item in wanted]
    source_manifest_path = source_run / "manifest.json"
    source_manifest = (
        json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if source_manifest_path.is_file()
        else {}
    )
    source_publishable = bool(source_manifest.get("publishable", True))
    source_website_import = bool(source_manifest.get("website_import", True))
    for sample in samples:
        sample["publishable"] = bool(
            sample.get("publishable", source_publishable)
        ) and source_publishable
        sample["website_import"] = bool(
            sample.get("website_import", source_website_import)
        ) and source_website_import
    rules_by_axis = load_atomic_rules()
    contract = output_contract(rules_by_axis)
    system_prompt = load_system_prompt()
    hashes = {
        "system_prompt": sha256_text(system_prompt),
        "rules": sha256_json(rules_by_axis),
        "contract": sha256_json(contract),
    }
    judge_lock = verify_judge_lock(model=model, hashes=hashes)
    run_dir = OUTPUT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(run_dir / "rules_snapshot.json", rules_by_axis)
    write_json_atomic(run_dir / "output_contract.json", contract)
    outcomes: list[dict[str, Any]] = []
    started_at = utc_now()
    with ThreadPoolExecutor(max_workers=min(workers, len(samples))) as executor:
        futures = {
            executor.submit(
                run_one,
                sample=sample,
                run_dir=run_dir,
                api_key=api_key,
                model=model,
                timeout=timeout,
                system_prompt=system_prompt,
                rules_by_axis=rules_by_axis,
                contract=contract,
                hashes=hashes,
                force=force,
            ): sample["id"]
            for sample in samples
        }
        for future in as_completed(futures):
            outcome = future.result()
            outcomes.append(outcome)
            print(
                f"RULE_GUIDED_JUDGE {outcome['sample_id']} {outcome['status']} "
                f"reused={outcome['reused']}",
                flush=True,
            )
    counts = Counter(outcome["status"] for outcome in outcomes)
    judging_complete = counts == Counter({"success": len(samples)})
    source_publishable = all(
        bool(sample.get("publishable", True)) for sample in samples
    )
    source_website_import = all(
        bool(sample.get("website_import", True)) for sample in samples
    )
    manifest = {
        "schema_version": 1,
        "release_label": RELEASE_LABEL,
        "publishable": judging_complete and source_publishable,
        "website_import": judging_complete and source_website_import,
        "run_id": run_id,
        "status": (
            "complete" if judging_complete else "incomplete"
        ),
        "started_at": started_at,
        "completed_at": utc_now(),
        "source_run": manifest_path(source_run),
        "judge_model": model,
        "prompt_version": PROMPT_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "system_prompt_path": manifest_path(SYSTEM_PROMPT_PATH),
        "system_prompt_addendum_path": (
            manifest_path(SYSTEM_PROMPT_ADDENDUM_PATH)
            if SYSTEM_PROMPT_ADDENDUM_PATH is not None
            else None
        ),
        "rulebook_path": manifest_path(RULEBOOK_PATH),
        "system_prompt_sha256": hashes["system_prompt"],
        "rules_sha256": hashes["rules"],
        "output_contract_sha256": hashes["contract"],
        "judge_lock_path": manifest_path(JUDGE_LOCK_PATH),
        "judge_lock_status": judge_lock["status"],
        "requested_samples": len(samples),
        "logical_judge_calls": len(outcomes),
        "judge_calls_this_invocation": sum(not item["reused"] for item in outcomes),
        "reused_results": sum(item["reused"] for item in outcomes),
        "workers": workers,
        "counts": dict(sorted(counts.items())),
        "samples": sorted(outcomes, key=lambda item: item["sample_id"]),
    }
    write_json_atomic(run_dir / "manifest.json", manifest)
    return manifest, run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--source-run", type=Path, default=SOURCE_RUN)
    parser.add_argument(
        "--sample-ids",
        default="",
        help="Comma-separated sample ids to run (default: all in source-run)",
    )
    return parser.parse_args()


def parse_sample_ids(raw: str) -> list[str] | None:
    if not raw.strip():
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def main() -> int:
    args = parse_args()
    load_dotenv(args.env_file)
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key and not args.model.strip().lower().startswith("local-"):
        raise SystemExit(f"OPENROUTER_API_KEY not found (checked {args.env_file})")
    manifest, _ = run_batch(
        api_key=api_key,
        run_id=args.run_id,
        model=args.model,
        workers=args.workers,
        timeout=args.timeout,
        force=args.force,
        source_run=args.source_run,
        sample_ids=parse_sample_ids(args.sample_ids),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
