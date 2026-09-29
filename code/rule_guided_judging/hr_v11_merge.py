"""Merge v11 sycophancy and CV result dirs into one dual-axis run for scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rule_guided_judging import run as base
from rule_guided_judging.hr_v11_common import MERGED_PROMPT_VERSION, VALIDATOR_VERSION

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_result(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_sample(
    *,
    syc_result: dict[str, Any],
    cv_result: dict[str, Any],
) -> dict[str, Any]:
    sample_id = syc_result["sample_id"]
    if cv_result.get("sample_id") != sample_id:
        raise ValueError(f"sample_id mismatch: {sample_id} vs {cv_result.get('sample_id')}")
    if syc_result.get("status") != "success":
        raise ValueError(f"{sample_id} syc status={syc_result.get('status')}")
    if cv_result.get("status") != "success":
        raise ValueError(f"{sample_id} cv status={cv_result.get('status')}")
    syc_j = syc_result["judgment"]
    cv_j = cv_result["judgment"]
    judgment = {
        "rubric_version": syc_j.get("rubric_version") or cv_j.get("rubric_version"),
        "sycophancy": syc_j["sycophancy"],
        "calibrated_validation": cv_j["calibrated_validation"],
        "confidence": "medium",
        "needs_adjudication": bool(syc_j.get("needs_adjudication") or cv_j.get("needs_adjudication")),
        "adjudication_reasons": list(syc_j.get("adjudication_reasons") or [])
        + list(cv_j.get("adjudication_reasons") or []),
        "axis_scored": "both",
    }
    return {
        "schema_version": 1,
        "release_label": "v2",
        "publishable": False,
        "website_import": False,
        "sample_id": sample_id,
        "judge_model": syc_result.get("judge_model"),
        "prompt_version": MERGED_PROMPT_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "judge_reasoning_effort": syc_result.get("judge_reasoning_effort"),
        "syc_prompt_version": syc_result.get("prompt_version"),
        "cv_prompt_version": cv_result.get("prompt_version"),
        "syc_run_sample": syc_result.get("sample_id"),
        "status": "success",
        "judgment": judgment,
        "normalizations": [],
        "syc_raw_response_path": syc_result.get("raw_response_path"),
        "cv_raw_response_path": cv_result.get("raw_response_path"),
        "raw_response_path": syc_result.get("raw_response_path"),
        "completed_at": base.utc_now(),
    }


def merge_runs(*, syc_run: Path, cv_run: Path, out_run: Path) -> dict[str, Any]:
    syc_dir = syc_run / "results"
    cv_dir = cv_run / "results"
    out_results = out_run / "results"
    out_results.mkdir(parents=True, exist_ok=True)
    syc_ids = {path.stem for path in syc_dir.glob("*.json")}
    cv_ids = {path.stem for path in cv_dir.glob("*.json")}
    shared = sorted(syc_ids & cv_ids)
    missing_syc = sorted(cv_ids - syc_ids)
    missing_cv = sorted(syc_ids - cv_ids)
    merged = 0
    skipped: list[str] = []
    for sample_id in shared:
        syc_result = load_result(syc_dir / f"{sample_id}.json")
        cv_result = load_result(cv_dir / f"{sample_id}.json")
        try:
            combined = merge_sample(syc_result=syc_result, cv_result=cv_result)
        except ValueError:
            skipped.append(sample_id)
            continue
        base.write_json_atomic(out_results / f"{sample_id}.json", combined)
        merged += 1
    manifest = {
        "schema_version": 1,
        "run_id": out_run.name,
        "status": "complete" if not skipped and not missing_syc and not missing_cv else "partial",
        "prompt_version": MERGED_PROMPT_VERSION,
        "syc_run": str(syc_run).replace("\\", "/"),
        "cv_run": str(cv_run).replace("\\", "/"),
        "merged": merged,
        "skipped_invalid": skipped,
        "missing_from_syc": missing_syc,
        "missing_from_cv": missing_cv,
    }
    base.write_json_atomic(out_run / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--syc-run", type=Path, required=True)
    parser.add_argument("--cv-run", type=Path, required=True)
    parser.add_argument("--out-run", type=Path, required=True)
    args = parser.parse_args()
    manifest = merge_runs(
        syc_run=args.syc_run,
        cv_run=args.cv_run,
        out_run=args.out_run,
    )
    print(json.dumps(manifest, indent=2))
    if manifest["skipped_invalid"] or manifest["missing_from_syc"] or manifest["missing_from_cv"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
