"""Locked hr-v11 dual-axis judge pipeline for generated runs.

Runs, in order, all with the frozen paper prompts:
  syc score-only (syc-score-v1) -> CV score-only (cv-score-v1) ->
  syc rules (syc-rules-v1, conditioned on locked syc scores) ->
  CV rules (cv-rules-v1, conditioned on locked CV scores) ->
  merged dual-axis run.

Used by ``archetype_guided_benchmark`` ``--run-judge``. Set
``JUDGE_MAX_TOKENS=64000`` in the environment: hard samples need
>32k reasoning room and return empty content on smaller budgets.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from rule_guided_judging import run as base
from rule_guided_judging import hr_v11_common as common
from rule_guided_judging import hr_v11_cv_rules as cv_rules
from rule_guided_judging import hr_v11_merge as merge
from rule_guided_judging import hr_v11_syc_rules as syc_rules

DEFAULT_JUDGE_RUN_SUFFIX = "judge-hrv11"
# Default OpenRouter judge for `--judge api`. The paper's reported numbers use the local
# GLM-5.3-Flash judge (`--judge local`); any API judge is a different protocol, and its
# results are labelled `paper_judge: false`. Pass `judge_model="z-ai/glm-5.3-flash"` for GLM
# through OpenRouter.
API_JUDGE_MODEL = "google/gemini-3.8-flash"
PAPER_JUDGE_MODEL = "local-glm-5.3-flash"
JUDGE_TIMEOUT = 900.0


def _run_stage(
    *,
    api_key: str,
    run_id: str,
    model: str,
    workers: int,
    source_run: Path,
) -> dict[str, Any]:
    manifest, _ = base.run_batch(
        api_key=api_key,
        run_id=run_id,
        model=model,
        workers=workers,
        timeout=JUDGE_TIMEOUT,
        force=False,
        source_run=source_run,
    )
    return manifest


def run_locked_judge(
    *,
    api_key: str,
    source_run: Path,
    judge_run_id: str,
    judge_model: str,
    workers: int,
) -> dict[str, Any]:
    """Run the full locked score+rules pipeline; return the merged manifest."""
    common.configure_axis_runner("sycophancy", score_only=True)
    syc_score_id = f"{judge_run_id}-syc-score"
    manifest = _run_stage(
        api_key=api_key,
        run_id=syc_score_id,
        model=judge_model,
        workers=workers,
        source_run=source_run,
    )
    if manifest.get("status") != "complete":
        return manifest
    common.configure_axis_runner("calibrated_validation", score_only=True)
    cv_score_id = f"{judge_run_id}-cv-score"
    manifest = _run_stage(
        api_key=api_key,
        run_id=cv_score_id,
        model=judge_model,
        workers=workers,
        source_run=source_run,
    )
    if manifest.get("status") != "complete":
        return manifest
    syc_rules.load_locked_scores(base.OUTPUT_ROOT / syc_score_id)
    syc_rules.configure_rules_runner()
    syc_rules_id = f"{judge_run_id}-syc-rules"
    manifest = _run_stage(
        api_key=api_key,
        run_id=syc_rules_id,
        model=judge_model,
        workers=workers,
        source_run=source_run,
    )
    if manifest.get("status") != "complete":
        return manifest
    cv_rules.load_locked_scores(base.OUTPUT_ROOT / cv_score_id)
    cv_rules.configure_rules_runner()
    cv_rules_id = f"{judge_run_id}-cv-rules"
    manifest = _run_stage(
        api_key=api_key,
        run_id=cv_rules_id,
        model=judge_model,
        workers=workers,
        source_run=source_run,
    )
    if manifest.get("status") != "complete":
        return manifest
    out_run = base.OUTPUT_ROOT / judge_run_id
    merged = merge.merge_runs(
        syc_run=base.OUTPUT_ROOT / syc_rules_id,
        cv_run=base.OUTPUT_ROOT / cv_rules_id,
        out_run=out_run,
    )
    results = sorted((out_run / "results").glob("*.json"))
    counts = Counter(
        json.loads(path.read_text(encoding="utf-8")).get("status")
        for path in results
    )
    merged["counts"] = dict(counts)
    merged["stages"] = {
        "syc_score": syc_score_id,
        "cv_score": cv_score_id,
        "syc_rules": syc_rules_id,
        "cv_rules": cv_rules_id,
    }
    (out_run / "manifest.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return merged
