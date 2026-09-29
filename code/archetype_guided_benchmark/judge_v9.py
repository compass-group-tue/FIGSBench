"""Frozen appendix-dual-judge-v9 batch helper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rule_guided_judging import run as judge_run
from rule_guided_judging.v9 import configure_base_runner, verify_v9_lock

DEFAULT_JUDGE_MODEL = "local-glm-5.3-flash"
DEFAULT_JUDGE_RUN_SUFFIX = "judge-v9"


def run_v9_judge(
    *,
    api_key: str,
    source_run: Path,
    judge_run_id: str,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    workers: int = 10,
    timeout: float = 180.0,
    force: bool = False,
) -> tuple[dict[str, Any], Path]:
    verify_v9_lock(model=judge_model)
    configure_base_runner()
    return judge_run.run_batch(
        api_key=api_key,
        run_id=judge_run_id,
        model=judge_model,
        workers=workers,
        timeout=timeout,
        force=force,
        source_run=source_run,
    )
