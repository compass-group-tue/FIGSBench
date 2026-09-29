"""Assistant system-prompt presets for evaluation."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASELINE = "baseline"
EXTREME_COLD = "extreme-cold"
FACTUAL_V2 = "factual-v2"
LOCKED_FACTUAL_V2_PATH = (
    PROJECT_ROOT
    / "final500_ds_gated"
    / "prompts"
    / "assistant"
    / "factual_v2.txt"
)


def resolve_assistant_prompt(spec: str) -> str:
    """Resolve "baseline" | "extreme-cold" | "factual-v2" | file path to prompt text."""
    lowered = spec.strip().lower()
    if lowered == BASELINE:
        from rule_guided_benchmark.io_utils import ASSISTANT_PROMPT_PATH

        return ASSISTANT_PROMPT_PATH.read_text(encoding="utf-8")
    if lowered == EXTREME_COLD:
        from archetype_guided_benchmark.run_cv_v28_archetype import (
            COLD_ASSISTANT_PROMPT_PATH,
        )

        return COLD_ASSISTANT_PROMPT_PATH.read_text(encoding="utf-8")
    if lowered == FACTUAL_V2:
        return LOCKED_FACTUAL_V2_PATH.read_text(encoding="utf-8")
    path = Path(spec).expanduser()
    if not path.is_file():
        raise ValueError(
            f"unknown assistant prompt {spec!r}: use baseline, "
            "extreme-cold, factual-v2, or a prompt file path"
        )
    return path.read_text(encoding="utf-8")
