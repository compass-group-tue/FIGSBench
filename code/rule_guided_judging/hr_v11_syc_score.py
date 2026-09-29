"""Score-only first pass for sycophancy (HR v11). No rule tags."""

from __future__ import annotations

from rule_guided_judging import run as base
from rule_guided_judging.hr_v11_common import (
    SCORE_PROMPT_VERSION,
    SYC_JUDGE_MODEL,
    configure_axis_runner,
)

PROMPT_VERSION = SCORE_PROMPT_VERSION


def configure_base_runner() -> None:
    configure_axis_runner("sycophancy", score_only=True)
    base.PROMPT_VERSION = PROMPT_VERSION
    base.DEFAULT_MODEL = SYC_JUDGE_MODEL


def main() -> int:
    configure_base_runner()
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
